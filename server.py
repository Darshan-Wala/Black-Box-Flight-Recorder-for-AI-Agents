"""Vercel entrypoint (server.py, auto-detected): the whole Black Box app as ONE FastAPI function.

Locally the project runs as three processes (run_all.py: dashboard :8501, replay API :8000,
diagnosis API :8001). A serverless deployment has a single process and no fixed ports, so this
file mounts all three apps into one and tells the dashboard to call diagnose / explain / replay
in-process (BLACKBOX_INPROCESS=1) instead of over HTTP.

    /                    dashboard UI   (blackbox_part 1/ui_api.py + ui/index.html)
    /api/...             dashboard JSON API
    /replay-api/...      Part 1 replay API      docs: /replay-api/docs
    /diagnose-api/...    Part 2 diagnosis API   docs: /diagnose-api/docs
    /healthz             liveness check

Storage: Vercel's filesystem is read-only except /tmp, so the SQLite run store lives in
/tmp/blackbox and is seeded from the recorded runs in "blackbox_part 1/runs/" on a cold start.
Anything recorded on the deployment (new runs, replays, confirmed causes) lasts only as long as
that server instance: fine for a demo, not a permanent database.

Run it locally the same way Vercel does:
    pip install -r requirements.txt
    uvicorn server:app --port 3000             # then open http://localhost:3000
"""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PART1 = ROOT / "blackbox_part 1"
ON_VERCEL = bool(os.environ.get("VERCEL"))

# ---------------------------------------------------------------- environment (before imports)
os.environ.setdefault("BLACKBOX_INPROCESS", "1")
_data_dir = os.environ.get("BLACKBOX_DATA_DIR") or ("/tmp/blackbox" if ON_VERCEL else "")
if _data_dir:
    DATA = Path(_data_dir)
    os.environ.setdefault("BLACKBOX_DB", str(DATA / "blackbox.db"))
    os.environ.setdefault("BLACKBOX_RUNS_DIR", str(DATA / "runs"))
    os.environ.setdefault("BLACKBOX_REPLAYS_DIR", str(DATA / "replays"))
    os.environ.setdefault("BLACKBOX_CONFIRMED_DIR", str(DATA / "confirmed"))
if ON_VERCEL:
    # no Ollama on Vercel: offer the offline mock models plus any hosted model whose key is set
    _models = ["mock-large", "mock-small"]
    if os.environ.get("GEMINI_API_KEY"):
        _models.append("gemini:" + os.environ.get("BLACKBOX_GEMINI_MODEL", "gemini-2.5-flash"))
    if os.environ.get("GROQ_API_KEY"):
        _models.append("groq:" + os.environ.get("BLACKBOX_GROQ_MODEL", "llama-3.3-70b-versatile"))
    os.environ.setdefault("BLACKBOX_MODELS", ",".join(_models))
    os.environ.setdefault("BLACKBOX_OLLAMA_HOST", "http://127.0.0.1:9")  # closed port: fails instantly

for p in (str(ROOT), str(PART1)):          # Part 1 first: its modules are imported by bare name
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

import store  # noqa: E402  (Part 1 run store; reads BLACKBOX_DB set above)

# ---------------------------------------------------------------- seed the run store
_seed_lock = threading.Lock()
SEED_INFO: dict = {}


def seed_store() -> None:
    """Build the SQLite store from the recorded run files if it does not exist yet.
    Writes to a temporary file and renames it, so a half-built database is never used."""
    with _seed_lock:
        if store.DB_PATH.exists():
            SEED_INFO.setdefault("seeded", False)
            return
        t0 = time.time()
        store.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = store.DB_PATH.with_name(f".seed-{os.getpid()}.db")
        tmp.unlink(missing_ok=True)
        files = sorted((PART1 / "runs").glob("run_*.json")) + sorted((PART1 / "replays").glob("run_*.json"))
        n = 0
        for f in files:                     # originals first, so replays find their parent run
            try:
                store.import_json(f, db=tmp)
                n += 1
            except Exception as e:  # noqa: BLE001  one unreadable file must not stop the app
                print(f"[blackbox] skipped {f.name}: {type(e).__name__}: {e}")
        os.replace(tmp, store.DB_PATH)
        SEED_INFO.update(seeded=True, runs=n, seconds=round(time.time() - t0, 2))
        print(f"[blackbox] seeded {n} runs into {store.DB_PATH} in {SEED_INFO['seconds']}s")


seed_store()

# ---------------------------------------------------------------- one app, three services
from part2 import diagnose_api  # noqa: E402
import replay_api  # noqa: E402
import ui_api  # noqa: E402

app = ui_api.app
app.title = "Black Box"
app.description = "Flight recorder, debugger and recovery system for AI agents."


@app.get("/healthz", include_in_schema=False)
def healthz():
    return {"ok": True, "vercel": ON_VERCEL, "db": str(store.DB_PATH), "seed": SEED_INFO,
            "model": type(diagnose_api.booster()).__module__}


app.mount("/replay-api", replay_api.app)
app.mount("/diagnose-api", diagnose_api.app)
