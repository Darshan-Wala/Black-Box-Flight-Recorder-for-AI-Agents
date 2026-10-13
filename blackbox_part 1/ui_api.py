"""Black Box dashboard (Part 1): serves ui/index.html and the JSON it needs.  Port 8501.

  uvicorn ui_api:app --port 8501

Nothing here changes the frozen contract: it only reads the run store and calls the same
clients (diagnose / explain / replay) the Streamlit UI (app.py) uses.
"""
from __future__ import annotations

import concurrent.futures
import contextvars
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

import apikeys
import clients
import custom
import inject
import replay
import runner
import store
import tasks

ROOT = Path(__file__).resolve().parent
UI_DIR = ROOT / "ui"
CONFIRMED_DIR = Path(os.environ.get("BLACKBOX_CONFIRMED_DIR", store.ROOT / "confirmed"))
RESULTS_MD = Path(os.environ.get("BLACKBOX_RESULTS_MD", ROOT.parent / "part2" / "results.md"))
# models offered in the "record a run" picker; BLACKBOX_MODELS (comma-separated) overrides it
MODELS = [m.strip() for m in os.environ.get("BLACKBOX_MODELS", "mock-large,mock-small,llama3,mistral").split(",")
          if m.strip()]
MOCK_TTL = 30  # seconds a built-in-mock diagnosis is kept before Part 2's API is tried again

app = FastAPI(title="Black Box - dashboard (Part 1)", version="1.0")
app.add_middleware(apikeys.KeysMiddleware)   # visitor's own API keys, scoped to each request
_diag: dict[str, tuple[float, dict]] = {}
_replay_cache: dict[str, Any] = {"n": -1, "rows": []}


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(UI_DIR / "index.html", headers={"Cache-Control": "no-store"})


def _up(url: str) -> bool:
    if clients.INPROCESS:      # Vercel: diagnose and replay run inside this process
        return True
    try:
        return requests.get(f"{url}/health", timeout=0.6).ok
    except Exception:  # noqa: BLE001
        return False


@app.get("/api/meta")
def meta():
    return {
        "tasks": [{"task_id": t.task_id, "text": t.text, "demo": t.task_id in tasks.DEMO_TASK_IDS} for t in tasks.TASKS],
        "models": MODELS, "frameworks": ["custom", "langgraph"], "fault_types": inject.FAULT_TYPES,
        "diagnose_url": "in-process" if clients.INPROCESS else clients.DIAGNOSE_URL,
        "replay_url": "in-process" if clients.INPROCESS else clients.REPLAY_URL,
        "services": {"diagnose": _up(clients.DIAGNOSE_URL), "replay": _up(clients.REPLAY_URL)},
        "explain_llm": os.environ.get("BLACKBOX_EXPLAIN_LLM", ""),
        "key_providers": [{"prefix": p, "key_env": e, "server_key": bool(os.environ.get(e)),
                           "user_key": apikeys.has_user_key(p)} for p, e in apikeys.KEY_ENV.items()],
    }


# ------------------------------------------------------------------ bring your own key
_NOT_CHAT = re.compile(r"embed|imagen|veo|tts|whisper|dall-e|moderation|transcri|audio|realtime|image|aqa|"
                       r"computer-use|search-preview|babbage|davinci", re.I)


class KeyCheckReq(BaseModel):
    provider: str
    key: str
    base_url: Optional[str] = None


@app.post("/api/keys/check")
def check_key(req: KeyCheckReq):
    """Test a visitor's key against the provider and list the chat models it can use.
    The key is used for this one call and not stored."""
    import llm as llm_mod
    if req.provider not in llm_mod.PROVIDERS:
        raise HTTPException(422, f"unknown provider `{req.provider}`")
    try:
        key = apikeys._clean_key(req.key)
        base = llm_mod.PROVIDERS[req.provider][0] or apikeys.check_base_url(req.base_url or "")
    except apikeys.KeysError as e:
        raise HTTPException(422, str(e))
    if not key:
        raise HTTPException(422, "Paste a key first.")
    auth = {"Authorization": f"Bearer {key}"}
    try:
        if req.provider == "openrouter":     # its model list is public, so check the key separately
            k = requests.get("https://openrouter.ai/api/v1/key", headers=auth, timeout=10)
            if k.status_code >= 400:
                return {"ok": False, "error": f"The provider rejected this key (HTTP {k.status_code}).", "models": []}
        r = requests.get(f"{base.rstrip('/')}/models", headers=auth, timeout=15)
    except requests.RequestException as e:
        return {"ok": False, "error": f"Could not reach the provider ({type(e).__name__}).", "models": []}
    if r.status_code in (401, 403):
        return {"ok": False, "error": f"The provider rejected this key (HTTP {r.status_code}).", "models": []}
    if r.status_code >= 400:
        return {"ok": False, "error": f"The provider answered HTTP {r.status_code}.", "models": []}
    try:
        data = r.json()
        items = data.get("data") or data.get("models") or []
        ids = {str(m.get("id") or m.get("name") or "").removeprefix("models/") for m in items if isinstance(m, dict)}
    except Exception:  # noqa: BLE001
        return {"ok": True, "error": "Key accepted, but the model list could not be read. Type a model id yourself.",
                "models": []}
    models = sorted(m for m in ids if m and not _NOT_CHAT.search(m))
    return {"ok": True, "models": models[:500], "error": None}


def _confirmed_ids() -> set[str]:
    return {p.name.split("_step")[0] for p in CONFIRMED_DIR.glob("*_step*.json")} if CONFIRMED_DIR.exists() else set()


@app.get("/api/runs")
def list_runs():
    """Every recorded run (not replays) with the per-run totals the activity feed shows."""
    with store._conn() as c:
        agg = {r[0]: r[1:] for r in c.execute(
            "SELECT run_id, COUNT(*), SUM(tokens), SUM(error IS NOT NULL) FROM steps GROUP BY run_id")}
        rep = {r[0]: r[1:] for r in c.execute(
            "SELECT parent_run_id, COUNT(*), SUM(outcome='success') FROM runs "
            "WHERE parent_run_id IS NOT NULL GROUP BY parent_run_id")}
        rows = c.execute("SELECT * FROM runs WHERE parent_run_id IS NULL ORDER BY created_at DESC, run_id DESC").fetchall()
    confirmed = _confirmed_ids()
    out = []
    for r in rows:
        m = json.loads(r["meta"] or "{}")
        n_steps, tokens, n_err = agg.get(r["run_id"], (0, 0, 0))
        n_rep, n_ok = rep.get(r["run_id"], (0, 0))
        out.append({
            "run_id": r["run_id"], "framework": r["framework"], "model": r["model"], "task": r["task"],
            "task_id": r["task_id"], "outcome": r["outcome"], "faulty_step": r["faulty_step"],
            "created_at": r["created_at"], "fault_type": (m.get("fault") or {}).get("type"),
            "final_answer": m.get("final_answer"), "n_steps": n_steps, "tokens": tokens or 0,
            "n_errors": n_err or 0, "n_replays": n_rep, "n_replays_ok": n_ok or 0,
            "confirmed": r["run_id"] in confirmed,
            "custom": (m.get("custom") or {}).get("mode"),
        })
    return out


@app.get("/api/replays")
def list_replays():
    """One row per saved replay, for the overview charts (tokens saved, fix rate)."""
    with store._conn() as c:
        n = c.execute("SELECT COUNT(*) FROM runs WHERE parent_run_id IS NOT NULL").fetchone()[0]
        if n == _replay_cache["n"]:
            return _replay_cache["rows"]
        toks: dict[str, dict[str, int]] = {}
        for run_id, step_no, t in c.execute(
                "SELECT s.run_id, s.step_no, s.tokens FROM steps s JOIN runs r ON r.run_id = s.run_id "
                "WHERE r.parent_run_id IS NOT NULL"):
            toks.setdefault(run_id, {})[str(step_no)] = t or 0
        parents = {r[0]: r[1] for r in c.execute("SELECT run_id, outcome FROM runs WHERE parent_run_id IS NULL")}
        rows = []
        for r in c.execute("SELECT run_id, parent_run_id, outcome, created_at, meta FROM runs WHERE parent_run_id IS NOT NULL"):
            rp = json.loads(r["meta"] or "{}").get("replay") or {}
            t = toks.get(r["run_id"], {})
            total = sum(t.values())
            reused = sum(v for k, v in t.items() if rp.get("status", {}).get(k) == "reused")
            rows.append({"run_id": r["run_id"], "parent": r["parent_run_id"], "outcome": r["outcome"],
                         "parent_outcome": parents.get(r["parent_run_id"]), "created_at": r["created_at"],
                         "edited_step": rp.get("edited_step"),
                         "tokens_saved_pct": round(100 * reused / total, 1) if total else 0})
    _replay_cache.update(n=n, rows=rows)
    return rows


def _run_or_404(run_id: str) -> dict:
    r = store.get_run(run_id)
    if not r:
        raise HTTPException(404, f"run {run_id} not found")
    return r


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    run = _run_or_404(run_id)
    task = tasks.get_task(run.get("task_id") or "") or tasks.get_task_by_text(run["task"])
    with store._conn() as c:
        reps = []
        for r in c.execute("SELECT run_id, outcome, created_at, model, meta FROM runs WHERE parent_run_id=? "
                           "ORDER BY created_at DESC, run_id DESC", (run_id,)):
            rp = json.loads(r["meta"] or "{}").get("replay") or {}
            reps.append({"run_id": r["run_id"], "outcome": r["outcome"], "created_at": r["created_at"],
                         "model": r["model"], "edited_step": rp.get("edited_step"), "edit": rp.get("edit")})
    exp = task.expected if task else (run.get("custom") or {}).get("expected")
    return {"run": run, "expected": tasks.fmt_value(exp) if exp is not None else None, "replays": reps,
            "confirmed": run_id in _confirmed_ids()}


def _diagnose(run: dict, refresh: bool = False) -> dict:
    rid = run["run_id"]
    hit = _diag.get(rid)
    if hit and not refresh and (not hit[1]["mock"] or time.time() - hit[0] < MOCK_TTL):
        return hit[1]
    data, src = clients.diagnose(run)
    suspects = data.get("suspects", [])
    actors = {s["step_no"]: s["actor"] for s in run["steps"]}
    for s in suspects:
        s["actor"] = actors.get(s["step_no"])
    res = {"suspects": suspects, "source": src, "mock": src.startswith("built-in mock"),
           "faulty_step": run.get("faulty_step"),
           "match": (suspects[0]["step_no"] == run["faulty_step"]) if suspects and run.get("faulty_step") else None}
    _diag[rid] = (time.time(), res)
    return res


@app.get("/api/runs/{run_id}/diagnosis")
def get_diagnosis(run_id: str, refresh: bool = False):
    return _diagnose(_run_or_404(run_id), refresh)


@app.get("/api/runs/{run_id}/rewind/{step_no}")
def get_rewind(run_id: str, step_no: int):
    try:
        rw = replay.rewind(_run_or_404(run_id), step_no)
    except replay.ReplayError as e:
        raise HTTPException(422, str(e))
    return {"will_rerun": rw["will_rerun"], "will_reuse": rw["will_reuse"], "n_before": len(rw["before"]), "state": rw["state"]}


@app.get("/api/compare/{a_id}/{b_id}")
def get_compare(a_id: str, b_id: str):
    return replay.compare(_run_or_404(a_id), _run_or_404(b_id))


class ExplainReq(BaseModel):
    run_id: str
    llm: Optional[str] = None


@app.post("/api/explain")
def post_explain(req: ExplainReq):
    return clients.explain(_run_or_404(req.run_id), (req.llm or "").strip() or None)


class ReplayReq(BaseModel):
    run_id: str
    step_no: int
    edit: Optional[dict[str, Any]] = None
    n_runs: int = 3
    branch: bool = False
    model: Optional[str] = None


@app.post("/api/replay")
def post_replay(req: ReplayReq):
    _run_or_404(req.run_id)
    try:
        data, src = clients.replay(req.run_id, req.step_no, req.edit, n_runs=req.n_runs, branch=req.branch,
                                   model=(req.model or "").strip() or None)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(422, str(e))
    return {**data, "source": src}


class RecordReq(BaseModel):
    task_id: str
    model: str = "mock-large"
    framework: str = "custom"
    fault_type: Optional[str] = None
    fault_step: Optional[int] = None


@app.post("/api/record")
def post_record(req: RecordReq):
    t = tasks.get_task(req.task_id)
    if not t:
        raise HTTPException(404, f"task {req.task_id} not found")
    fault = None
    if req.fault_type and req.fault_type != "none":
        fault = {"type": req.fault_type, "step_no": req.fault_step or None, "seed": 1}
    try:
        r = runner.record_run(t, req.model.strip(), req.framework, fault=fault)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(422, f"Run failed: {e}")
    return {"run_id": r["run_id"], "outcome": r["outcome"]}


class ConfirmReq(BaseModel):
    run_id: str
    step_no: int
    fixed_run_id: str
    edit: Optional[dict[str, Any]] = None
    model: Optional[str] = None


@app.post("/api/confirm")
def post_confirm(req: ConfirmReq):
    """Save a replay-confirmed cause (same file the Streamlit UI writes: training data + regression test)."""
    run = _run_or_404(req.run_id)
    CONFIRMED_DIR.mkdir(parents=True, exist_ok=True)
    p = CONFIRMED_DIR / f"{run['run_id']}_step{req.step_no}.json"
    p.write_text(json.dumps({"run_id": run["run_id"], "confirmed_step": req.step_no,
                             "edit": req.edit or {"branch": req.model}, "fixed_run_id": req.fixed_run_id,
                             "suspects": _diagnose(run)["suspects"], "run": run}, indent=2), encoding="utf-8")
    return {"saved": p.name}


# ------------------------------------------------------------------ Custom runs
CUSTOM_TIMEOUT = int(os.environ.get("BLACKBOX_CUSTOM_TIMEOUT", "300"))   # seconds for one custom run
_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)


@app.get("/api/custom/meta")
def custom_meta():
    """What a custom run can use right now: runnable models, frameworks, fault types, searchable facts."""
    return {**custom.supported_models(), "frameworks": ["custom", "langgraph"],
            "fault_types": [{"type": t, "help": custom.FAULT_HELP.get(t, "")} for t in inject.FAULT_TYPES],
            "knowledge": custom.knowledge(), "web_search": custom.web_enabled(), "max_query": custom.MAX_QUERY, "timeout_s": CUSTOM_TIMEOUT,
            "examples": [t.text for t in tasks.TASKS if t.task_id in ("t01", "t11", "t16", "t23")]}


class CustomRunReq(BaseModel):
    query: str
    model: str
    framework: str = "custom"
    mode: str = "normal"                 # "normal" | "fault"
    fault_type: Optional[str] = None
    fault_step: Optional[int] = None     # empty = the first step the fault fits
    expected: Optional[float] = None     # optional known answer, used to judge the outcome


def _custom_summary(run: dict, reference: Optional[dict] = None) -> dict:
    c, f = run.get("custom") or {}, run.get("fault")
    notes = []
    if f and not f.get("applied"):
        notes.append("The fault was not applied: the run had no step this fault type fits"
                     + (f" at step {f.get('step_no')}." if f.get("step_no") else "."))
    elif f and run["outcome"] == "success":
        notes.append("The fault was applied but the run still succeeded, so there is no failure to diagnose.")
    missed = custom.lookups_missed(run)
    live = c.get("search") == "live"
    if not any(s["kind"] == "tool_call" for s in run["steps"]):
        notes.append("The model did not produce a usable plan, so nothing was looked up. Ask for something that "
                     "can be worked out from numbers, for example a total, a difference or an average.")
    if missed and live:
        notes.append(f"{missed} lookup(s) came back empty: the model had no answer for them (or its call failed).")
    elif missed:
        notes.append(f"{missed} search step(s) found nothing. This run used the local fact table; "
                     "see \"How lookups work\".")
    if live:
        src = {v.get("source") for v in (c.get("lookups") or {}).values()}
        parts = ([] if not src & {"wikidata", "worldbank"} else [
                     " and ".join(n for k, n in (("worldbank", "the World Bank"), ("wikidata", "Wikidata")) if k in src)
                     + " (latest dated value on record)"]) + \
                ([] if "wikipedia+model" not in src else ["the model reading a Wikipedia extract"]) + \
                ([] if "model" not in src else ["the model's own knowledge, which can be out of date"])
        if parts:
            notes.append("Lookups in this run came from " + ", ".join(parts) + ". "
                         + ("" if src <= {"wikidata", "worldbank"} else "Model answers are not verified facts."))
    if c.get("judged_by") == "completed":
        notes.append("No expected answer was available, so the outcome only says whether the run finished with a "
                     "number and no step error. It does not say the answer is correct.")
    if reference is not None and c.get("expected_source") != "reference_run":
        notes.append(f"The fault-free reference run {reference['run_id']} did not produce a usable answer, "
                     "so it could not be used as the expected answer.")
    return {
        "run_id": run["run_id"], "query": run["task"], "model": run["model"], "framework": run["framework"],
        "mode": c.get("mode"), "outcome": run["outcome"], "final_answer": run.get("final_answer"),
        "expected": tasks.fmt_value(c["expected"]) if c.get("expected") is not None else None,
        "expected_source": c.get("expected_source"), "judged_by": c.get("judged_by"),
        "fault": ({"type": f["type"], "requested_step": None if f.get("applied") else f.get("step_no"),
                   "applied": bool(f.get("applied")), "step_no": f.get("step_no") if f.get("applied") else None}
                  if f else None),
        "faulty_step": run.get("faulty_step"), "reference_run_id": c.get("reference_run_id"),
        "n_steps": len(run["steps"]), "tokens": sum(s["tokens"] for s in run["steps"]),
        "duration_ms": round(sum(t["ms"] for t in run.get("timing") or []), 1),
        "created_at": run.get("created_at"), "notes": notes,
        "search": c.get("search") or "fact_table",
        "lookups": [{"query": q, **v} for q, v in (c.get("lookups") or {}).items()],
    }


@app.post("/api/custom/run")
def custom_run(req: CustomRunReq):
    """Run the user's query through the normal agent + recorder and save it like any other run."""
    if req.mode not in ("normal", "fault"):
        raise HTTPException(422, "mode must be `normal` or `fault`")
    try:
        # a normal run never carries a fault, whatever else the request contains
        fault = custom.check_fault(req.fault_type, req.fault_step) if req.mode == "fault" else None
        custom.check_model(req.model, (req.query or "").strip())
    except custom.CustomError as e:
        raise HTTPException(422, str(e))
    # copy_context: the worker thread must see this request's API keys
    job = _pool.submit(contextvars.copy_context().run, custom.record_custom, req.query, req.model.strip(), req.framework, fault, req.expected)
    try:
        res = job.result(timeout=CUSTOM_TIMEOUT)
    except concurrent.futures.TimeoutError:
        raise HTTPException(504, f"The run is taking longer than {CUSTOM_TIMEOUT} seconds. It keeps going in the "
                                 "background and will appear in the run history if it finishes.")
    except custom.CustomError as e:
        raise HTTPException(422, str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"The run crashed before it could be recorded: {type(e).__name__}: {e}")
    return _custom_summary(res["run"], res["reference"])


@app.get("/api/custom/runs/{run_id}")
def custom_get(run_id: str):
    run = _run_or_404(run_id)
    if not run.get("custom"):
        raise HTTPException(404, f"{run_id} is not a custom run")
    return _custom_summary(run)


class VerifyReq(BaseModel):
    run_id: str
    n_runs: int = 1


@app.post("/api/custom/verify")
def custom_verify(req: VerifyReq):
    """Verify loop for one run: replay each suspect in order with a plain retry of that step (the
    same generic fix Part 2's verify loop falls back to) through the existing /replay. The first
    suspect whose replay flips the run to success is the verified cause."""
    run = _run_or_404(req.run_id)
    if run["outcome"] != "fail":
        raise HTTPException(422, "This run succeeded, so there is no failure to verify.")
    diag = _diagnose(run)
    attempts, confirmed = [], None
    for s in diag["suspects"]:
        try:
            data, src = clients.replay(run["run_id"], s["step_no"], {"rerun": True}, n_runs=max(1, min(req.n_runs, 5)))
        except Exception as e:  # noqa: BLE001
            attempts.append({"step_no": s["step_no"], "error": str(e)})
            continue
        attempts.append({"step_no": s["step_no"], "outcome_flipped": data["outcome_flipped"],
                         "new_outcome": data["new_run"]["outcome"], "new_run_id": data["new_run"]["run_id"],
                         "success_rate": data.get("success_rate"), "steps_rerun": data["steps_rerun"],
                         "steps_reused": data["steps_reused"], "tokens_saved_pct": data["tokens_saved_pct"],
                         "state_restored": data.get("state_restored"), "source": src})
        if data["outcome_flipped"] and data["new_run"]["outcome"] == "success":
            confirmed = s["step_no"]
            break
    return {"run_id": run["run_id"], "verified": confirmed is not None, "confirmed_step": confirmed,
            "attempts": attempts, "diagnosis_source": diag["source"], "mock_diagnosis": diag["mock"],
            "fix_tried": "retry the step as-is", "injected_step": run.get("faulty_step"),
            "matches_injected": (confirmed == run["faulty_step"]) if confirmed and run.get("faulty_step") else None}


def parse_results(text: str) -> dict:
    """results.md -> {title, sections: [{title, text: [..], tables: [{headers, rows, caption}]}]}."""
    title, sections, cur, table, para, caption = "", [], None, None, [], None

    def flush():
        nonlocal para, caption
        if para and cur is not None:
            if caption:
                cur["text"].append(caption)
            caption = None
            joined = " ".join(para)
            if joined.endswith(":"):
                caption = joined[:-1]      # a "Something:" line right above a table is its caption
            else:
                cur["text"].append(joined)
        para = []

    for line in text.splitlines() + [""]:
        s = line.strip()
        if s.startswith("|"):
            cells = [x.strip() for x in s.strip("|").split("|")]
            if table is None:
                flush()
                table, caption = {"headers": cells, "rows": [], "caption": caption}, None
                cur["tables"].append(table)
            elif not re.fullmatch(r"[\s|:-]+", s):
                table["rows"].append(cells)
            continue
        table = None
        if s.startswith("# "):
            title = s[2:]
            cur = {"title": "", "text": [], "tables": []}
            sections.append(cur)
        elif s.startswith("## "):
            flush()
            cur = {"title": s[3:], "text": [], "tables": []}
            sections.append(cur)
        elif not s:
            flush()
        elif cur is not None:
            para.append(s)
    return {"title": title, "sections": sections}


@app.get("/api/results")
def get_results():
    if not RESULTS_MD.exists():
        raise HTTPException(404, f"{RESULTS_MD} not found. Run: python -m part2.eval")
    return {**parse_results(RESULTS_MD.read_text(encoding="utf-8")), "path": str(RESULTS_MD),
            "modified": RESULTS_MD.stat().st_mtime}
