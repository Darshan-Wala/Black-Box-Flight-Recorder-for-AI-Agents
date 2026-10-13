"""POST /diagnose: takes a run, returns the top 3 suspect steps.
POST /explain: the same suspects plus a plain-language explanation of the failure.

    uvicorn part2.diagnose_api:app --port 8001
"""
from fastapi import Body, FastAPI, HTTPException

from . import baseline
from . import explain as explainer
from . import model

app = FastAPI(title="Black Box diagnosis")
_booster = None


def booster():
    global _booster
    if _booster is None:
        _booster = model.load()
    return _booster


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/diagnose")
def diagnose(payload: dict = Body(...)):
    run = payload.get("run", payload)  # accepts the run itself or {"run": {...}}
    if not run.get("steps"):
        raise HTTPException(400, "run has no steps")
    return {"run_id": run.get("run_id"), "suspects": model.diagnose(booster(), run)}


@app.post("/explain")
def explain(payload: dict = Body(...)):
    """Body: the run, or {"run": {...}, "llm": "groq:<model>", "api_keys": {"GROQ_API_KEY": "..."}}.
    Without `llm` (and without the BLACKBOX_EXPLAIN_LLM environment variable) the explanation is built
    from the model's evidence. `api_keys` (optional) are used for this request only, never stored."""
    run = payload.get("run", payload)
    if not run.get("steps"):
        raise HTTPException(400, "run has no steps")
    suspects = model.diagnose(booster(), run)
    llm = payload.get("llm") if "run" in payload else None
    keys = payload.get("api_keys") if "run" in payload else None
    keys = {str(k): str(v) for k, v in keys.items() if k.endswith(("_API_KEY", "_BASE_URL")) and v} \
        if isinstance(keys, dict) else {}
    token = baseline.API_KEYS.set(keys)
    try:
        text = explainer.explain(run, suspects, llm)
    finally:
        baseline.API_KEYS.reset(token)
    return {"run_id": run.get("run_id"), "suspects": suspects, "explanation": text}
