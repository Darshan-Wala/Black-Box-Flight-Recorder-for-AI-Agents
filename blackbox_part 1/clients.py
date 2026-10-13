"""HTTP clients for the two cross-team seams, with offline fallbacks so Part 1 can be
tested without Part 2 running.

  DIAGNOSE_URL  (default http://localhost:8001)  -> Part 2's /diagnose
  REPLAY_URL    (default http://localhost:8000)  -> Part 1's own /replay API

With BLACKBOX_INPROCESS=1 (set by the Vercel entrypoint, server.py) there are no separate
services: diagnose / explain call Part 2's code directly and replay runs in this process.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import requests

DIAGNOSE_URL = os.environ.get("DIAGNOSE_URL", "http://localhost:8001").rstrip("/")
REPLAY_URL = os.environ.get("REPLAY_URL", "http://localhost:8000").rstrip("/")
INPROCESS = os.environ.get("BLACKBOX_INPROCESS", "").strip().lower() in ("1", "true", "yes", "on")


def _part2_api():
    """Part 2's diagnose_api module, imported from the repo root (one folder up)."""
    root = str(Path(__file__).resolve().parent.parent)
    if root not in sys.path:
        sys.path.append(root)
    from part2 import diagnose_api
    return diagnose_api


def url_given(url: str, default: str | None = None) -> bool:
    """True when a caller passed its own URL (then HTTP is used even in in-process mode)."""
    return url.rstrip("/") != (default or DIAGNOSE_URL)


def _clean(run: dict) -> dict:
    return {k: v for k, v in run.items() if not k.startswith("_")}


def diagnose(run: dict, url: str | None = None, timeout: int = 30) -> tuple[dict, str]:
    """Returns (response, source). Source tells the UI whether Part 2 or the mock answered."""
    url = (url or DIAGNOSE_URL).rstrip("/")
    try:
        if INPROCESS and not url_given(url):
            data = _part2_api().diagnose(_clean(run))
            where = "in-process"
        else:
            r = requests.post(f"{url}/diagnose", json=_clean(run), timeout=timeout)
            r.raise_for_status()
            data, where = r.json(), url
        from contract.validate import validate_diagnose
        validate_diagnose(data)
        return data, f"diagnose API ({where})"
    except Exception as e:  # noqa: BLE001
        from mocks.heuristics import diagnose as local
        return local(run), f"built-in mock (diagnose API unreachable: {type(e).__name__})"


def explain(run: dict, llm: str | None = None, url: str | None = None, timeout: int = 320) -> dict:
    """Plain-language explanation of a failed run from Part 2's POST /explain.
    Returns the explanation dict, or {"error": ...} when Part 2 is not reachable."""
    url = (url or DIAGNOSE_URL).rstrip("/")
    body = {"run": _clean(run)}
    if llm:
        body["llm"] = llm
        import apikeys
        keys = apikeys.user_keys()
        # the visitor's own keys go along, but only in-process, to this machine, or over HTTPS
        if keys and ((INPROCESS and not url_given(url)) or url.startswith("https://")
                     or url.split("://")[-1].startswith(("localhost", "127.0.0.1"))):
            body["api_keys"] = keys
    if INPROCESS and not url_given(url):
        try:
            return _part2_api().explain(body)["explanation"]
        except Exception as e:  # noqa: BLE001
            return {"error": f"explanation failed ({type(e).__name__}: {e})"}
    try:
        r = requests.post(f"{url}/explain", json=body, timeout=timeout)
        r.raise_for_status()
        return r.json()["explanation"]
    except Exception as e:  # noqa: BLE001
        return {"error": f"explanation needs Part 2's API at {url} ({type(e).__name__})"}


def replay(run_id: str, step_no: int, edit: dict | None, n_runs: int = 3, run: dict | None = None,
           url: str | None = None, timeout: int = 600, branch: bool = False,
           model: str | None = None) -> tuple[dict, str]:
    """Smart replay (POST /replay) or, with branch=True, alternative execution from the
    checkpoint (POST /branch, optionally with another model)."""
    url = (url or REPLAY_URL).rstrip("/")
    body = {"run_id": run_id, "step_no": step_no, "n_runs": n_runs}
    if edit is not None:
        body["edit"] = edit
    if branch and model:
        body["model"] = model
    if run is not None:
        body["run"] = _clean(run)
    path = "branch" if branch else "replay"
    if INPROCESS and not url_given(url, REPLAY_URL):
        import replay as rp
        if not branch and edit is None:
            raise RuntimeError("/replay 422: an edit is required")
        try:
            return rp.smart_replay(run_id=run_id, step_no=step_no, edit=edit, run=run, n_runs=n_runs,
                                   branch=branch, model=model if branch else None), "replay (in-process)"
        except LookupError as e:
            raise RuntimeError(f"/{path} 404: {e}")
        except (rp.ReplayError, ValueError) as e:
            raise RuntimeError(f"/{path} 422: {e}")
    try:
        r = requests.post(f"{url}/{path}", json=body, timeout=timeout)
    except requests.ConnectionError:
        import replay as rp
        return rp.smart_replay(run_id=run_id, step_no=step_no, edit=edit, run=run, n_runs=n_runs,
                               branch=branch, model=model if branch else None), \
            "in-process (replay API not running)"
    if r.status_code >= 400:
        raise RuntimeError(f"/{path} {r.status_code}: {r.text}")
    return r.json(), f"replay API ({url})"
