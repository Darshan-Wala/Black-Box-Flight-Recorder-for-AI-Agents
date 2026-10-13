"""Bring-your-own-key: a dashboard visitor's API keys (X-Blackbox-Keys header)."""
import json
import sys
from pathlib import Path

import pytest
import requests
from fastapi.testclient import TestClient

import apikeys
import llm
import store
import ui_api

KEY = "gsk_visitor_key_123"


def _hdr(**keys):
    return {"X-Blackbox-Keys": json.dumps(keys)}


@pytest.fixture
def fake_llm(monkeypatch):
    """A hosted model that answers like mock-large and records the Authorization header it got."""
    mock, seen = llm.MockLLM("mock-large"), []

    class Resp:
        status_code, headers = 200, {}

        def __init__(self, d): self.d = d
        def json(self): return self.d

    def fake_post(url, json=None, timeout=None, headers=None):
        seen.append((headers or {}).get("Authorization"))
        text = mock.complete(json["messages"][0]["content"]).text
        return Resp({"choices": [{"message": {"content": text}}], "usage": {"total_tokens": 5}})

    monkeypatch.setattr(requests, "post", fake_post)
    for env in apikeys.KEY_ENV.values():
        monkeypatch.delenv(env, raising=False)
    return seen


def test_parse_and_validation():
    assert apikeys.parse(json.dumps({"groq": KEY, "nope": "x"})) == {"GROQ_API_KEY": KEY}
    assert apikeys.parse(None) == {} and apikeys.parse("") == {}
    for bad in ("not json", "[1]", json.dumps({"groq": "has space"})):
        with pytest.raises(apikeys.KeysError):
            apikeys.parse(bad)
    assert apikeys.check_base_url("https://api.example.com/v1/") == "https://api.example.com/v1"
    for bad in ("http://api.example.com", "https://localhost:8000", "https://127.0.0.1", "https://10.0.0.5/v1",
                "https://169.254.169.254", "https://box.local"):
        with pytest.raises(apikeys.KeysError):
            apikeys.check_base_url(bad)


def test_bad_header_is_a_clear_400():
    r = TestClient(ui_api.app).get("/api/meta", headers={"X-Blackbox-Keys": "{oops"})
    assert r.status_code == 400 and "API keys" in r.json()["detail"]


def test_keys_are_per_request(fake_llm):
    c = TestClient(ui_api.app)
    with_key = {p["prefix"]: p for p in c.get("/api/custom/meta", headers=_hdr(groq=KEY)).json()["providers"]}
    without = {p["prefix"]: p for p in c.get("/api/custom/meta").json()["providers"]}
    assert with_key["groq"]["available"] and with_key["groq"]["user_key"]
    assert not without["groq"]["available"] and not without["groq"]["user_key"]
    assert apikeys.get("GROQ_API_KEY") == ""          # nothing leaks outside the request


def test_record_uses_visitor_key_and_never_stores_it(fake_llm):
    c = TestClient(ui_api.app)
    r = c.post("/api/record", json={"task_id": "t01", "model": "groq:some-model"}, headers=_hdr(groq=KEY))
    assert r.status_code == 200, r.text
    assert r.json()["outcome"] == "success" and fake_llm and set(fake_llm) == {f"Bearer {KEY}"}
    run = store.get_run(r.json()["run_id"])
    assert run["model"] == "groq:some-model" and KEY not in json.dumps(run)
    assert KEY.encode() not in Path(store.DB_PATH).read_bytes()
    # without the header the same model has no key
    r2 = c.post("/api/record", json={"task_id": "t01", "model": "groq:some-model"}).json()
    assert r2["outcome"] == "fail" and store.get_run(r2["run_id"])["steps"][0]["error"] == "llm_error: GROQ_API_KEY not set"


def test_custom_run_worker_thread_sees_key(fake_llm):
    c = TestClient(ui_api.app)
    q = "What is the combined population of Canada and Italy in millions?"
    r = c.post("/api/custom/run", json={"query": q, "model": "groq:m"})
    assert r.status_code == 422 and "API keys" in r.json()["detail"]
    r = c.post("/api/custom/run", json={"query": q, "model": "groq:m"}, headers=_hdr(groq=KEY))
    assert r.status_code == 200, r.text
    assert fake_llm and set(fake_llm) == {f"Bearer {KEY}"}


def test_explain_forwards_keys_to_part2(monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from part2 import baseline, diagnose_api
    seen = {}

    def fake_ask(llm_id, prompt, url):
        seen["key"] = baseline._env("GROQ_API_KEY")
        return json.dumps({"summary": "s", "root_cause": "r", "how_it_spread": "h", "suggested_fix": "f"})

    monkeypatch.setattr(baseline, "_ask", fake_ask)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    run = json.loads((Path(__file__).resolve().parents[1] / "contract" / "samples" / "sample_fail_wrong_number.json")
                     .read_text(encoding="utf-8"))
    out = diagnose_api.explain({"run": run, "llm": "groq:m", "api_keys": {"GROQ_API_KEY": KEY, "PATH": "x"}})
    assert out["explanation"]["root_cause"] == "r" and seen["key"] == KEY
    assert baseline._env("GROQ_API_KEY") == ""        # reset after the request

    # the dashboard's in-process client sends the visitor's keys along
    import clients
    monkeypatch.setattr(clients, "INPROCESS", True)
    token = apikeys._user.set({"GROQ_API_KEY": KEY})
    try:
        seen.clear()
        assert clients.explain(run, llm="groq:m")["root_cause"] == "r" and seen["key"] == KEY
    finally:
        apikeys._user.reset(token)


def test_key_check_endpoint(monkeypatch):
    calls = []

    class Resp:
        def __init__(self, code, d=None): self.status_code, self.d = code, d or {}
        def json(self): return self.d

    def fake_get(url, headers=None, timeout=None):
        calls.append((url, headers))
        if headers["Authorization"] != f"Bearer {KEY}":
            return Resp(401)
        return Resp(200, {"data": [{"id": "models/gemini-2.5-flash"}, {"id": "text-embedding-004"}, {"id": "gemini-2.5-pro"}]})

    monkeypatch.setattr(requests, "get", fake_get)
    c = TestClient(ui_api.app)
    r = c.post("/api/keys/check", json={"provider": "gemini", "key": KEY}).json()
    assert r["ok"] and r["models"] == ["gemini-2.5-flash", "gemini-2.5-pro"]
    assert calls[-1][0] == "https://generativelanguage.googleapis.com/v1beta/openai/models"
    r = c.post("/api/keys/check", json={"provider": "gemini", "key": "wrong"}).json()
    assert not r["ok"] and "rejected" in r["error"]
    assert c.post("/api/keys/check", json={"provider": "api", "key": KEY, "base_url": "http://x.com"}).status_code == 422
    assert c.post("/api/keys/check", json={"provider": "nope", "key": KEY}).status_code == 422
