import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import inject
import replay
import runner
import store
import tasks
from contract.validate import validate_diagnose, validate_replay_request, validate_replay_response, validate_run
from mocks.heuristics import diagnose
from recorder import detect_deps

ROOT = Path(__file__).resolve().parent.parent
T01 = tasks.get_task("t01")


# ---------------------------------------------------------------- contract
def test_samples_valid():
    files = sorted((ROOT / "contract" / "samples").glob("*.json"))
    runs = [json.loads(f.read_text()) for f in files]
    assert len(runs) == 3 and sorted(r["outcome"] for r in runs) == ["fail", "fail", "success"]
    for r in runs:
        validate_run(r)


@pytest.mark.parametrize("framework", ["custom", "langgraph"])
@pytest.mark.parametrize("model", ["mock-large", "mock-small"])
def test_recorded_runs_valid(model, framework):
    for t in tasks.TASKS[:6]:
        r = runner.record_run(t, model, framework, save=False)
        r["run_id"] = "x"
        validate_run(r)


def test_mock_large_solves_all_tasks():
    assert all(runner.record_run(t, "mock-large", save=False)["outcome"] == "success" for t in tasks.TASKS)


# ---------------------------------------------------------------- recorder / deps
def test_depends_on_rule():
    prev = [{"step_no": 1, "output": "France has a population of 68.2 million people."},
            {"step_no": 2, "output": "5"}, {"step_no": 3, "output": "1,234 apples"}]
    assert detect_deps(prev, "Text: France has a population of 68.2 million people.") == {1}
    assert detect_deps(prev, "5 + 5") == set()            # short output never matches
    assert detect_deps([{"step_no": 1, "output": "unknown"}], "If none, reply: unknown") == set()
    assert detect_deps(prev, "1234+1") == {3}             # number match, commas stripped


def test_branches_independent():
    r = runner.record_run(T01, "mock-large", save=False)
    # steps: 1 plan, 2 search A, 3 read A, 4 search B, 5 read B, 6 calc, 7 answer
    assert replay.dependents(r, 2) == {3, 6, 7}
    assert 4 not in replay.dependents(r, 2) and 5 not in replay.dependents(r, 2)


# ---------------------------------------------------------------- faults
@pytest.mark.parametrize("ftype", inject.FAULT_TYPES)
def test_each_fault_breaks_run(ftype):
    clean = runner.record_run(T01, "mock-large", save=False)
    step = inject.eligible_steps(clean, ftype)[0]
    r = runner.record_run(T01, "mock-large", fault={"step_no": step, "type": ftype, "seed": 3}, save=False)
    assert r["fault"]["applied"] and r["outcome"] == "fail" and r["faulty_step"] == step


def test_auto_fault_step():
    r = runner.record_run(T01, "mock-large", fault={"type": "empty_search", "step_no": None}, save=False)
    assert r["faulty_step"] == 2 and r["steps"][1]["error"] == "empty_result"


# ---------------------------------------------------------------- replay
def _faulty(step=4, ftype="wrong_tool_result"):
    r = runner.record_run(T01, "mock-large", fault={"step_no": step, "type": ftype, "seed": 5})
    assert r["outcome"] == "fail"
    return r


def test_replay_edit_output_flips_and_reuses():
    r = _faulty(4)
    good = tasks.fact_sentence("population", "Italy")
    res = replay.smart_replay(r["run_id"], 4, {"output": good}, n_runs=3)
    assert res["outcome_flipped"] and res["success_rate"] == 1.0
    assert res["steps_reused"] == 3 and res["steps_rerun"] == 4   # 1,2,3 reused; 4 edited; 5,6,7 rerun
    assert res["tokens_saved_pct"] > 0
    st = res["new_run"]["replay"]["status"]
    assert st["2"] == "reused" and st["4"] == "edited" and st["5"] == "rerun"
    validate_replay_response(res)
    assert store.get_run(res["new_run"]["run_id"])["parent_run_id"] == r["run_id"]


def test_replay_rerun_heals_injected_fault():
    for ftype, step in [("empty_search", 2), ("wrong_tool", 1), ("wrong_number", 5)]:
        r = _faulty(step, ftype)
        res = replay.smart_replay(r["run_id"], step, {"rerun": True}, n_runs=1)
        assert res["outcome_flipped"], ftype


def test_replay_wrong_suspect_does_not_flip():
    r = _faulty(4)
    res = replay.smart_replay(r["run_id"], 2, {"rerun": True}, n_runs=1)
    assert not res["outcome_flipped"] and res["new_run"]["outcome"] == "fail"


def test_replay_edit_input():
    r = _faulty(4, "empty_search")
    res = replay.smart_replay(r["run_id"], 4, {"input": "population of Italy"}, n_runs=1)
    assert res["outcome_flipped"]


def test_replay_inline_run_not_in_store():
    run = json.loads((ROOT / "contract/samples/sample_fail_wrong_number.json").read_text())
    res = replay.smart_replay(run=run, step_no=run["faulty_step"], edit={"rerun": True}, n_runs=1, save=False)
    assert res["outcome_flipped"]


def test_replay_bad_requests():
    r = _faulty(4)
    with pytest.raises(replay.ReplayError):
        replay.smart_replay(r["run_id"], 99, {"rerun": True})
    with pytest.raises(replay.ReplayError):
        replay.smart_replay(r["run_id"], 2, {"output": "a", "input": "b"})
    with pytest.raises(LookupError):
        replay.smart_replay("run_nope", 1, {"rerun": True})


def test_rewind():
    r = runner.record_run(T01, "mock-large", save=False)
    w = replay.rewind(r, 4)
    assert [s["step_no"] for s in w["before"]] == [1, 2, 3]
    assert w["will_rerun"] == [5, 6, 7] and w["will_reuse"] == [1, 2, 3]


# ---------------------------------------------------------------- API
def test_replay_api():
    from replay_api import app
    c = TestClient(app)
    r = _faulty(4)
    body = {"run_id": r["run_id"], "step_no": 4, "edit": {"rerun": True}, "n_runs": 1}
    validate_replay_request(body)
    resp = c.post("/replay", json=body)
    assert resp.status_code == 200
    validate_replay_response(resp.json())
    assert resp.json()["outcome_flipped"]
    assert c.post("/replay", json={**body, "run_id": "run_none"}).status_code == 404
    assert c.post("/replay", json={**body, "step_no": 50}).status_code == 422
    assert c.get(f"/runs/{r['run_id']}").json()["run_id"] == r["run_id"]
    assert c.get(f"/runs/{r['run_id']}/rewind/4").json()["will_rerun"] == [5, 6, 7]
    assert c.post("/record", json={"task_id": "t16", "model": "mock-large"}).json()["outcome"] == "success"


def test_mock_diagnose_contract_and_api():
    from mocks.mock_diagnose import app
    c = TestClient(app)
    r = _faulty(2, "empty_search")
    d = c.post("/diagnose", json=r).json()
    validate_diagnose(d)
    assert d["suspects"][0]["step_no"] == 2 and "empty_result" in d["suspects"][0]["evidence"]
    validate_diagnose(diagnose(runner.record_run(T01, "mock-large", save=False)))


def test_store_roundtrip_and_export():
    r = runner.record_run(T01, "mock-small")
    got = store.get_run(r["run_id"])
    assert got["steps"] == r["steps"] and got["outcome"] == r["outcome"]
    assert (store.RUNS_DIR / f"{r['run_id']}.json").exists()


def test_batch_runner():
    import run_batch
    stats = run_batch.main(["--models", "mock-large", "--tasks", "3"])
    assert stats["clean_success"] == 3 and sum(v for k, v in stats.items() if k.endswith("_fail")) >= 10


# ---------------------------------------------------------------- Ollama client (stubbed HTTP)
def test_ollama_client_and_agent_with_chatty_llm(monkeypatch):
    import requests
    from llm import MockLLM, OllamaLLM
    mock = MockLLM("mock-large")
    sent = []

    class Resp:
        def __init__(self, d): self.d = d
        def raise_for_status(self): pass
        def json(self): return self.d

    def fake_post(url, json=None, timeout=None):
        sent.append((url, json))
        text = mock.complete(json["prompt"]).text
        if json["prompt"].startswith("### PLAN"):
            text = "Sure! Here is my plan:\n" + text + "\nGood luck."      # real models add prose
        return Resp({"response": text, "prompt_eval_count": 100, "eval_count": 20})

    monkeypatch.setattr(requests, "post", fake_post)
    r = runner.record_run(T01, "llama3", save=False)
    assert r["outcome"] == "success" and r["model"] == "llama3"
    assert sent[0][0].endswith("/api/generate") and sent[0][1]["model"] == "llama3"
    assert sent[0][1]["format"] == "json" and r["steps"][0]["tokens"] == 120

    def down(*a, **k):
        raise requests.ConnectionError("refused")
    monkeypatch.setattr(requests, "post", down)
    r = runner.record_run(T01, "llama3", save=False)
    assert r["outcome"] == "fail" and r["steps"][0]["error"].startswith("llm_error")


# ---------------------------------------------------------------- state restoration
def test_state_restored_exactly_and_reported():
    r = _faulty(4)
    res = replay.smart_replay(r["run_id"], 4, {"rerun": True}, n_runs=2)
    rst = res["restore"]
    assert res["state_restored"] and rst["exact"] and rst["prefix_steps"] == 3 and rst["restored"] == 3
    assert rst["state_hash"] == rst["replayed_hash"] == replay.rewind(r, 4)["state"]["state_hash"]
    assert res["new_run"]["steps"][:3] == r["steps"][:3]


def test_restore_by_content_when_step_numbers_shift():
    # wrong_tool at the planner: step 2 was a failed calculator call. Fixing the plan changes what
    # step 2 does, but the untouched search for the second fact must still be restored, not rerun.
    r = _faulty(1, "wrong_tool")
    clean = runner.record_run(T01, "mock-large", save=False)
    res = replay.smart_replay(r["run_id"], 1, {"output": clean["steps"][0]["output"]}, n_runs=1)
    assert res["outcome_flipped"]
    # even though every later step depends on the plan, none of them is restored by position alone
    st = res["new_run"]["replay"]["status"]
    assert st["1"] == "edited" and all(v in ("rerun", "edited") for v in st.values())

    class Agent:   # an agent that inserts one extra step before repeating the saved ones
        def run(self, task, rec):
            for s in r["steps"][:2]:
                rec.step(s["actor"], s["kind"], s["input"], lambda i: ("x", 1, None))
            rec.step("extra", "tool_call", "new step", lambda i: ("y", 0, None))
            for s in r["steps"][3:5]:
                rec.step(s["actor"], s["kind"], s["input"], lambda i: ("x", 1, None))
            return "Final answer: 0"
    from recorder import Recorder, ReplayCtx
    ctx = ReplayCtx(original=r, edit_step=3, edit={"rerun": True}, dirty={3})
    rec = Recorder("custom", "mock-large", r["task"], replay=ctx)
    Agent().run(r["task"], rec)
    assert rec.status == {1: "reused", 2: "reused", 3: "edited", 4: "reused", 5: "reused"}
    assert rec.restored_from == {1: 1, 2: 2, 4: 4, 5: 5}
    assert rec.steps[3]["output"] == r["steps"][3]["output"]


def test_rewind_state():
    r = runner.record_run(T01, "mock-large", save=False)
    stt = replay.rewind(r, 4)["state"]
    assert stt["steps_done"] == 3 and stt["plan"]["expression"] == "{0}+{1}"
    assert [x["step_no"] for x in stt["results_so_far"]] == [2, 3] and stt["branch_would_rerun"] == [4, 5, 6, 7]
    assert stt["tokens_spent"] == sum(s["tokens"] for s in r["steps"][:3])


# ---------------------------------------------------------------- alternative execution from a checkpoint
def test_branch_from_checkpoint():
    r = _faulty(4)                       # bad search result at step 4, run with mock-large
    res = replay.branch(r["run_id"], 4, model="mock-small", n_runs=1)
    new = res["new_run"]
    st = new["replay"]["status"]
    assert res["mode"] == "branch" and res["model"] == "mock-small" and new["model"] == "mock-small"
    assert [st[str(i)] for i in (1, 2, 3)] == ["reused"] * 3            # before the checkpoint: restored
    assert all(st[str(i)] == "rerun" for i in range(4, len(new["steps"]) + 1))   # from it on: all rerun
    assert res["state_restored"] and res["steps_reused"] == 3
    assert res["outcome_flipped"] and new["parent_run_id"] == r["run_id"]
    # a branch reruns unaffected later steps too; a smart replay does not
    r2 = _faulty(2)
    smart = replay.smart_replay(r2["run_id"], 2, {"rerun": True}, n_runs=1)
    br = replay.branch(r2["run_id"], 2, n_runs=1)
    assert br["steps_rerun"] > smart["steps_rerun"] and br["outcome_flipped"] and smart["outcome_flipped"]


def test_branch_with_edit_and_api():
    from replay_api import app
    c = TestClient(app)
    r = _faulty(4, "empty_search")
    resp = c.post("/branch", json={"run_id": r["run_id"], "step_no": 4, "edit": {"input": "population of Italy"},
                                   "model": "mock-small", "n_runs": 1})
    assert resp.status_code == 200 and resp.json()["outcome_flipped"] and resp.json()["mode"] == "branch"
    validate_replay_response(resp.json())
    assert c.post("/branch", json={"run_id": r["run_id"], "step_no": 99}).status_code == 422
    assert c.post("/branch", json={"run_id": "nope", "step_no": 1}).status_code == 404


# ---------------------------------------------------------------- original vs modified trace
def test_compare_traces():
    r = _faulty(4)
    res = replay.smart_replay(r["run_id"], 4, {"rerun": True}, n_runs=1)
    cmp_ = res["comparison"]
    sm = cmp_["summary"]
    assert sm["first_divergence"] == 4 and sm["same"] == 3 and sm["changed"] == 4 and sm["added"] == sm["removed"] == 0
    assert sm["outcome"] == ["fail", "success"] and sm["outcome_changed"]
    row4 = cmp_["steps"][3]
    assert row4["status"] == "changed" and row4["output_changed"] and not row4["input_changed"]
    assert row4["replay_status"] == "edited" and cmp_["steps"][0]["replay_status"] == "reused"
    assert replay.compare(r, r)["summary"]["changed"] == 0


def test_compare_when_step_count_changes_and_api():
    from replay_api import app
    r = _faulty(1, "wrong_tool")         # 6 steps: the calculator never ran
    res = replay.smart_replay(r["run_id"], 1, {"rerun": True}, n_runs=1)
    sm = res["comparison"]["summary"]
    assert sm["steps"] == [6, 7] and sm["added"] >= 1 and res["outcome_flipped"]
    assert all(row["a_step"] is None for row in res["comparison"]["steps"] if row["status"] == "added")
    c = TestClient(app)
    got = c.get(f"/compare/{r['run_id']}/{res['new_run']['run_id']}")
    assert got.status_code == 200 and got.json()["summary"]["steps"] == [6, 7]
    assert c.get(f"/compare/{r['run_id']}/nope").status_code == 404


# ---------------------------------------------------------------- hosted LLM client (stubbed HTTP)
def test_openai_compatible_client(monkeypatch):
    import requests
    import llm
    from llm import MockLLM
    mock, sent = MockLLM("mock-large"), []

    class Resp:
        def __init__(self, code, d=None, headers=None): self.status_code, self.d, self.headers = code, d, headers or {}
        def json(self): return self.d

    def fake_post(url, json=None, timeout=None, headers=None):
        sent.append((url, json, headers))
        if len(sent) == 1:
            return Resp(429, headers={"retry-after": "0"})          # first call is rate limited
        text = mock.complete(json["messages"][0]["content"]).text
        return Resp(200, {"choices": [{"message": {"content": "```json\n" + text + "\n```" if "response_format" in json else text}}],
                          "usage": {"total_tokens": 77}})

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setenv("GROQ_API_KEY", "k")
    r = runner.record_run(T01, "groq:some-model", save=False)
    assert r["outcome"] == "success" and r["model"] == "groq:some-model" and r["steps"][0]["tokens"] == 77
    url, body, headers = sent[1]
    assert url == "https://api.groq.com/openai/v1/chat/completions" and body["model"] == "some-model"
    assert headers["Authorization"] == "Bearer k" and body["response_format"] == {"type": "json_object"}
    # a replay of a run recorded with a hosted model calls the same model again
    assert isinstance(llm.get_llm(r["model"]), llm.OpenAICompatLLM) and isinstance(llm.get_llm("llama3"), llm.OllamaLLM)

    monkeypatch.delenv("GROQ_API_KEY")
    r = runner.record_run(T01, "groq:some-model", save=False)
    assert r["outcome"] == "fail" and r["steps"][0]["error"] == "llm_error: GROQ_API_KEY not set"


# ---------------------------------------------------------------- failure explanation (Part 2's /explain)
def test_explain_client(monkeypatch):
    import requests
    import clients
    r = _faulty(4)
    assert "error" in clients.explain(r)                 # Part 2 not running: a clear message, no crash

    class Resp:
        def raise_for_status(self): pass
        def json(self): return {"explanation": {"summary": "s", "root_cause": "r", "how_it_spread": "h",
                                                "suggested_fix": "f", "source": "x"}}
    sent = []
    monkeypatch.setattr(requests, "post", lambda url, json=None, timeout=None: sent.append((url, json)) or Resp())
    assert clients.explain(r, llm="groq:m")["root_cause"] == "r"
    assert sent[0][0].endswith("/explain") and sent[0][1]["llm"] == "groq:m" and sent[0][1]["run"]["run_id"] == r["run_id"]


# ---------------------------------------------------------------- custom runs (own query, optional fault)
FREE = "Add up how many people live in France and Spain, in millions."      # not one of the built-in question shapes


def _hosted_stub(monkeypatch):
    """A fake hosted LLM that plans two lookups for any query, then reads and answers like the mock."""
    import requests
    from llm import MockLLM
    mock = MockLLM("mock-large")

    class Resp:
        status_code, headers = 200, {}
        def __init__(self, text): self.text = text
        def json(self): return {"choices": [{"message": {"content": self.text}}], "usage": {"total_tokens": 50}}

    def fake_post(url, json=None, timeout=None, headers=None):
        p = json["messages"][0]["content"]
        if p.startswith("### PLAN"):
            assert FREE in p                               # the user's query reaches the model unchanged
            return Resp('{"steps": [{"tool": "search", "query": "population of France"}, '
                        '{"tool": "search", "query": "population of Spain"}], "expression": "{0}+{1}"}')
        if p.startswith("### LOOKUP"):                     # live search: the model answers each lookup itself
            calls.append(p)
            if "Atlantis" in p:
                return Resp("No results found.")
            return Resp("France has 68200000 people." if "France" in p else "Spain has 48100000 people.")
        return Resp(mock.complete(p).text)
    calls = []
    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setenv("GROQ_API_KEY", "k")
    return calls


def test_custom_normal_run_fact_table_query():
    import custom
    q = "What is the combined population of Japan and Canada in millions?"
    res = custom.record_custom(q, "mock-large")
    run = res["run"]
    validate_run(run)
    assert run["task"] == q and run["outcome"] == "success" and "fault" not in run and res["reference"] is None
    assert run["custom"]["mode"] == "normal" and run["custom"]["expected_source"] == "fact_table"
    assert run["custom"]["judged_by"] == "expected" and abs(run["custom"]["expected"] - 164.6) < 1e-6
    assert [t["step_no"] for t in run["timing"]] == [s["step_no"] for s in run["steps"]]
    assert store.get_run(run["run_id"])["custom"] == run["custom"]           # saved through the normal store


def test_custom_fault_run_is_explicit_and_replayable():
    import custom
    q = "What is the combined population of Japan and Canada in millions?"
    run = custom.record_custom(q, "mock-large", fault=custom.check_fault("wrong_tool_result", None))["run"]
    assert run["task"] == q                                                   # query not altered
    assert run["outcome"] == "fail" and run["fault"]["applied"] and run["faulty_step"] == run["fault"]["step_no"] == 2
    res = replay.smart_replay(run["run_id"], 2, {"rerun": True}, n_runs=1)    # existing replay judges it
    assert res["outcome_flipped"] and res["new_run"]["custom"]["expected"] == run["custom"]["expected"]
    assert not replay.smart_replay(run["run_id"], 4, {"rerun": True}, n_runs=1)["outcome_flipped"]


def test_custom_free_form_query_uses_reference_run(monkeypatch):
    import custom
    calls = _hosted_stub(monkeypatch)
    normal = custom.record_custom(FREE, "groq:some-model")["run"]
    assert normal["outcome"] == "success" and normal["custom"]["judged_by"] == "completed"
    assert normal["custom"]["expected"] is None and "fault" not in normal
    # a free-form query on a real model looks things up live, in full digits, and says where from
    assert normal["custom"]["search"] == "live" and normal["steps"][1]["output"] == "France has 68.2 million people."
    assert normal["custom"]["lookups"]["population of France"] == {"source": "model", "title": None}
    assert tasks.parse_answer(normal["final_answer"]) == 116.3 and len(calls) == 2
    res = custom.record_custom(FREE, "groq:some-model", fault=custom.check_fault("wrong_number", None))
    run, ref = res["run"], res["reference"]
    assert len(calls) == 4                                  # the fault run reuses its reference run's lookups
    assert ref["custom"]["mode"] == "reference" and "fault" not in ref and ref["outcome"] == "success"
    assert run["custom"]["expected_source"] == "reference_run" and run["custom"]["reference_run_id"] == ref["run_id"]
    assert run["outcome"] == "fail" and run["fault"]["applied"] and run["faulty_step"] == run["fault"]["step_no"]
    assert replay.smart_replay(run["run_id"], run["faulty_step"], {"rerun": True}, n_runs=1)["outcome_flipped"]
    given = custom.record_custom(FREE, "groq:some-model", expected=999.0)["run"]   # user-supplied expected answer wins
    assert given["outcome"] == "fail" and given["custom"]["expected_source"] == "user"


def test_custom_live_search_sources_replay_and_isolation(monkeypatch):
    import custom
    import requests
    import tools
    calls = _hosted_stub(monkeypatch)

    class Wiki:
        def raise_for_status(self): pass
        def json(self): return {"query": {"pages": {"1": {"index": 1, "title": "Demographics of France",
                                                          "extract": "France has  68.2 million inhabitants."}}}}
    monkeypatch.setenv("BLACKBOX_WEB_SEARCH", "1")
    monkeypatch.setattr(custom, "_web_down_until", 0.0)
    monkeypatch.setattr(custom, "wikidata", lambda q: None)     # this test is about the Wikipedia + model path
    monkeypatch.setattr(requests, "get", lambda url, **kw: Wiki())
    for framework in ("custom", "langgraph"):
        run = custom.record_custom(FREE, "groq:some-model", framework)["run"]
        assert run["outcome"] == "success" and run["custom"]["lookups"]["population of France"] == {
            "source": "wikipedia+model", "title": "Demographics of France"}
    assert "Demographics of France: France has 68.2 million inhabitants." in calls[0]

    def offline(url, **kw): raise requests.ConnectionError()
    monkeypatch.setattr(requests, "get", offline)              # Wikipedia unreachable -> the model alone, no crash
    run = custom.record_custom(FREE, "groq:some-model")["run"]
    assert run["outcome"] == "success" and run["custom"]["lookups"]["population of Spain"]["source"] == "model"
    assert custom.wikipedia("anything") is None and custom._web_down_until > 0
    # replaying a search step of a live run looks it up live again; other steps are reused
    n = len(calls)
    res = replay.smart_replay(run["run_id"], 2, {"rerun": True}, n_runs=1)
    assert len(calls) == n + 1 and res["new_run"]["outcome"] == "success" and res["new_run"]["custom"]["search"] == "live"
    # a lookup the model cannot answer is an empty search, and the run fails without crashing
    monkeypatch.setattr(custom, "LOOKUP", custom.LOOKUP + "Atlantis")
    lost = custom.record_custom(FREE, "groq:some-model")["run"]
    assert lost["outcome"] == "fail" and lost["steps"][1]["error"] == "empty_result" and custom.lookups_missed(lost) == 2
    assert custom.full_digits("about 1.43 billion people and 146 million") == "about 1430000000 people and 146000000"
    # outside a live custom run the search tool is the unchanged local fact table
    assert tools.run_tool("search", "population of France") == ("France has a population of 68.2 million people.", None)
    assert tools.run_tool("search", "population of India") == (tools.NOT_FOUND, "empty_result")
    with pytest.raises(custom.CustomError):                    # mock models cannot look up names outside the table
        custom.record_custom("What is the combined population of India and China in millions?", "mock-large")


def test_custom_wikidata_lookup_is_current_and_unit_aware(monkeypatch):
    import custom
    import requests
    calls = _hosted_stub(monkeypatch)
    Q = "http://www.wikidata.org/entity/"

    def claim(amount, unit="1", year=None, rank="normal"):
        c = {"rank": rank, "mainsnak": {"datavalue": {"value": {"amount": amount, "unit": unit}}}}
        if year:
            c["qualifiers"] = {"P585": [{"datavalue": {"value": {"time": f"+{year}-00-00T00:00:00Z"}}}]}
        return c
    ENT = {"Q142": {"labels": {"en": {"value": "France"}}, "claims": {
               "P298": [{"mainsnak": {"datavalue": {"value": "FRA"}}}],
               "P1082": [claim("+66000000", year=2015, rank="preferred"), claim("+68600000", year=2025),
                         claim("+99", year=2030, rank="deprecated")],
               "P2046": [claim("+643801", Q + "Q712226")]}},
           "Q29": {"labels": {"en": {"value": "Spain"}}, "claims": {"P1082": [claim("+49100000", year=2025)]}},
           "Q999": {"labels": {"en": {"value": "France (band)"}}, "claims": {}},
           "Q3392": {"labels": {"en": {"value": "Nile"}}, "claims": {"P2043": [claim("+6650000", Q + "Q11573")]}}}
    seen = []

    class R:
        def __init__(self, d): self.d = d
        def raise_for_status(self): pass
        def json(self): return self.d

    def fake_get(url, params=None, **kw):
        seen.append(params)
        if "/country/FRA/indicator/SP.POP.TOTL" in url:           # World Bank has a newer figure for France only
            return R([{"page": 1}, [{"countryiso3code": "FRA", "date": wb_year[0], "value": 68912345}]])
        if "/indicator/" in url:
            return R([{"message": "no data"}])
        if params["action"] == "wbsearchentities":
            hits = {"France": ["Q999", "Q142"], "Spain": ["Q29"], "Nile": ["Q3392"]}.get(params["search"], [])
            return R({"search": [{"id": i} for i in hits]})
        if params["action"] == "wbgetentities":
            return R({"entities": {i: ENT[i] for i in params["ids"].split("|")}})
        return R({"query": {"pages": {}}})                       # Wikipedia: nothing found
    wb_year = ["2019"]
    monkeypatch.setenv("BLACKBOX_WEB_SEARCH", "1")
    monkeypatch.setattr(custom, "_web_down_until", 0.0)
    monkeypatch.setattr(custom, "_wikidata_down_until", 0.0)
    monkeypatch.setattr(requests, "get", fake_get)
    # the latest dated, non-deprecated value wins; the first search hit without the property is skipped
    monkeypatch.setattr(custom, "_wb_down_until", 0.0)
    monkeypatch.setattr(custom, "_wb_cache", {})
    assert custom.wikidata("population of France 2026") == {"label": "France", "noun": "population", "value": 68600000.0,
                                                            "unit": "people", "year": "2025", "id": "Q142", "iso3": "FRA"}
    assert custom.wikidata("land area of France")["value"] == 643801.0
    assert custom.wikidata("length of the Nile river")["value"] == 6650.0          # metres -> kilometres
    assert custom.wikidata("capital of France") is None and custom.wikidata("population of Atlantis") is None
    run = custom.record_custom(FREE, "groq:some-model")["run"]                     # FREE asks "in millions"
    assert run["steps"][1]["output"] == "France has a population of 68.6 million people." and not calls
    assert run["custom"]["lookups"]["population of France"] == {"source": "wikidata", "title": "France (Q142)", "as_of": "2025"}
    assert run["outcome"] == "success" and tasks.parse_answer(run["final_answer"]) == 117.7
    # numbers are written so a small model cannot misread them: never three digits after the point
    assert [custom._num(v) for v in (1326.093, 214.212, 37.563, 0.9264, 1326093247.0, 8849.0, 6650.4)] == [
        "1326.1", "214.2", "37.6", "0.93", "1326093247", "8849", "6650.4"]
    # the World Bank figure replaces Wikidata's only when it is newer
    assert custom.worldbank("FRA", "population") == (68912345.0, "2019") and custom.worldbank("ESP", "population") is None
    wb_year[0] = "2026"
    monkeypatch.setattr(custom, "_wb_cache", {})
    monkeypatch.setattr(custom, "_wb_down_until", 0.0)
    run = custom.record_custom(FREE, "groq:some-model")["run"]
    assert run["steps"][1]["output"] == "France has a population of 68.9 million people."
    assert run["custom"]["lookups"]["population of France"] == {"source": "worldbank", "title": "France (FRA)", "as_of": "2026"}
    assert run["custom"]["lookups"]["population of Spain"]["source"] == "wikidata"
    assert custom.task_scale("total, in billions?") == (1e9, "billion") and custom.task_scale("in Spain") == (1.0, "")
    assert custom.rescale("Spain has 48,100,000 people.", 1e6, "million") == "Spain has 48.1 million people."

    def boom(url, **kw): raise requests.ConnectionError()
    monkeypatch.setattr(requests, "get", boom)                    # no web at all -> the model alone, still in millions
    run = custom.record_custom(FREE, "groq:some-model")["run"]
    assert run["steps"][1]["output"] == "France has 68.2 million people." and run["custom"]["lookups"][
        "population of France"]["source"] == "model" and tasks.parse_answer(run["final_answer"]) == 116.3


def test_custom_validation(monkeypatch):
    import custom
    monkeypatch.setattr(custom, "ollama_models", lambda timeout=0.8: None)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    for query, model in [("", "mock-large"), (FREE, "mock-large"), (FREE, "llama3"), (FREE, "groq:m"), (FREE, "groq:"),
                         ("x" * 3000, "mock-large")]:
        with pytest.raises(custom.CustomError):
            custom.record_custom(query, model)
    with pytest.raises(custom.CustomError):
        custom.check_fault("made_up", None)
    with pytest.raises(custom.CustomError):
        custom.check_fault("wrong_number", 0)
    monkeypatch.setattr(custom, "ollama_models", lambda timeout=0.8: ["llama3:latest"])
    custom.check_model("llama3", FREE)
    with pytest.raises(custom.CustomError):
        custom.check_model("mistral", FREE)
    ids = [m["id"] for m in custom.supported_models()["models"]]
    assert ids[:3] == ["mock-large", "mock-small", "llama3:latest"]


def test_custom_api_flow(monkeypatch):
    import ui_api
    c = TestClient(ui_api.app)
    meta = c.get("/api/custom/meta").json()
    assert [f["type"] for f in meta["fault_types"]] == inject.FAULT_TYPES and meta["knowledge"] and meta["models"]
    q = "How much taller is Everest than K2, in meters?"
    ok = c.post("/api/custom/run", json={"query": q, "model": "mock-large", "fault_type": "wrong_number"})   # normal mode
    assert ok.status_code == 200 and ok.json()["fault"] is None and ok.json()["outcome"] == "success"
    bad = c.post("/api/custom/run", json={"query": q, "model": "mock-large", "mode": "fault", "fault_type": "empty_search"}).json()
    assert bad["outcome"] == "fail" and bad["fault"] == {"type": "empty_search", "requested_step": None, "applied": True, "step_no": 2}
    assert c.post("/api/custom/run", json={"query": q, "model": "mock-large", "mode": "fault"}).status_code == 422
    assert c.post("/api/custom/run", json={"query": FREE, "model": "mock-large"}).status_code == 422
    miss = c.post("/api/custom/run", json={"query": q, "model": "mock-large", "mode": "fault", "fault_type": "wrong_tool",
                                           "fault_step": 5}).json()
    assert miss["fault"]["applied"] is False and miss["outcome"] == "success" and "not applied" in miss["notes"][0]
    listed = {r["run_id"]: r for r in c.get("/api/runs").json()}
    assert listed[bad["run_id"]]["custom"] == "fault" and listed[ok.json()["run_id"]]["custom"] == "normal"
    assert c.get(f"/api/custom/runs/{bad['run_id']}").json()["run_id"] == bad["run_id"]
    d = c.get(f"/api/runs/{bad['run_id']}/diagnosis").json()                 # existing diagnosis endpoint
    assert d["suspects"] and d["suspects"][0]["step_no"] == 2
    v = c.post("/api/custom/verify", json={"run_id": bad["run_id"]}).json()  # existing replay, one suspect at a time
    assert v["verified"] and v["confirmed_step"] == 2 and v["matches_injected"] and v["attempts"][0]["outcome_flipped"]
    assert c.get(f"/api/runs/{bad['run_id']}").json()["replays"]             # the replay is saved under the run
    assert c.post("/api/custom/verify", json={"run_id": ok.json()["run_id"]}).status_code == 422
