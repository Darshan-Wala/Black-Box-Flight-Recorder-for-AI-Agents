# Black Box: Part 1 (recorder, replay, interface)

Part 1 records any agent run in the shared step format. It rewinds and smart-replays runs,
and shows everything in a Streamlit UI. It talks to Part 2 only through the frozen `contract/`:

| Seam | Direction | Where |
|---|---|---|
| `run.json` files | Part 1 → Part 2 | `runs/run_XXXX.json` |
| `POST /diagnose` | Part 1 UI → Part 2 | port **8001** (`DIAGNOSE_URL`) |
| `POST /replay` | Part 2 verify loop → Part 1 | port **8000** |
| Streamlit UI | you | port **8501** |

Full API spec: [`contract/api.md`](contract/api.md).

## Setup (Windows PowerShell)
```powershell
cd "C:\Users\Atharva Wadekar\OneDrive\Documents\Hackathon\blackbox"
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest -q tests          # 42 tests, offline, a few seconds
```

## Test Part 1 alone (no Part 2, no Ollama)
```powershell
python run_batch.py --models mock-large,mock-small   # about 280 runs in seconds -> runs/
python contract\validate.py runs                     # every run.json checked against the contract
python start.py --mock-diagnose                      # replay API :8000 + mock diagnose :8001 + UI :8501
# in a second terminal: play Part 2's verify loop against your /replay
python -m mocks.mock_verify
```
If nothing is listening on 8001, the UI falls back to the built-in mock diagnose, and its
"Source" caption says so. Set `MOCK_ORACLE=1` before `start.py` to make the mock put the labelled
faulty step first (UI rehearsal only).

## Real runs with Ollama (what you hand to Part 2)
```powershell
ollama pull llama3; ollama pull mistral
python run_batch.py --models llama3,mistral --tasks 10 --repeat 2
```
Each clean run is followed by one run per fault type (`empty_search, wrong_tool_result, wrong_number,
skipped_step, wrong_tool`). The agent **actually runs with the broken step**, and `faulty_step` is the label.
Faults that don't break the run are dropped. Use `--framework langgraph` for the second framework.
Zip `runs/` and send it to Part 2.

## Real LLMs
| Model id | Runs on | Needs |
|---|---|---|
| `mock-large`, `mock-small` | offline rule-based stand-in | nothing |
| `llama3`, `mistral`, `ollama:<model>` | Ollama on this laptop | `ollama pull <model>` |
| `groq:<model>`, `openai:<model>`, `gemini:<model>`, `openrouter:<model>` | the provider's API | `GROQ_API_KEY` / `OPENAI_API_KEY` / `GEMINI_API_KEY` / `OPENROUTER_API_KEY` |
| `api:<model>` | any OpenAI-style server | `BLACKBOX_LLM_BASE_URL`, `BLACKBOX_LLM_API_KEY` |

```powershell
$env:GROQ_API_KEY="..."                      # set before start.py / run_batch.py
python run_batch.py --models groq:<model>,llama3 --tasks 10 --repeat 2
```
The id is saved in the run, so a replay calls the same model again. Rate limits (HTTP 429) are retried;
a call that still fails is recorded as a step with `error: llm_error: ...`, never a crash. In the UI,
type the id in "…or another model id".

## Custom runs (your own query)
The dashboard's **Custom** page (`http://localhost:8501/#/custom`) runs a query you type through the same
agent, recorder, fault injection and store as every other run, so it shows up under Activity and works
with diagnosis, replay and verify.

* **Models offered:** only what can run right now: the mock models, models installed in a running Ollama,
  and hosted providers whose API key is set (type `groq:<model>` etc.).
* **Modes:** Normal run (never carries a fault) or Fault injection (fault type, optional step). The query is
  sent to the agent unchanged; the fault replaces one step's output and the run records which step.
* **How the outcome is judged** (saved in the run as `custom`, and reused by replays):
  an expected answer you give; else the local fact table when the query matches one of its question
  shapes; else, for a fault run, a fault-free reference run recorded first; else "completed" (ended with a
  number and no step error, which does not mean the answer is correct).
* **Verify:** `POST /api/custom/verify` replays each suspect in order with a plain retry through the
  existing replay and reports the first one that flips the run.
* **Lookups** (saved in the run as `custom.search`, and reused by replays):
  * `live`: any free-form query on a real model. Each search tries, in order: (1) **Wikidata** for
    population, area, elevation/height, length and GDP, taking the most recently dated value on record
    (read by code, no model involved, so it does not depend on what the model remembers; for a country's
    population or GDP the World Bank's latest figure is used instead when it is newer); (2) the model
    you picked, with a Wikipedia extract as reference; (3) the model alone when the web is not reachable.
    `custom.lookups` records the source of each one (`wikidata` or `worldbank` with `as_of` year, `wikipedia+model` or
    `model`) and the trace shows it. Numbers are in full digits, or in the unit the query asks for
    ("in millions"), so the lookups of one run can be combined. Model answers are not verified facts.
    `BLACKBOX_WEB_SEARCH=0` turns the web off. A fault run reuses the lookups of its reference run.
  * `fact_table`: the built-in question shapes about names in `tasks.py`, and always for the mock models.
  Runs that are not custom never use live lookups; `tools.search` is unchanged for them.
* **Limits:** the agent answers with a number, so ask for something that can be worked out from numbers.
  The mock models only run the four built-in question shapes with names from the fact table. The diagnosis
  backend ranks steps; it does not name a fault type.

Endpoints (in `ui_api.py`): `GET /api/custom/meta`, `POST /api/custom/run`, `GET /api/custom/runs/{id}`,
`POST /api/custom/verify`. Logic: `custom.py`. Every run now also saves `timing` (start time and duration per step).

## Replay features
* **State restoration.** A replay restores a saved answer when the step about to run has the same actor,
  kind and input as a saved step the edit does not affect. Matching is by content, not step number, so it
  still works when an edit makes the run longer or shorter. Every replay reports `restore`: how many steps
  before the checkpoint came back from the record, whether that was exact, and a state hash for both runs.
  `GET /runs/{id}/rewind/{step}` returns the restored state (plan in force, results so far, tokens spent).
* **Alternative execution from a checkpoint.** `POST /branch` `{"run_id", "step_no", "model"?, "edit"?, "n_runs"?}`
  restores everything before `step_no` and runs that step and all later steps for real, optionally with
  another model. In the UI: Edit → "Branch from here". Same response shape as `/replay`, with `mode: "branch"`.
* **Original vs modified trace.** Every `/replay` and `/branch` response carries `comparison`: per step
  `same | changed | added | removed`, what changed (input, output, error), and a summary (first difference,
  outcome, final answer, steps, tokens). `GET /compare/{run_a}/{run_b}` compares any two stored runs.
  The UI shows the two traces aligned row by row with changed words highlighted.

These are additive: the frozen `/replay` request and response in `contract/` are unchanged.

## Integrate with Part 2's real /diagnose
```powershell
$env:DIAGNOSE_URL="http://<friend-B-ip>:8001"; python start.py
```
Part 2 points its verify loop at `http://<your-ip>:8000/replay`. Run uvicorn with `--host 0.0.0.0`
if it's on another laptop. To replay Part 2's synthetic runs (not in your store), they send the run
inline as `"run": {...}`.

## Files
| File | What it does |
|---|---|
| `recorder.py` | Wraps every model call, tool call and decision as a contract step. Computes `depends_on` and applies faults. In replay mode it reuses saved outputs. `wrap_model` / `wrap_tool` are the one-line hooks for a new agent. |
| `adapters/custom_agent.py` | Plain Python agent: plan → (search → read) per fact → calculate → answer |
| `adapters/langgraph_agent.py` | The same agent as a LangGraph `StateGraph` (second framework, about 20 lines of hooks) |
| `replay.py` | `dependency_map`, `dependents`, `rewind`, `smart_replay` (n_runs, savings) |
| `replay_api.py` | FastAPI: `POST /replay`, `GET /runs`, `GET /runs/{id}`, `GET /runs/{id}/rewind/{step}`, `POST /record` |
| `store.py` | SQLite (`data/blackbox.db`, runs + steps tables) and run.json export |
| `ui_api.py` + `ui/index.html` | Dashboard on port 8501 (default UI): activity feed of runs, filters, overview charts, run detail with timeline, top-3 suspects, explanation, edit/rewind/branch, side-by-side diff, savings meter, "save confirmed cause", and a Results page for `part2/results.md`. Additive only, same clients and store as `app.py` |
| `app.py` | Old Streamlit UI, still works via `python start.py --streamlit`: timeline, top-3 suspects, edit/rewind panel, side-by-side view, savings meter, "save confirmed cause" |
| `inject.py` | Fault-injection flag `fault={"step_no": 4, "type": "empty_search"}` |
| `custom.py` | Custom runs: model discovery, validation, live lookups, outcome judging, `record_custom` |
| `apikeys.py` | Bring-your-own-key: reads a visitor's keys from the `X-Blackbox-Keys` header and scopes them to that request (dashboard page: **API keys**) |
| `scripts/load_runs.py` | Rebuilds `data/blackbox.db` from `runs/*.json` on a fresh clone (the database is not in git) |
| `tasks.py`, `tools.py` | 25 demo tasks over a local fact corpus, search and calculator tools. Demo tasks: `t01`, `t16`. |
| `llm.py` | `MockLLM` (offline, deterministic) and `OllamaLLM` |
| `mocks/` | Stand-ins for Part 2: `/diagnose` heuristics and server, verify-loop client |
| `contract/` | Frozen schemas, API spec, 3 sample runs, validator |

## Key rules (agreed with Part 2)
* **depends_on**: the steps the agent explicitly used, plus any earlier step whose output text (12 or more chars)
  or whose numbers (3 or more chars) appear in this step's input.
* **Smart replay** reuses a saved output only when the step is not downstream of the edit **and**
  its input is unchanged. Everything else reruns for real.
* **Edits**: `{"output": ...}` replaces the output, `{"input": ...}` reruns with a new input, `{"rerun": true}` retries as-is.

## Plug in a new agent (about 20 lines)
```python
from recorder import Recorder
rec = Recorder("myframework", "llama3", task_text)
llm_call = rec.wrap_model(my_llm, actor="planner")   # every call is now recorded
search  = rec.wrap_tool(my_search_fn, actor="search_agent")
answer = my_agent(task_text, llm_call, search)
run = rec.finish(answer, "success" if ok(answer) else "fail")
store.save_run(run)
```
