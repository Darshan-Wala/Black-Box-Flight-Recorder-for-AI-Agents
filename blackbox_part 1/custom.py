"""Custom runs: the user's own query, any supported model, an optional injected fault.

A custom run goes through the same agent adapters, Recorder, fault injection and store as
every other run, so diagnosis, replay and verification work on it unchanged. What is
different is how the outcome is judged, because a free-text query has no built-in answer:

  expected answer given by the user            -> compared with the agent's answer
  query is about facts in the local fact table -> expected answer computed from the table
  fault-injection run, nothing else known      -> a fault-free reference run is recorded first
                                                  and its answer is the expected one
  none of these                                -> "completed": the run counts as a success when
                                                  it ends with a number and no step errored

How the outcome was judged is saved in the run under `custom`, and replays of the run are
judged the same way.

Search in a custom run (saved as `custom.search`, and reused by replays):

  fact_table  the query is one of the built-in question shapes about facts in the local table
              (and always for the mock models): the normal local search tool.
  live        any other query on a real model. Each search tries, in order:
                1. Wikidata, for population, area, elevation/height, length and GDP: the most
                   recently dated value on record, read by code (no model involved). For a
                   country's population or GDP the World Bank's latest figure is used instead
                   when it is newer;
                2. the run's own model, with a Wikipedia extract as reference;
                3. the run's own model alone, when the web is not reachable.
              BLACKBOX_WEB_SEARCH=0 turns 1 and 2 off. Numbers are written in full digits, or
              in the unit the query asks for ("in millions"), so the lookups of one run can be
              combined. Runs that are not custom never use live search.
"""
from __future__ import annotations

import contextlib
import contextvars
import os
import re
import time

import apikeys
import inject
import llm as llm_mod
import store
import tasks
import tools
from adapters import FRAMEWORKS, get_agent
from recorder import Recorder

MAX_QUERY = 2000
FAULT_HELP = {
    "empty_search": "A search returns nothing.",
    "wrong_tool_result": "A search returns a fact about a different entity.",
    "wrong_number": "A reader or the calculator returns a wrong number.",
    "skipped_step": "A tool call returns an empty result with no error.",
    "wrong_tool": "The plan sends the first lookup to the calculator instead of search.",
}


LOOKUP = """### LOOKUP
You are the search tool of a research agent. Answer the lookup below in ONE short sentence.
The sentence must contain exactly one number: the value asked for, written in full digits with
no words such as thousand, million or billion (write 1430000000, not 1.43 billion), followed
by its unit. Do not mention years, dates or any other number.
Use the reference text when it contains the answer, and prefer its most recent figure;
otherwise use your own knowledge.
If you do not know, reply exactly: No results found.

Lookup: {query}
Reference text: {reference}
"""
WIKI_URL = "https://en.wikipedia.org/w/api.php"
_web_down_until = 0.0                    # after a failed request the web is skipped for a minute
_live: contextvars.ContextVar = contextvars.ContextVar("blackbox_live_search", default=None)


def web_enabled() -> bool:
    return os.environ.get("BLACKBOX_WEB_SEARCH", "1").strip().lower() not in ("0", "false", "off", "no")


def wikipedia(query: str, timeout: float = 4.0) -> tuple[str, str] | None:
    """(page title, intro text) of the best Wikipedia matches for a lookup, or None. Never raises."""
    global _web_down_until
    if not web_enabled() or time.time() < _web_down_until:
        return None
    import requests
    try:
        r = requests.get(os.environ.get("BLACKBOX_WIKI_URL", WIKI_URL), timeout=timeout,
                         headers=_UA,
                         params={"action": "query", "format": "json", "generator": "search", "gsrsearch": query,
                                 "gsrlimit": 2, "prop": "extracts", "exintro": 1, "explaintext": 1, "exlimit": 2})
        r.raise_for_status()
        pages = sorted(((r.json().get("query") or {}).get("pages") or {}).values(), key=lambda p: p.get("index", 99))
    except Exception:  # noqa: BLE001  offline, blocked or a bad reply: fall back to the model alone
        _web_down_until = time.time() + 60
        return None
    pages = [p for p in pages if (p.get("extract") or "").strip()]
    if not pages:
        return None
    text = "\n".join(f"{p.get('title', '')}: {' '.join(p['extract'].split())[:1200]}" for p in pages)
    return str(pages[0].get("title", "")), text


_SCALE = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}
_SCALED = re.compile(r"(\d[\d,]*\.?\d*)\s*(thousand|million|billion|trillion)\b", re.I)


def full_digits(text: str) -> str:
    """'1.43 billion' -> '1430000000', for models that ignore the full-digits instruction."""
    def repl(m):
        try:
            return tasks.fmt_value(round(float(m.group(1).replace(",", "").rstrip(".")) * _SCALE[m.group(2).lower()], 4))
        except ValueError:
            return m.group(0)
    return _SCALED.sub(repl, text)


WIKIDATA_URL = "https://www.wikidata.org/w/api.php"
_wikidata_down_until = 0.0
_UA = {"User-Agent": "BlackBoxFlightRecorder/1.0 (agent debugging demo)"}
_Q = "http://www.wikidata.org/entity/"
# what a lookup asks for -> (noun for the sentence, Wikidata properties to try, unit, unit id -> factor to that unit)
_ATTRS = [
    (r"gdp per capita|per[- ]capita gdp", "GDP per capita", ["P2132"], "US dollars", {"Q4917": 1.0}),
    (r"\bgdp\b|gross domestic product", "GDP", ["P2131"], "US dollars", {"Q4917": 1.0}),
    (r"population|inhabitants|people liv", "population", ["P1082"], "people", {"1": 1.0}),
    (r"\barea\b", "area", ["P2046"], "square kilometers",
     {"Q712226": 1.0, "Q25343": 1e-6, "Q35852": 0.01, "Q232291": 2.589988}),
    (r"elevation|height|\btall\b|\bhigh\b", "height", ["P2044", "P2048"], "meters",
     {"Q11573": 1.0, "Q3710": 0.3048, "Q828224": 1000.0}),
    (r"length|\blong\b", "length", ["P2043"], "kilometers", {"Q828224": 1.0, "Q11573": 0.001, "Q253276": 1.609344}),
]
_FILLER = {"the", "of", "a", "an", "in", "is", "what", "whats", "total", "current", "currently", "latest", "recent",
           "estimated", "estimate", "approximate", "number", "figure", "value", "land", "surface", "nominal", "how",
           "many", "much", "for", "as", "now", "today", "population", "inhabitants", "people", "living", "live",
           "area", "elevation", "height", "tall", "high", "length", "long", "gdp", "gross", "domestic", "product",
           "per", "capita", "km", "sq", "square", "kilometers", "kilometres", "meters", "metres", "millions",
           "billions", "thousands", "usd", "dollars"}
_KIND = {"river", "mountain", "mount", "country", "city", "lake", "island", "state", "peak"}
_UNITS = re.compile(r"\bin\s+(?:the\s+)?(thousand|million|billion|trillion)s\b", re.I)


def task_scale(task: str) -> tuple[float, str]:
    """(divisor, word) when the query asks for its answer "in millions" etc., else (1, "")."""
    m = _UNITS.search(task or "")
    return (_SCALE[m.group(1).lower()], m.group(1).lower()) if m else (1.0, "")


def _latest(claims: list, factors: dict) -> tuple[float, str | None] | None:
    """(value in our unit, year) of the most recently dated usable statement of a Wikidata property."""
    best = None
    for c in claims or []:
        try:
            if c.get("rank") == "deprecated":
                continue
            v = c["mainsnak"]["datavalue"]["value"]
            unit = str(v.get("unit", "1")).replace(_Q, "")
            if unit not in factors:
                continue
            when = ((c.get("qualifiers") or {}).get("P585") or [{}])[0].get("datavalue", {}).get("value", {}).get("time", "")
            year = when[1:5] if len(when) >= 5 and when[1:5].isdigit() else None
            cand = ((year or "0000", c.get("rank") == "preferred"), float(v["amount"]) * factors[unit], year)
        except (KeyError, TypeError, ValueError, IndexError):
            continue
        if best is None or cand[0] > best[0]:
            best = cand
    return (best[1], best[2]) if best else None


WORLDBANK_URL = "https://api.worldbank.org/v2"
_WB = {"population": "SP.POP.TOTL", "GDP": "NY.GDP.MKTP.CD", "GDP per capita": "NY.GDP.PCAP.CD"}
_wb_cache: dict = {}
_wb_down_until = 0.0


def worldbank(iso3: str, noun: str, timeout: float = 4.0) -> tuple[float, str] | None:
    """(value, year) of the World Bank's most recent figure for a country, or None. Never raises."""
    global _wb_down_until
    ind = _WB.get(noun)
    if not ind or not iso3 or not web_enabled():
        return None
    if (iso3, ind) in _wb_cache:
        return _wb_cache[(iso3, ind)]
    if time.time() < _wb_down_until:
        return None
    import requests
    try:
        r = requests.get(f"{os.environ.get('BLACKBOX_WORLDBANK_URL', WORLDBANK_URL)}/country/{iso3}/indicator/{ind}",
                         timeout=timeout, headers=_UA, params={"format": "json", "mrnev": 1})
        r.raise_for_status()
        data = r.json()
    except Exception:  # noqa: BLE001  offline or blocked: keep the Wikidata figure
        _wb_down_until = time.time() + 60
        return None
    try:
        row = data[1][0]
        got = (float(row["value"]), str(row["date"])[:4])
        if not got[1].isdigit():
            got = None
    except (IndexError, KeyError, TypeError, ValueError):   # no figure for this country
        got = None
    _wb_cache[(iso3, ind)] = got
    return got


def wikidata(query: str, timeout: float = 4.0) -> dict | None:
    """Current value for a lookup such as "population of India", from Wikidata, or None.

    Returns {"label", "noun", "value", "unit", "year", "id", "iso3"}. Never raises."""
    global _wikidata_down_until
    attr = next((a for a in _ATTRS if re.search(a[0], query, re.I)), None)
    words = [w for w in re.findall(r"[^\W_]+(?:[-'.][^\W_]+)*", query) if w.lower() not in _FILLER and not w.isdigit()]
    if attr is None or not words or not web_enabled() or time.time() < _wikidata_down_until:
        return None
    import requests
    url = os.environ.get("BLACKBOX_WIKIDATA_URL", WIKIDATA_URL)
    _, noun, props, unit, factors = attr
    names = [" ".join(words)] + ([" ".join(w for w in words if w.lower() not in _KIND)] if any(
        w.lower() in _KIND for w in words) and any(w.lower() not in _KIND for w in words) else [])
    try:
        for name in names:
            r = requests.get(url, timeout=timeout, headers=_UA, params={
                "action": "wbsearchentities", "search": name, "language": "en", "type": "item", "limit": 5, "format": "json"})
            r.raise_for_status()
            ids = [h["id"] for h in r.json().get("search", []) if h.get("id")]
            if not ids:
                continue
            r = requests.get(url, timeout=timeout, headers=_UA, params={
                "action": "wbgetentities", "ids": "|".join(ids), "props": "claims|labels", "languages": "en", "format": "json"})
            r.raise_for_status()
            ents = r.json().get("entities") or {}
            for i in ids:                       # best search match that actually has the property
                e = ents.get(i) or {}
                for p in props:
                    got = _latest((e.get("claims") or {}).get(p), factors)
                    if got:
                        label = ((e.get("labels") or {}).get("en") or {}).get("value") or name
                        try:
                            iso3 = e["claims"]["P298"][0]["mainsnak"]["datavalue"]["value"]
                        except (KeyError, IndexError, TypeError):
                            iso3 = None
                        return {"label": label, "noun": noun, "value": got[0], "unit": unit, "year": got[1], "id": i,
                                "iso3": iso3 if isinstance(iso3, str) else None}
    except Exception:  # noqa: BLE001  offline, blocked or a bad reply: fall back to Wikipedia / the model
        _wikidata_down_until = time.time() + 60
    return None


def _num(v: float) -> str:
    """A number a small model reads back unchanged. Never three digits after the point: llama3 read
    "1326.093 million" as 1326093 (it took the point for a thousands separator)."""
    a = abs(v)
    return tasks.fmt_value(round(v, 0 if a >= 1e6 or a >= 1000 and v == int(v) else 1 if a >= 10 else 2))


def rescale(text: str, scale: float, word: str) -> str:
    """Rewrite the first number of a full-digits sentence in the unit the query asked for."""
    if scale == 1.0:
        return text
    m = re.search(r"\d[\d,]*\.?\d*", text)
    if not m:
        return text
    try:
        v = float(m.group(0).replace(",", "").rstrip("."))
    except ValueError:
        return text
    return f"{text[:m.start()]}{_num(v / scale)} {word}{text[m.end():]}"


def _key(query: str) -> str:
    return " ".join(query.lower().split())


def live_search(query: str, ctx: dict) -> str:
    """One lookup of a live custom run: a one-sentence fact from the run's model (+ Wikipedia)."""
    hit = ctx["cache"].get(_key(query))
    if hit:                                  # a fault run repeats the lookups of its reference run
        ctx["lookups"][query] = hit[1]
        return hit[0]
    scale, word = ctx["scale"]
    fact = wikidata(query)
    if fact:                                 # a dated value read by code; the model is not involved
        src, wb = "wikidata", worldbank(fact["iso3"], fact["noun"])
        if wb and wb[1] > (fact["year"] or "0000"):          # a country: take whichever figure is newer
            src, fact = "worldbank", {**fact, "value": wb[0], "year": wb[1]}
        ctx["lookups"][query] = {"source": src, "title": f"{fact['label']} ({fact['iso3'] if src == 'worldbank' else fact['id']})",
                                 "as_of": fact["year"]}
        return (f"{fact['label']} has a {fact['noun']} of {_num(fact['value'] / scale)} "
                f"{word + ' ' if word else ''}{fact['unit']}.")
    ref = wikipedia(query)
    r = ctx["llm"].complete(LOOKUP.format(query=query, reference=ref[1] if ref else "(none available)"))
    text = full_digits(next((ln.strip() for ln in (r.text or "").splitlines() if ln.strip()), "")[:300])
    if r.error or not any(ch.isdigit() for ch in text) or "no results found" in text.lower():
        raise tools.ToolError("empty_result")
    ctx["lookups"][query] = {"source": "wikipedia+model", "title": ref[0]} if ref else {"source": "model", "title": None}
    return rescale(text, scale, word)


@contextlib.contextmanager
def live_scope(llm, cache: dict | None = None, task: str = ""):
    """While open, the search tool of the agent running in this context does live lookups."""
    ctx = {"llm": llm, "cache": dict(cache or {}), "lookups": {}, "scale": task_scale(task)}
    token = _live.set(ctx)
    try:
        yield ctx
    finally:
        _live.reset(token)


def search_scope(run: dict, llm):
    """The search a replay of `run` must use: live for a live custom run, else the normal tool."""
    if (run.get("custom") or {}).get("search") == "live":
        return live_scope(llm, task=run.get("task") or "")
    return contextlib.nullcontext({"lookups": {}})


def lookup_cache(run: dict) -> dict:
    """query -> (output, source) for the searches of a live run that found something."""
    seen = (run.get("custom") or {}).get("lookups") or {}
    return {_key(s["input"]): (s["output"], seen[s["input"]]) for s in run["steps"]
            if s["actor"] == "search_agent" and not s.get("error") and s["input"] in seen}


if not getattr(tools.TOOLS["search"], "_blackbox_custom", False):
    _local_search = tools.TOOLS["search"]

    def _search(query: str) -> str:
        ctx = _live.get()
        return _local_search(query) if ctx is None else live_search(query, ctx)

    _search._blackbox_custom = True
    tools.TOOLS["search"] = _search      # identical to the local tool outside a live custom run


class CustomError(ValueError):
    """Bad request for a custom run; the message is shown to the user as is."""


def judge(answer: str, steps: list[dict], expected: float | None = None, tolerance: float = 0.06) -> tuple[str, str]:
    """(outcome, how it was judged)."""
    value = tasks.parse_answer(answer)
    if expected is not None:
        ok = value is not None and abs(value - expected) <= max(tolerance, 0.005 * abs(expected))
        return ("success" if ok else "fail"), "expected"
    clean = value is not None and not any(s.get("error") for s in steps)
    return ("success" if clean else "fail"), "completed"


def expected_from_facts(query: str) -> float | None:
    """Expected answer when the query is one of the question shapes the local fact table can answer."""
    t = tasks.get_task_by_text(query)
    if t:
        return t.expected
    spec = tasks.parse_task(query)
    if spec and all(e in tasks.CORPUS.get(spec.attr, {}) for e in spec.entities) and \
            len(spec.entities) >= (1 if spec.op in ("add", "avg") else 2):
        try:
            return tasks.compute_expected(spec)
        except (ZeroDivisionError, IndexError):
            return None
    return None


def ollama_models(timeout: float = 0.8) -> list[str] | None:
    """Models installed in the local Ollama, or None when Ollama is not reachable."""
    import requests
    host = os.environ.get("BLACKBOX_OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    try:
        r = requests.get(f"{host}/api/tags", timeout=timeout)
        r.raise_for_status()
        return sorted(m["name"] for m in r.json().get("models", []))
    except Exception:  # noqa: BLE001
        return None


def supported_models() -> dict:
    """What can actually run right now. Hosted providers appear only when their key is set."""
    out = [{"id": m, "group": "Offline mock", "available": True,
            "note": "Rule-based stand-in. Only understands the four built-in question shapes."}
           for m in ("mock-large", "mock-small")]
    installed = ollama_models()
    for m in installed or []:
        out.append({"id": m, "group": "Ollama (local)", "available": True, "note": "Local model."})
    providers = []
    for prefix, (base, key_env) in llm_mod.PROVIDERS.items():
        ready = bool(apikeys.get(key_env)) and bool(base or apikeys.get("BLACKBOX_LLM_BASE_URL"))
        providers.append({"prefix": prefix, "key_env": key_env, "available": ready,
                          "user_key": apikeys.has_user_key(prefix)})
    ready = {p["prefix"] for p in providers if p["available"]}
    with store._conn() as c:   # hosted models that already produced a successful run here
        used = [r[0] for r in c.execute("SELECT DISTINCT model FROM runs WHERE outcome='success' AND model LIKE '%:%'")]
    for m in sorted(used):
        if m.partition(":")[0] in ready:
            out.append({"id": m, "group": "Hosted API", "available": True, "note": "Used successfully before on this machine."})
    return {"models": out, "providers": providers, "ollama_running": installed is not None}


def check_model(model: str, query: str) -> None:
    """Raise CustomError unless this model can run this query right now."""
    model = (model or "").strip()
    if not model:
        raise CustomError("Pick a model.")
    if model.startswith("mock"):
        if model not in ("mock-large", "mock-small"):
            raise CustomError(f"Unknown mock model `{model}`. Use mock-large or mock-small.")
        if tasks.parse_task(query) is None:
            raise CustomError("The mock models only understand the four built-in question shapes (combined population, "
                              "how much taller, how many times larger an area, average river length). "
                              "Pick a real model for a free-form query.")
        if expected_from_facts(query) is None:
            raise CustomError("The mock models can only look things up in the local fact table, and this query names "
                              "something that is not in it. Pick a real model, or use names from the fact table.")
        return
    prefix, sep, name = model.partition(":")
    if sep and prefix in llm_mod.PROVIDERS:
        base, key_env = llm_mod.PROVIDERS[prefix]
        if not name:
            raise CustomError(f"Add the model name after `{prefix}:`.")
        if not apikeys.get(key_env):
            raise CustomError(f"No API key for `{prefix}:` models. Add yours under \"API keys\" in the sidebar "
                              f"(or set {key_env} on the server).")
        if not base and not apikeys.get("BLACKBOX_LLM_BASE_URL"):
            raise CustomError("`api:` models need a base URL. Add it with your key under \"API keys\" "
                              "(or set BLACKBOX_LLM_BASE_URL on the server).")
        return
    installed = ollama_models()
    if installed is None:
        raise CustomError(f"`{model}` would run on Ollama, but Ollama is not reachable on this machine. "
                          "Start Ollama, or use a hosted model such as groq:<model>.")
    name = model.removeprefix("ollama:")
    if name not in installed and f"{name}:latest" not in installed:
        raise CustomError(f"Ollama does not have `{name}` installed. Run: ollama pull {name}")


def check_fault(fault_type: str | None, fault_step: int | None) -> dict:
    if fault_type not in inject.FAULT_TYPES:
        raise CustomError(f"Pick a fault type: {', '.join(inject.FAULT_TYPES)}.")
    if fault_step is not None and not (1 <= int(fault_step) <= 60):
        raise CustomError("Fault step must be between 1 and 60, or left empty for the first step the fault fits.")
    return {"type": fault_type, "step_no": int(fault_step) if fault_step else None, "seed": 1}


def _execute(query, model, framework, fault, expected, tolerance, meta, live=False, cache=None):
    llm = llm_mod.get_llm(model)
    agent = get_agent(framework, llm)
    rec = Recorder(framework, model.removeprefix("ollama:"), query, None, fault=fault)
    with (live_scope(llm, cache, query) if live else contextlib.nullcontext({"lookups": {}})) as ctx:
        answer = agent.run(query, rec)
    outcome, how = judge(answer, rec.steps, expected, tolerance)
    run = rec.finish(answer, outcome)
    run["custom"] = {**meta, "expected": expected, "tolerance": tolerance, "judged_by": how,
                     "search": "live" if live else "fact_table", "lookups": ctx["lookups"]}
    store.save_run(run)
    return run


def record_custom(query: str, model: str, framework: str = "custom", fault: dict | None = None,
                  expected: float | None = None, tolerance: float = 0.06) -> dict:
    """Run the agent on the user's query and save it. Returns {"run", "reference"}.

    The query is passed to the agent exactly as typed. `fault` None means no fault is injected."""
    query = (query or "").strip()
    if len(query) < 3:
        raise CustomError("Type a query first.")
    if len(query) > MAX_QUERY:
        raise CustomError(f"The query is too long ({len(query)} characters; the limit is {MAX_QUERY}).")
    if framework not in FRAMEWORKS:
        raise CustomError(f"Unknown framework `{framework}`. Supported: {', '.join(FRAMEWORKS)}.")
    check_model(model, query)
    facts = expected_from_facts(query)
    live = facts is None and not model.startswith("mock")     # see "Search in a custom run" above
    source = "user" if expected is not None else None
    if expected is None and facts is not None:
        expected, source = facts, "fact_table"
    reference = None
    if fault and expected is None:
        # nothing to compare a broken run with: record the same query without the fault first
        reference = _execute(query, model, framework, None, None, tolerance,
                             {"mode": "reference", "expected_source": None}, live)
        value = tasks.parse_answer(reference.get("final_answer") or "")
        if reference["outcome"] == "success" and value is not None:
            expected, source = value, "reference_run"
    meta = {"mode": "fault" if fault else "normal", "expected_source": source,
            "reference_run_id": reference["run_id"] if reference else None}
    cache = lookup_cache(reference) if reference and live else None
    return {"run": _execute(query, model, framework, fault, expected, tolerance, meta, live, cache),
            "reference": reference}


def knowledge() -> list[dict]:
    """What the local fact table holds (used for the built-in question shapes and the mock models)."""
    units = {"population": "millions of people", "area": "thousand square kilometers", "height": "meters", "length": "kilometers"}
    return [{"attribute": a, "unit": units.get(a, ""), "entities": sorted(ents)} for a, ents in tasks.CORPUS.items()]


def lookups_missed(run: dict) -> int:
    """Search steps that found nothing without a fault being injected there."""
    hit = (run.get("fault") or {}).get("step_no") if (run.get("fault") or {}).get("applied") else None
    return sum(1 for s in run["steps"] if s.get("error") == "empty_result" and s["step_no"] != hit
               and s.get("output") == tools.NOT_FOUND)
