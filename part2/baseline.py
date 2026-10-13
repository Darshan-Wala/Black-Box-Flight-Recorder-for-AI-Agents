"""Baselines to beat.

heuristic - no model: blame the first step with an error, else the last step.
llm_judge - an LLM reads the whole log and names the bad step. Ollama model name (llama3), or a
            hosted model as groq:<model>, openai:<model>, gemini:<model>, openrouter:<model>.
"""
import contextvars
import json
import os
import time
import urllib.error
import urllib.request

from .signals import _err


def heuristic(run, k=3):
    steps = run["steps"]
    order = [s["step_no"] for s in steps if _err(s)]
    order += [s["step_no"] for s in reversed(steps) if s["step_no"] not in order]
    return order[:k]


def render(run, width=300):
    lines = [f"TASK: {run.get('task')}", "The run FAILED. Steps:"]
    for s in run["steps"]:
        lines.append(f"[{s['step_no']}] {s.get('actor')} ({s.get('kind')}) depends_on={s.get('depends_on')}"
                     f" error={s.get('error')}\n  input: {str(s.get('input'))[:width]}"
                     f"\n  output: {str(s.get('output'))[:width]}")
    return "\n".join(lines)


# provider prefix -> (base URL, environment variable holding the API key); same list as Part 1's llm.py
PROVIDERS = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "api": (None, "BLACKBOX_LLM_API_KEY"),
}






# API keys sent with one /explain request ("api_keys": {"GEMINI_API_KEY": ...}); they override the
# environment for that request only (set by diagnose_api.explain, never stored)
API_KEYS: contextvars.ContextVar = contextvars.ContextVar("blackbox_api_keys", default={})


def _env(name, default=""):
    return API_KEYS.get().get(name) or os.environ.get(name, default)


def _ask(llm, prompt, url):
    """Send one prompt, return the reply text. `groq:<model>` etc. go to that provider's
    OpenAI-style chat API; anything else goes to Ollama."""
    provider, _, name = llm.partition(":")
    if name and provider in PROVIDERS:
        base, key_env = PROVIDERS[provider]
        base = base or _env("BLACKBOX_LLM_BASE_URL")
        if not base or not _env(key_env):
            raise SystemExit(f"Set {key_env}" + ("" if base else " and BLACKBOX_LLM_BASE_URL") + f" to use {llm}")
        body = json.dumps({"model": name, "messages": [{"role": "user", "content": prompt}],
                           "response_format": {"type": "json_object"}}).encode()
        req = urllib.request.Request(base.rstrip("/") + "/chat/completions", body,
                                     {"Content-Type": "application/json",
                                      "Authorization": "Bearer " + _env(key_env),
                                      "User-Agent": "python-urllib"})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=300) as resp:
                    return json.loads(resp.read())["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as e:
                if e.code not in (429, 500, 502, 503) or attempt == 3:
                    raise
                time.sleep(min(20, 2 ** attempt * 2))   # rate limit: wait and retry
    body = json.dumps({"model": llm.removeprefix("ollama:"), "prompt": prompt, "stream": False, "format": "json",
                       "options": {"temperature": 0, "seed": 0}}).encode()
    req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())["response"]


def llm_judge(run, llm="llama3", k=3, url="http://localhost:11434/api/generate", failures=None):
    """Ask the LLM for up to k suspect steps. A call that fails or returns nothing usable
    counts as a miss and is noted in `failures`."""
    prompt = (render(run) + "\n\nExactly one step is the root cause of the failure. "
              f"Reply with JSON only: {{\"steps\": [up to {k} step numbers, most likely first]}}")
    hosted = llm.partition(":")[0] in PROVIDERS and ":" in llm
    try:
        text = _ask(llm, prompt, url)
        steps = json.loads(text[text.find("{"): text.rfind("}") + 1]).get("steps", [])
        return [int(x) for x in (steps if isinstance(steps, list) else [steps])][:k]
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")

        if hosted:
            raise SystemExit(
                f"Groq/API Error: HTTP {e.code}\n"
                f"Response: {error_body}"
            )

        raise SystemExit(
            f"The LLM rejected the request ({e.code}). "
            f"Check the model name. Details: {error_body}"
        )
    except urllib.error.URLError as e:
        if not hosted and isinstance(getattr(e, "reason", None), ConnectionRefusedError):
            raise SystemExit(f"Cannot reach Ollama at {url}. Start it (ollama serve) and run: ollama pull {llm}")
        reason = e
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, OSError) as e:
        reason = e
    if failures is not None:
        failures.append((run.get("run_id"), str(reason)))
    return []
