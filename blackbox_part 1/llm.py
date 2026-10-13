"""LLM clients.

  mock-large / mock-small       MockLLM, deterministic and offline (tests, CI)
  llama3, mistral, ollama:<m>   OllamaLLM, a free local model (http://localhost:11434)
  groq:<m>, openai:<m>, gemini:<m>, openrouter:<m>
                                OpenAICompatLLM, a hosted model over the OpenAI-style chat API.
                                Needs the provider's key in the environment (see PROVIDERS).
  api:<m>                       any other OpenAI-style server: set BLACKBOX_LLM_BASE_URL and
                                BLACKBOX_LLM_API_KEY
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass

import apikeys
import tasks


@dataclass
class LLMResult:
    text: str
    tokens: int
    error: str | None = None


def est_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def extract_json(text: str):
    """First JSON object found in an LLM reply (models love wrapping JSON in prose)."""
    if not text:
        return None
    a, b = text.find("{"), text.rfind("}")
    if a == -1 or b <= a:
        return None
    try:
        return json.loads(text[a:b + 1])
    except json.JSONDecodeError:
        return None


class OllamaLLM:
    """Free local model via Ollama (http://localhost:11434)."""

    def __init__(self, model: str, host: str | None = None, temperature: float = 0.2,
                 timeout: int = 180):
        self.name = model
        self.host = (host or os.environ.get("BLACKBOX_OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.temperature, self.timeout = temperature, timeout

    def complete(self, prompt: str, json_mode: bool = False) -> LLMResult:
        import requests
        body = {"model": self.name, "prompt": prompt, "stream": False,
                "options": {"temperature": self.temperature}}
        if json_mode:
            body["format"] = "json"
        try:
            r = requests.post(f"{self.host}/api/generate", json=body, timeout=self.timeout)
            r.raise_for_status()
            d = r.json()
        except Exception as e:  # recorded as a step error rather than crashing the run
            return LLMResult("", 0, f"llm_error: {type(e).__name__}")
        text = d.get("response", "")
        toks = d.get("prompt_eval_count", 0) + d.get("eval_count", 0) or est_tokens(prompt + text)
        return LLMResult(text.strip(), toks)


# provider prefix -> (base URL, environment variable holding the API key)
PROVIDERS = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "api": (None, "BLACKBOX_LLM_API_KEY"),
}


class OpenAICompatLLM:
    """Hosted model over the OpenAI-style /chat/completions API (OpenAI, Groq, Gemini, OpenRouter, ...)."""

    def __init__(self, name: str, timeout: int = 120, retries: int = 3):
        provider, _, model = name.partition(":")
        base, key_env = PROVIDERS[provider]
        self.name, self.model, self.key_env = name, model, key_env
        self.base = (apikeys.get("BLACKBOX_LLM_BASE_URL") if provider == "api" else base) or ""
        self.timeout, self.retries = timeout, retries

    def complete(self, prompt: str, json_mode: bool = False) -> LLMResult:
        import time

        import requests
        key = apikeys.get(self.key_env)   # the visitor's own key (dashboard "API keys"), else the env var
        if not self.base:
            return LLMResult("", 0, "llm_error: BLACKBOX_LLM_BASE_URL not set")
        if not key and "localhost" not in self.base and "127.0.0.1" not in self.base:
            return LLMResult("", 0, f"llm_error: {self.key_env} not set")
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}]}
        if os.environ.get("BLACKBOX_LLM_TEMPERATURE"):
            body["temperature"] = float(os.environ["BLACKBOX_LLM_TEMPERATURE"])
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        err = "llm_error: no attempt"
        for attempt in range(self.retries):
            try:
                r = requests.post(f"{self.base.rstrip('/')}/chat/completions", json=body, timeout=self.timeout,
                                  headers={"Authorization": f"Bearer {key}"})
            except Exception as e:  # noqa: BLE001  recorded as a step error rather than crashing the run
                err = f"llm_error: {type(e).__name__}"
                continue
            if r.status_code == 429 or r.status_code >= 500:   # rate limit or server hiccup: wait and retry
                err = f"llm_error: HTTP {r.status_code}"
                try:
                    wait = float(r.headers.get("retry-after", 2 ** attempt))
                except ValueError:
                    wait = 2 ** attempt
                if attempt + 1 < self.retries:
                    time.sleep(min(wait, 20))
                continue
            if r.status_code >= 400:
                return LLMResult("", 0, f"llm_error: HTTP {r.status_code}")
            try:
                d = r.json()
                text = d["choices"][0]["message"]["content"] or ""
            except Exception:  # noqa: BLE001
                return LLMResult("", 0, "llm_error: bad response")
            toks = (d.get("usage") or {}).get("total_tokens") or est_tokens(prompt + text)
            return LLMResult(text.strip(), int(toks))
        return LLMResult("", 0, err)


class MockLLM:
    """Rule-based stand-in that behaves like a model reading our prompts.

    style 'small'/'large' phrase queries differently (two distinct "models" for the
    unseen-model test). 'small' occasionally misreads a number (deterministic per prompt),
    giving natural failures that have no known faulty step.
    """

    def __init__(self, name: str = "mock-large"):
        self.name = name
        self.small = "small" in name
        self.flaky = 0.04 if self.small else 0.0

    def _flake(self, prompt: str) -> bool:
        h = int(hashlib.sha256(prompt.encode()).hexdigest(), 16) % 1000
        return h < self.flaky * 1000

    def complete(self, prompt: str, json_mode: bool = False) -> LLMResult:
        if prompt.startswith("### PLAN"):
            text = self._plan(prompt)
        elif prompt.startswith("### READ"):
            text = self._read(prompt)
        elif prompt.startswith("### ANSWER"):
            text = self._answer(prompt)
        else:
            text = ""
        return LLMResult(text, est_tokens(prompt) + est_tokens(text))

    def _plan(self, prompt: str) -> str:
        m = re.search(r"^Task: (.*)$", prompt, re.M)
        spec = tasks.parse_task(m.group(1)) if m else None
        if not spec:
            return "I cannot plan this task."
        steps = []
        for e in spec.entities:
            q = f"{spec.attr} of {e}" if self.small else f"{e} total {spec.attr}"
            steps.append({"tool": "search", "query": q})
        n = len(spec.entities)
        ph = [f"{{{i}}}" for i in range(n)]
        expr = {"add": "+".join(ph), "sub": f"{ph[0]}-{ph[1]}", "div": f"{ph[0]}/{ph[1]}",
                "avg": f"({'+'.join(ph)})/{n}"}[spec.op]
        return json.dumps({"steps": steps, "expression": expr})

    def _read(self, prompt: str) -> str:
        m = re.search(r"^Text: (.*)$", prompt, re.M | re.S)
        text = m.group(1).strip() if m else ""
        v = tasks.parse_number(text)
        if v is None:
            return "unknown"
        if self._flake(prompt):  # simulated misreading
            v = v * 10
        return tasks.fmt_value(v)

    def _answer(self, prompt: str) -> str:
        m = re.search(r"^Calculation result: (.*)$", prompt, re.M)
        v = tasks.parse_number(m.group(1)) if m else None
        if v is None:
            return "Final answer: unknown"
        t = re.search(r"^Task: (.*)$", prompt, re.M)
        digits = 1 if (t and "decimal place" in t.group(1)) else 2
        return f"Final answer: {tasks.fmt_value(round(v, digits))}"


def get_llm(model: str):
    """'mock-*' -> MockLLM, '<provider>:<model>' -> hosted API, 'ollama:xyz' or anything else -> Ollama."""
    if model.startswith("mock"):
        return MockLLM(model)
    if model.partition(":")[0] in PROVIDERS and ":" in model:
        return OpenAICompatLLM(model)
    return OllamaLLM(model.removeprefix("ollama:"))
