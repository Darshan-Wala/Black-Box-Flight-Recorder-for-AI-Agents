"""Bring-your-own-key: API keys that a dashboard visitor enters in the browser.

The dashboard keeps the keys in the visitor's browser and sends them with each request in one
header, `X-Blackbox-Keys` (JSON, e.g. {"gemini": "AIza...", "groq": "gsk_..."}). The middleware
below puts them in a context variable for that request only, so concurrent visitors never see
each other's keys. Nothing here writes a key to disk, the run store or the logs.

Code that needs a key calls `get("GEMINI_API_KEY")`: the visitor's key when they sent one,
else the server's environment variable (so local use with `$env:GROQ_API_KEY` works as before).
"""
from __future__ import annotations

import contextvars
import ipaddress
import json
import os
from urllib.parse import urlparse

HEADER = "x-blackbox-keys"
BASE_URL_ENV = "BLACKBOX_LLM_BASE_URL"
# provider prefix -> environment variable holding its key (same names as llm.PROVIDERS)
KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "groq": "GROQ_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "api": "BLACKBOX_LLM_API_KEY",
}
_MAX_LEN = 512

_user: contextvars.ContextVar[dict] = contextvars.ContextVar("blackbox_user_keys", default={})


class KeysError(ValueError):
    pass


def get(env_name: str) -> str:
    """The visitor's value for this variable if they sent one, else the server's environment."""
    return _user.get().get(env_name) or os.environ.get(env_name, "")


def user_keys() -> dict:
    """{env_name: value} the visitor sent with this request (empty outside a request)."""
    return dict(_user.get())


def has_user_key(prefix: str) -> bool:
    return bool(_user.get().get(KEY_ENV.get(prefix, "")))


def check_base_url(url: str) -> str:
    """A visitor-supplied base URL for `api:<model>` must be public HTTPS (no localhost or
    private addresses, so the server cannot be pointed at its own network)."""
    url = (url or "").strip().rstrip("/")
    p = urlparse(url)
    host = (p.hostname or "").lower()
    if p.scheme != "https" or not host:
        raise KeysError("The base URL must start with https://")
    if host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
        raise KeysError("The base URL must be a public address.")
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return url
    if not ip.is_global:
        raise KeysError("The base URL must be a public address.")
    return url


def _clean_key(v) -> str:
    v = str(v or "").strip()
    if len(v) > _MAX_LEN or any(c.isspace() or ord(c) < 33 or ord(c) > 126 for c in v):
        raise KeysError("An API key may only contain printable characters without spaces.")
    return v


def parse(raw: str | None) -> dict:
    """Header value -> {env_name: value}. Unknown providers are ignored; bad values raise."""
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except ValueError:
        raise KeysError(f"{HEADER} is not valid JSON")
    if not isinstance(data, dict):
        raise KeysError(f"{HEADER} must be a JSON object")
    out = {}
    for prefix, env in KEY_ENV.items():
        key = _clean_key(data.get(prefix))
        if key:
            out[env] = key
    if data.get("api_base"):
        out[BASE_URL_ENV] = check_base_url(str(data["api_base"]))
    return out


class KeysMiddleware:
    """Pure ASGI middleware: reads the header and scopes the keys to this request."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        raw = None
        for k, v in scope.get("headers") or []:
            if k.decode("latin-1").lower() == HEADER:
                raw = v.decode("latin-1")
                break
        try:
            keys = parse(raw)
        except KeysError as e:
            body = json.dumps({"detail": f"API keys: {e}"}).encode()
            await send({"type": "http.response.start", "status": 400,
                        "headers": [(b"content-type", b"application/json")]})
            await send({"type": "http.response.body", "body": body})
            return
        token = _user.set(keys)
        try:
            await self.app(scope, receive, send)
        finally:
            _user.reset(token)
