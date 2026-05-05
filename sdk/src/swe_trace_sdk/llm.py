"""Optional LLM provider interface and caching for semantic equivalence.

This module is **fully optional** — the core SDK operates without it.
It is used by :class:`~swe_trace_sdk.equivalence.StateEquivalence` when
``use_llm=True`` and no custom ``llm_fn`` is provided.

Public surface
--------------
- :func:`llm_equivalence_check` — compare two states via LLM.
- :func:`get_client` — get a configured OpenAI-compatible client.
- :func:`get_model_and_client` — convenience for (client, model, temperature).
- :class:`LLMCache` — lightweight SQLite cache for LLM completions.
- :func:`cached_completion` — invoke an LLM with transparent caching.

Environment variables
~~~~~~~~~~~~~~~~~~~~~
Provider configuration (evaluated lazily, only when an LLM call is made):

* ``SWE_TRACE_LLM`` or ``DEFAULT_LLM`` — ``"provider:model[:temperature]"``
  (e.g. ``"openai:gpt-4o:0.3"``).
* ``OPENAI_API_KEY``, ``OPENAI_BASE_URL`` — for the *openai* provider.
* ``AZURE_OPENAI_API_KEY``, ``AZURE_OPENAI_ENDPOINT``,
  ``AZURE_OPENAI_API_VERSION`` — for the *azure* provider.* ``ANTHROPIC_API_KEY`` (or ``AZURE_OPENAI_API_KEY``),
  ``ANTHROPIC_BASE_URL`` (or ``AZURE_OPENAI_ENDPOINT``)
  — for the *anthropic* provider (requires ``pip install anthropic``).
Cache configuration:

* ``LLM_CACHE_PATH`` — SQLite DB path (default ``./llm_cache.db``).
* ``LLM_CACHE_DISABLE`` — truthy to skip reads (still writes).
* ``LLM_CACHE_TTL_SECONDS`` — TTL in seconds.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .models import State

logger = logging.getLogger(__name__)

# Track which models don't support certain parameters to avoid repeated 400 errors
_MODEL_PARAM_CACHE: Dict[str, set] = {}

__all__ = [
    "llm_equivalence_check",
    "llm_semantic_content_check",
    "get_client",
    "get_model_and_client",
    "LLMCache",
    "cached_completion",
]


# ---------------------------------------------------------------------------
# Provider client helpers
# ---------------------------------------------------------------------------

_CLIENT_CACHE: Dict[str, Any] = {}
_CONFIG_LOGGED: bool = False


def _truthy(val: Optional[str]) -> bool:
    return bool(val) and val.lower() in {"1", "true", "yes", "on"}


def _safe_llm_call(
    client: Any,
    model: str,
    messages: list,
    max_tokens: int = 200,
    temperature: float = 0.1,
    retries: int = 3,
    timeout: float = 30.0,
) -> Any:
    """Make an LLM call with model-appropriate parameters.

    Caches which parameters work for each model to avoid repeated 400 errors.
    Includes retry logic for transient network errors.
    """
    global _MODEL_PARAM_CACHE

    def _make_call(kwargs: Dict[str, Any]) -> Any:
        last_error = None
        for attempt in range(retries):
            try:
                return client.chat.completions.create(**kwargs, timeout=timeout)
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                is_transient = any(err in error_str for err in [
                    "timeout", "connection", "network", "reset", "eof",
                    "ssl", "socket", "temporarily unavailable", "503", "502", "429",
                ])
                if is_transient and attempt < retries - 1:
                    wait_time = (attempt + 1) * 2
                    logger.warning("Transient error on attempt %d/%d, retrying in %ds: %s",
                                   attempt + 1, retries, wait_time, e)
                    time.sleep(wait_time)
                    continue
                raise
        raise last_error  # type: ignore[misc]

    cache_key = model
    if cache_key in _MODEL_PARAM_CACHE:
        working_params = _MODEL_PARAM_CACHE[cache_key]
        kwargs: Dict[str, Any] = {"model": model, "messages": messages}
        if "max_completion_tokens" in working_params:
            kwargs["max_completion_tokens"] = max_tokens
        elif "max_tokens" in working_params:
            kwargs["max_tokens"] = max_tokens
        if "temperature" in working_params:
            kwargs["temperature"] = temperature
        return _make_call(kwargs)

    param_combinations = [
        {"max_completion_tokens": max_tokens},
        {"max_tokens": max_tokens},
        {"temperature": temperature, "max_completion_tokens": max_tokens},
        {"temperature": temperature, "max_tokens": max_tokens},
        {},
    ]

    last_error: Optional[Exception] = None
    for params in param_combinations:
        try:
            kwargs = {"model": model, "messages": messages}
            kwargs.update(params)
            response = _make_call(kwargs)
            _MODEL_PARAM_CACHE[cache_key] = set(params.keys())
            return response
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            if "400" not in str(e) and "invalid" not in error_str and "unsupported" not in error_str:
                raise

    raise last_error  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Anthropic → OpenAI adapter
# ---------------------------------------------------------------------------
# All existing callers use client.chat.completions.create(...) and read
# response.choices[0].message.content.  The adapter translates Anthropic's
# messages.create() response into that shape so every call-site works
# without changes.
# ---------------------------------------------------------------------------

class _AnthropicUsageShim:
    """Mimic openai Usage object from Anthropic usage."""
    __slots__ = ("prompt_tokens", "completion_tokens", "total_tokens")

    def __init__(self, input_tokens: int, output_tokens: int):
        self.prompt_tokens = input_tokens
        self.completion_tokens = output_tokens
        self.total_tokens = input_tokens + output_tokens


class _AnthropicMessageShim:
    __slots__ = ("content",)

    def __init__(self, content: str):
        self.content = content


class _AnthropicChoiceShim:
    __slots__ = ("message", "finish_reason")

    def __init__(self, message: _AnthropicMessageShim, finish_reason: str):
        self.message = message
        self.finish_reason = finish_reason


class _AnthropicResponseShim:
    """Looks like an OpenAI ChatCompletion."""

    def __init__(self, choices, usage, model):
        self.choices = choices
        self.usage = usage
        self.model = model

    def model_dump(self):
        return {
            "choices": [
                {
                    "message": {"content": c.message.content},
                    "finish_reason": c.finish_reason,
                }
                for c in self.choices
            ],
            "usage": {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "total_tokens": self.usage.total_tokens,
            },
            "model": self.model,
        }


class _AnthropicCompletionsProxy:
    """Proxy for client.chat.completions that routes to Anthropic."""

    def __init__(self, anthropic_client):
        self._client = anthropic_client

    def create(self, *, model, messages, **kwargs):
        # Separate system messages (Anthropic takes system as a top-level param)
        system_parts = []
        user_messages = []
        for msg in messages:
            if msg["role"] == "system":
                text = msg["content"]
                if isinstance(text, list):  # multimodal content blocks
                    text = " ".join(
                        p["text"] for p in text if isinstance(p, dict) and p.get("type") == "text"
                    )
                system_parts.append(str(text))
            else:
                user_messages.append(msg)

        # Build Anthropic kwargs
        anthro_kwargs: dict = {
            "model": model,
            "messages": user_messages,
            "max_tokens": (
                kwargs.get("max_tokens")
                or kwargs.get("max_completion_tokens")
                or 1024
            ),
        }
        if system_parts:
            anthro_kwargs["system"] = "\n\n".join(system_parts)
        if "temperature" in kwargs:
            anthro_kwargs["temperature"] = kwargs["temperature"]

        response = self._client.messages.create(**anthro_kwargs)

        # Map content
        content = ""
        if response.content:
            block = response.content[0]
            content = getattr(block, "text", str(block))

        # Map finish_reason
        _STOP_MAP = {
            "end_turn": "stop",
            "max_tokens": "length",
            "stop_sequence": "stop",
        }
        finish_reason = _STOP_MAP.get(response.stop_reason, response.stop_reason)

        usage = _AnthropicUsageShim(
            response.usage.input_tokens,
            response.usage.output_tokens,
        )
        choice = _AnthropicChoiceShim(_AnthropicMessageShim(content), finish_reason)
        return _AnthropicResponseShim([choice], usage, response.model)


class _AnthropicChatProxy:
    """Proxy for client.chat (has .completions)."""

    def __init__(self, anthropic_client):
        self.completions = _AnthropicCompletionsProxy(anthropic_client)


class _AnthropicOpenAIAdapter:
    """Wraps an Anthropic client to expose an OpenAI-compatible interface.

    Usage::

        adapter = _AnthropicOpenAIAdapter(anthropic_client)
        resp = adapter.chat.completions.create(model=..., messages=...)
        print(resp.choices[0].message.content)
    """

    def __init__(self, anthropic_client):
        self._client = anthropic_client
        self.chat = _AnthropicChatProxy(anthropic_client)


def get_client(provider: str) -> Any:
    """Return a cached OpenAI-compatible client for *provider*.

    Supported providers: ``openai``, ``azure``, ``anthropic``.

    Raises :class:`ImportError` if the required package is not installed.
    """
    if provider in _CLIENT_CACHE:
        return _CLIENT_CACHE[provider]

    try:
        from openai import OpenAI, AzureOpenAI  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "The 'openai' package is required for LLM features. "
            "Install it with: pip install openai"
        ) from exc

    provider = provider.strip().lower()

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        client = OpenAI(api_key=api_key, base_url=base_url, max_retries=1)

    elif provider == "azure":
        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            max_retries=1,
        )

    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
        base_url = os.getenv("ANTHROPIC_BASE_URL") or os.getenv("AZURE_OPENAI_ENDPOINT")
        if not api_key:
            raise ValueError(
                "Neither ANTHROPIC_API_KEY nor AZURE_OPENAI_API_KEY is set. "
                "Set one of them for the 'anthropic' provider."
            )
        if not base_url:
            raise ValueError(
                "Neither ANTHROPIC_BASE_URL nor AZURE_OPENAI_ENDPOINT is set. "
                "Set one of them for the 'anthropic' provider."
            )
        try:
            from anthropic import Anthropic  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "The 'anthropic' package is required for the anthropic provider. "
                "Install it with: pip install anthropic"
            ) from exc
        raw_client = Anthropic(api_key=api_key, base_url=base_url)
        client = _AnthropicOpenAIAdapter(raw_client)

    else:
        raise ValueError(f"Unsupported LLM provider: {provider!r}")

    _CLIENT_CACHE[provider] = client
    return client


def _parse_llm_spec(spec: str) -> Tuple[str, str, float]:
    """Parse ``"provider:model[:temperature]"`` → (provider, model, temp)."""
    parts = spec.strip().split(":")
    if len(parts) < 2:
        raise ValueError(f"Invalid LLM spec (expected provider:model[:temp]): {spec!r}")
    provider = parts[0]
    temp = 0.3
    model_parts = parts[1:]
    if len(parts) > 2:
        try:
            temp = float(parts[-1])
            model_parts = parts[1:-1]
        except ValueError:
            pass
    model = ":".join(model_parts)
    return provider, model, temp


def get_model_and_client(
    prefix: str = "SWE_TRACE",
) -> Tuple[Any, str, float]:
    """Return ``(client, model, temperature)`` from environment.

    Looks up ``<PREFIX>_LLM`` then ``DEFAULT_LLM``.
    """
    env_var = f"{prefix.upper()}_LLM"
    spec = os.getenv(env_var) or os.getenv("DEFAULT_LLM")
    if not spec:
        raise ValueError(f"Neither {env_var} nor DEFAULT_LLM is set")
    provider, model, temp = _parse_llm_spec(spec)
    global _CONFIG_LOGGED
    if not _CONFIG_LOGGED:
        cache_path = os.path.abspath(os.getenv("LLM_CACHE_PATH", "llm_cache.db"))
        logger.info("LLM config: provider=%s  model=%s  temperature=%.1f  (from %s)  cache=%s",
                    provider, model, temp,
                    env_var if os.getenv(env_var) else "DEFAULT_LLM",
                    cache_path)
        _CONFIG_LOGGED = True
    client = get_client(provider)
    return client, model, temp


# ---------------------------------------------------------------------------
# SQLite cache
# ---------------------------------------------------------------------------

class LLMCache:
    """Lightweight SQLite cache for LLM completions."""

    _instance: Optional["LLMCache"] = None
    _lock = threading.Lock()

    def __init__(
        self,
        path: str = "llm_cache.db",
        *,
        ttl: Optional[int] = None,
        disabled: bool = False,
    ) -> None:
        self.path = path
        self.ttl = ttl
        self.disabled = disabled
        self._conn: Optional[sqlite3.Connection] = None
        self._write_lock = threading.Lock()

    # ---- singleton ----

    @classmethod
    def get_default(cls) -> "LLMCache":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    path = os.getenv("LLM_CACHE_PATH", "llm_cache.db")
                    disabled = _truthy(os.getenv("LLM_CACHE_DISABLE"))
                    ttl_env = os.getenv("LLM_CACHE_TTL_SECONDS")
                    ttl = int(ttl_env) if ttl_env and ttl_env.isdigit() else None
                    cls._instance = cls(path, ttl=ttl, disabled=disabled)
        return cls._instance

    # ---- DB ----

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS llm_cache (
                    key TEXT PRIMARY KEY,
                    created_at INTEGER NOT NULL,
                    expires_at INTEGER,
                    request_json TEXT NOT NULL,
                    response_json TEXT NOT NULL
                )"""
            )
        return self._conn

    # ---- key ----

    @staticmethod
    def build_key(model: str, temperature: float, messages: Any, meta: Optional[Dict[str, Any]] = None) -> str:
        payload: Dict[str, Any] = {"model": model, "temperature": temperature, "messages": messages}
        if meta:
            payload["meta"] = meta
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    # ---- get / set ----

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        if self.disabled:
            return None
        conn = self._connect()
        row = conn.execute(
            "SELECT response_json, expires_at FROM llm_cache WHERE key=?", (key,)
        ).fetchone()
        if not row:
            return None
        response_json, expires_at = row
        if expires_at is not None and expires_at < int(time.time()):
            conn.execute("DELETE FROM llm_cache WHERE key=?", (key,))
            return None
        try:
            return json.loads(response_json)
        except json.JSONDecodeError:
            return None

    def set(
        self,
        key: str,
        request_payload: Dict[str, Any],
        response_payload: Dict[str, Any],
    ) -> None:
        conn = self._connect()
        now = int(time.time())
        expires = (now + self.ttl) if self.ttl else None
        with self._write_lock:
            conn.execute(
                "REPLACE INTO llm_cache"
                "(key, created_at, expires_at, request_json, response_json) "
                "VALUES (?,?,?,?,?)",
                (
                    key,
                    now,
                    expires,
                    json.dumps(request_payload, sort_keys=True, separators=(",", ":")),
                    json.dumps(response_payload, sort_keys=True, separators=(",", ":")),
                ),
            )

    def clear(self) -> int:
        conn = self._connect()
        with self._write_lock:
            cur = conn.execute("DELETE FROM llm_cache")
        return cur.rowcount or 0


# ---------------------------------------------------------------------------
# Cached completion helper
# ---------------------------------------------------------------------------

def cached_completion(
    call_fn: Callable[[], Dict[str, Any]],
    *,
    model: str,
    temperature: float,
    messages: Any,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Invoke *call_fn* with transparent SQLite caching."""
    cache = LLMCache.get_default()
    key = cache.build_key(model, temperature, messages, meta)
    cached = cache.get(key)
    if cached is not None:
        return {**cached, "_cached": True}
    result = call_fn()
    try:
        cache.set(
            key,
            {"model": model, "temperature": temperature, "messages": messages, "meta": meta},
            result,
        )
    except Exception:
        logger.debug("Failed to cache LLM response", exc_info=True)
    return {**result, "_cached": False}


# ---------------------------------------------------------------------------
# Semantic equivalence via LLM
# ---------------------------------------------------------------------------

_EQUIV_SYSTEM_PROMPT = """\
You are an expert at analyzing software engineering tasks. 
Your job is to determine if two states in a coding agent's execution represent the same semantic step.

Consider:
1. The tool/action being performed
2. The target (file, query, command)
3. The intent (what the agent is trying to accomplish)
4. The outcome/observation

Two states are EQUIVALENT if:
- They perform the same type of action
- On the same target (or semantically equivalent targets)
- With the same intent
- Leading to the same type of outcome

Two states are NOT EQUIVALENT if:
- They perform different actions
- On different targets
- With different intents
- Leading to different outcomes

Respond with EXACTLY a JSON object with two keys:
- "equivalent": boolean (true or false)
- "reasoning": string (1-2 sentences explaining why)\
"""


def _state_to_summary(state: "State") -> str:
    """Convert state to a summary string for LLM."""
    lines: List[str] = []
    if hasattr(state, "tool_used") and state.tool_used:
        lines.append(f"Tool: {state.tool_used}")
    if hasattr(state, "log_entry") and state.log_entry:
        entry = state.log_entry
        if hasattr(entry, "args") and entry.args:
            key_args: Dict[str, Any] = {}
            for key in ("filePath", "path", "query", "command"):
                if key in entry.args:
                    val = str(entry.args[key])
                    if len(val) > 100:
                        val = val[:100] + "..."
                    key_args[key] = val
            if key_args:
                lines.append(f"Arguments: {json.dumps(key_args)}")
        if hasattr(entry, "response") and entry.response:
            resp = str(entry.response)
            if len(resp) > 200:
                resp = resp[:200] + "..."
            lines.append(f"Response: {resp}")
    if hasattr(state, "observation") and state.observation:
        obs = state.observation
        if len(obs) > 300:
            obs = obs[:300] + "..."
        lines.append(f"Observation: {obs}")
    if hasattr(state, "files_touched") and state.files_touched:
        lines.append(f"Files touched: {list(state.files_touched)[:5]}")
    return "\n".join(lines) if lines else "(no details available)"


def llm_equivalence_check(
    state_a: "State",
    state_b: "State",
) -> Optional["EquivalenceResult"]:
    """Compare two states using an LLM, returning an EquivalenceResult.

    Returns *None* if the LLM is unavailable or the response is unparseable.
    """
    from .equivalence import EquivalenceResult

    try:
        client, model, temp = get_model_and_client()
    except Exception as exc:
        logger.warning("LLM client unavailable: %s", exc)
        return None

    # Build prompt matching prototype's _build_equivalence_prompt + _state_to_summary
    user_msg = (
        "Compare these two states from a coding agent execution:\n"
        "\n"
        "=== STATE 1 ===\n"
        f"{_state_to_summary(state_a)}\n"
        "\n"
        "=== STATE 2 ===\n"
        f"{_state_to_summary(state_b)}\n"
        "\n"
        "Are these states semantically equivalent?\n"
        'Respond with JSON: {"equivalent": true/false, "reasoning": "..."}'
    )

    messages = [
        {"role": "system", "content": _EQUIV_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    def _call() -> Dict[str, Any]:
        resp = _safe_llm_call(client, model, messages, max_tokens=500, temperature=0.1)
        if not resp.choices:
            return {"content": None, "finish_reason": "no_choices"}
        choice = resp.choices[0]
        return {"content": choice.message.content, "finish_reason": choice.finish_reason}

    try:
        result = cached_completion(_call, model=model, temperature=0.1, messages=messages,
                                   meta={"purpose": "llm_equivalence"})
    except Exception as exc:
        logger.error("LLM equivalence call failed: %s", exc)
        return None

    content = result.get("content")
    if not content:
        fr = result.get("finish_reason", "unknown")
        logger.warning("LLM returned empty content. Finish reason: %s", fr)
        return None
    content = content.strip()
    if not content:
        return None

    try:
        parsed = json.loads(content)
        return EquivalenceResult(
            equivalent=bool(parsed.get("equivalent", False)),
            confidence=0.85,  # hardcoded like prototype
            reasoning=str(parsed.get("reasoning", "")),
            method="llm",
        )
    except (json.JSONDecodeError, KeyError, TypeError):
        # Fallback: balanced-brace extraction
        start = content.find("{")
        if start != -1:
            brace = 0
            for i, ch in enumerate(content[start:], start):
                if ch == "{":
                    brace += 1
                elif ch == "}":
                    brace -= 1
                    if brace == 0:
                        try:
                            parsed = json.loads(content[start : i + 1])
                            return EquivalenceResult(
                                equivalent=bool(parsed.get("equivalent", False)),
                                confidence=0.85,
                                reasoning=str(parsed.get("reasoning", "")),
                                method="llm",
                            )
                        except json.JSONDecodeError:
                            pass
                        break
        # Regex fallback
        import re
        equiv_m = re.search(r'"equivalent"\s*:\s*(true|false)', content, re.IGNORECASE)
        reason_m = re.search(r'"reasoning"\s*:\s*"([^"]*)"', content)
        if equiv_m and reason_m:
            return EquivalenceResult(
                equivalent=equiv_m.group(1).lower() == "true",
                confidence=0.85,
                reasoning=reason_m.group(1),
                method="llm",
            )
        logger.warning("Failed to parse LLM equivalence response: %s", content[:200])
        return None


# ---------------------------------------------------------------------------
# Semantic content comparison via LLM
# ---------------------------------------------------------------------------

_SEMANTIC_CONTENT_SYSTEM_PROMPT = """\
You are an expert at analyzing coding agent trajectories in software development.
Your job is to determine if two code changes represent semantically equivalent \
STEPS in completing a task.

CONTEXT: We are comparing trajectories of AI coding agents solving the same task. 
We want to know if two changes represent the SAME STEP in the solution process,
NOT whether the exact code produced is identical.

Two changes are SEMANTICALLY EQUIVALENT (as steps) if:
- They accomplish the same logical step in solving the task
- They modify/create the same type of artifact for the same purpose
- The core functionality being implemented is the same
- Differences are implementation details (naming, exact values, formatting)

Two changes are NOT SEMANTICALLY EQUIVALENT if:
- They create/modify fundamentally different things
- They serve different purposes in solving the task
- One implements a feature the other doesn't touch at all

Examples:
- "Defined function 'add_numbers' that adds two values" vs "Defined function 'perform_addition' that adds two values"
  \u2192 EQUIVALENT (same step: creating an addition function, naming is implementation detail)
  
- "Created file 'math_ops.py' with arithmetic functions" vs "Created file 'calculator.py' with arithmetic functions"
  \u2192 EQUIVALENT (same step: creating a file for arithmetic operations)
  
- "Added error handling to 'calculate' function" vs "Created new 'calculate' function"
  \u2192 NOT EQUIVALENT (different types of steps)

- "Modified config to set port=8080" vs "Modified config to set debug=True"
  \u2192 NOT EQUIVALENT (different configuration changes)

Respond with EXACTLY a JSON object:
{"equivalent": true/false, "confidence": 0.0-1.0, "reasoning": "brief explanation"}\
"""


def llm_semantic_content_check(prompt: str) -> Optional[Dict[str, Any]]:
    """Compare two content descriptions using an LLM.

    Returns a dict with ``equivalent``, ``confidence``, ``reasoning``
    keys, or *None* if the LLM is unavailable or the response is
    unparseable.
    """
    try:
        client, model, temp = get_model_and_client()
    except Exception as exc:
        logger.warning("LLM client unavailable for semantic content: %s", exc)
        return None

    messages = [
        {"role": "system", "content": _SEMANTIC_CONTENT_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    def _call() -> Dict[str, Any]:
        resp = _safe_llm_call(client, model, messages, max_tokens=500, temperature=0.1)
        content = resp.choices[0].message.content.strip() if resp.choices else ""
        return {"content": content}

    try:
        result = cached_completion(_call, model=model, temperature=0.1, messages=messages,
                                   meta={"purpose": "semantic_content"})
    except Exception as exc:
        logger.error("LLM semantic content call failed: %s", exc)
        return None
    content = result.get("content", "")

    try:
        # Strip markdown code fences if present
        text = content.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        parsed = json.loads(text)
        return {
            "equivalent": bool(parsed.get("equivalent", False)),
            "confidence": float(parsed.get("confidence", 0.8)),
            "reasoning": str(parsed.get("reasoning", "LLM decision")),
        }
    except (json.JSONDecodeError, KeyError, TypeError):
        import re
        equiv_m = re.search(r'"equivalent"\s*:\s*(true|false)', content, re.IGNORECASE)
        if equiv_m:
            conf_m = re.search(r'"confidence"\s*:\s*([0-9.]+)', content)
            reason_m = re.search(r'"reasoning"\s*:\s*"([^"]*)"', content)
            return {
                "equivalent": equiv_m.group(1).lower() == "true",
                "confidence": float(conf_m.group(1)) if conf_m else 0.8,
                "reasoning": reason_m.group(1) if reason_m else "LLM decision",
            }
        logger.warning("Failed to parse semantic content LLM response: %s", content[:200])
        return None
