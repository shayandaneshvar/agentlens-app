"""Lightweight SQLite cache for LLM chat completions.

Environment variables:
  LLM_CACHE_PATH            Path to sqlite DB (default: ./llm_cache.db)
  LLM_CACHE_DISABLE         If set to a truthy value, disables reading from cache but still writes.
  LLM_CACHE_TTL_SECONDS     If set, entries older than now-ttl are invalid.
  LLM_CACHE_VERBOSE         If truthy, prints cache hit/miss info to stdout.

Public helpers:
  get_cache() -> singleton LLMCache
  cached_chat_completion(call_fn, *, model, temperature, messages, meta=None) -> response

The cache key is SHA256 over a canonical JSON containing:
  {
    "model": model,
    "temperature": temperature,
    "messages": messages,
    "meta": meta (optional, provider/prefix etc.)
  }

Table schema (created on first use):
  CREATE TABLE IF NOT EXISTS llm_cache (
      key TEXT PRIMARY KEY,
      created_at INTEGER NOT NULL,
      expires_at INTEGER,
      request_json TEXT NOT NULL,
      response_json TEXT NOT NULL
  );

Safe for concurrent reads; writes protected via a simple threading.Lock.
Not optimized for high throughput but sufficient for local dev & scripts.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import hashlib
from pathlib import Path
from typing import Any, Callable, Dict, Optional

_CACHE_SINGLETON = None
_LOCK = threading.Lock()


def _truthy(val: Optional[str]) -> bool:
    if not val:
        return False
    return val.lower() in {"1", "true", "yes", "on"}


class LLMCache:
    def __init__(self, path: str, ttl: Optional[int] = None, verbose: bool = False, disabled: bool = False):
        self.path = path
        self.ttl = ttl
        self.verbose = verbose
        self.disabled = disabled
        self._conn = None  # lazy
        self._write_lock = threading.Lock()

    # --------------- internal helpers -----------------
    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)  # autocommit
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS llm_cache (
                key TEXT PRIMARY KEY,
                created_at INTEGER NOT NULL,
                expires_at INTEGER,
                request_json TEXT NOT NULL,
                response_json TEXT NOT NULL
            )"""
            )
            # Basic index on expires_at for pruning (optional)
            try:
                self._conn.execute("CREATE INDEX IF NOT EXISTS idx_llm_cache_expires ON llm_cache(expires_at)")
            except sqlite3.OperationalError:
                pass
        return self._conn

    def _log(self, msg: str):
        if self.verbose:
            print(f"[LLMCache] {msg}")

    # --------------- keying -----------------
    @staticmethod
    def build_key(model: str, temperature: float, messages: Any, meta: Optional[Dict[str, Any]] = None) -> str:
        payload = {
            "model": model,
            "temperature": temperature,
            "messages": messages,
        }
        if meta:
            payload["meta"] = meta
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    # --------------- public API -----------------
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        if self.disabled:
            self._log(f"DISABLED - forcing MISS {key[:8]}")
            return None
        conn = self._connect()
        now = int(time.time())
        row = conn.execute("SELECT response_json, expires_at FROM llm_cache WHERE key=?", (key,)).fetchone()
        if not row:
            self._log(f"MISS {key[:8]}")
            return None
        response_json, expires_at = row
        if expires_at is not None and expires_at < now:
            # expired -> delete async (lazy prune)
            try:
                conn.execute("DELETE FROM llm_cache WHERE key=?", (key,))
            except Exception:
                pass
            self._log(f"EXPIRED {key[:8]}")
            return None
        try:
            data = json.loads(response_json)
        except json.JSONDecodeError:
            self._log(f"CORRUPT {key[:8]}")
            return None
        self._log(f"HIT {key[:8]}")
        return data

    def set(self, key: str, request_payload: Dict[str, Any], response_payload: Dict[str, Any]):
        # When disabled, we still write to cache to overwrite bad entries
        conn = self._connect()
        created = int(time.time())
        expires_at = None
        if self.ttl:
            expires_at = created + int(self.ttl)
        with self._write_lock:
            conn.execute(
                "REPLACE INTO llm_cache(key, created_at, expires_at, request_json, response_json) VALUES (?,?,?,?,?)",
                (
                    key,
                    created,
                    expires_at,
                    json.dumps(request_payload, sort_keys=True, separators=(",", ":")),
                    json.dumps(response_payload, sort_keys=True, separators=(",", ":")),
                ),
            )
        store_msg = f"STORE {key[:8]}"
        if self.disabled:
            store_msg += " (disabled mode - overwriting)"
        self._log(store_msg)

    def clear(self):
        if self.disabled:
            return 0
        conn = self._connect()
        with self._write_lock:
            cur = conn.execute("DELETE FROM llm_cache")
        n = cur.rowcount if cur.rowcount is not None else 0
        self._log(f"CLEARED {n} rows")
        return n

    def prune_expired(self):
        if self.disabled:
            return 0
        conn = self._connect()
        now = int(time.time())
        with self._write_lock:
            cur = conn.execute("DELETE FROM llm_cache WHERE expires_at IS NOT NULL AND expires_at < ?", (now,))
        n = cur.rowcount if cur.rowcount is not None else 0
        if n:
            self._log(f"PRUNED {n} expired")
        return n


def get_cache() -> LLMCache:
    global _CACHE_SINGLETON
    if _CACHE_SINGLETON is None:
        with _LOCK:
            if _CACHE_SINGLETON is None:
                path = os.getenv("LLM_CACHE_PATH", "llm_cache.db")
                disabled = _truthy(os.getenv("LLM_CACHE_DISABLE"))
                ttl_env = os.getenv("LLM_CACHE_TTL_SECONDS")
                ttl = int(ttl_env) if ttl_env and ttl_env.isdigit() else None
                verbose = _truthy(os.getenv("LLM_CACHE_VERBOSE"))
                _CACHE_SINGLETON = LLMCache(path=path, ttl=ttl, verbose=verbose, disabled=disabled)
    return _CACHE_SINGLETON  # type: ignore[return-value]


def cached_chat_completion(
    call_fn: Callable[[], Dict[str, Any]],
    *,
    model: str,
    temperature: float,
    messages: Any,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run call_fn (which performs the actual API call) with caching.

    The call_fn should return a dict-like response (already simplified / serialized).
    We store & return that object.
    """
    cache = get_cache()
    key = cache.build_key(model, temperature, messages, meta)
    cached = cache.get(key)
    if cached is not None:
        return {"cached": True, **cached}
    # Cache miss: perform call
    result = call_fn()
    # Store only on success-like shape (heuristic: success True if provided)
    try:
        cache.set(key, {"model": model, "temperature": temperature, "messages": messages, "meta": meta}, result)
    except Exception as e:  # pragma: no cover - best effort
        if cache.verbose:
            print(f"[LLMCache] WARNING failed to store: {e}")
    return {"cached": False, **result}
