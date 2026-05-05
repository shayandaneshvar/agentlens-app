"""Utility helpers to obtain LLM provider clients & resolve model configuration.

Any module can import from here to quickly get a configured OpenAI / Azure OpenAI
style client plus (provider, model, temperature) derived from environment
variables.

Environment variable conventions:
  <PREFIX>_LLM or DEFAULT_LLM -> "provider:model[:temperature]" OR a semicolon
  separated list of those entries for ensemble style usage.

Supported providers (strings are case-sensitive here):
  trapi, ollama, openai, azure, anthropic, codex

Examples:
  export DEFAULT_LLM="openai:gpt-4o:0.7"
  export SPEC_CHECKER_LLM="azure:my-deployment:0.3"
  export PIPELINE_LLM="openai:gpt-4o-mini:0.2;ollama:llama3:1.0"  # list form

NOTE: This module performs a lightweight dotenv load at import time so a
project-local .env is picked up automatically (safe if already loaded).
"""

from __future__ import annotations

import os
from typing import List, Tuple, Union

from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI
from azure.identity import (
    DefaultAzureCredential,
    ChainedTokenCredential,
    AzureCliCredential,
    get_bearer_token_provider,
)

# Load environment variables from a .env file if present (idempotent)
load_dotenv()

__all__ = [
    "get_client",
    "get_provider_and_model_from_env",
    "get_model_and_client",
    "evaluate_image_diff_with_llm",
    "build_multimodal_user_message",
    "prepare_semantic_diff_prompt",
    "run_multimodal_single",
]

# Optional lightweight response caching (see llm_cache.py). If the module is
# absent or errors, we silently proceed without caching.
try:  # pragma: no cover - optional
    from .llm_cache import cached_chat_completion
except Exception:  # noqa: pragma: no cover
    cached_chat_completion = None  # type: ignore


_CLIENT_CACHE = {}


# ---------------------------------------------------------------------------
# Anthropic -> OpenAI adapter
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
    """Wraps an AnthropicFoundry client to expose an OpenAI-compatible interface.

    Usage:
        adapter = _AnthropicOpenAIAdapter(anthropic_foundry_client)
        resp = adapter.chat.completions.create(model=..., messages=...)
        print(resp.choices[0].message.content)
    """

    def __init__(self, anthropic_client):
        self._client = anthropic_client
        self.chat = _AnthropicChatProxy(anthropic_client)


def _get_openai_retry_config():
    """Return kwargs for client retry configuration if supported by SDK.

    Defaults to max_retries=1. Override with CLAUDE_PROXY_BACKEND_RETRIES or OPENAI_MAX_RETRIES.
    Provider gateways (e.g., TRAPI) may still perform internal retries beyond SDK control.
    """
    retries = int(
        os.getenv(
            "CLAUDE_PROXY_BACKEND_RETRIES",
            os.getenv("OPENAI_MAX_RETRIES", "1"),
        )
    )
    if retries < 1:
        retries = 1
    return {"max_retries": retries}


def get_client(provider: str):
    """Return a cached client instance for the given provider.

    Parameters
    ----------
    provider : str
        One of: trapi, ollama, openai, azure, anthropic, codex
    """
    if provider in _CLIENT_CACHE:
        return _CLIENT_CACHE[provider]

    provider = provider.strip()

    if provider == "trapi":
        scope = os.getenv("TRAPI_SCOPE", "api://trapi/.default")
        credential = get_bearer_token_provider(
            ChainedTokenCredential(
                AzureCliCredential(),
                DefaultAzureCredential(
                    exclude_cli_credential=True,
                    # Exclude other credentials we are not interested in.
                    exclude_environment_credential=True,
                    exclude_shared_token_cache_credential=True,
                    exclude_developer_cli_credential=True,
                    exclude_powershell_credential=True,
                    exclude_interactive_browser_credential=True,
                    exclude_visual_studio_code_credentials=True,
                    managed_identity_client_id=os.environ.get("DEFAULT_IDENTITY_CLIENT_ID"),
                ),
            ),
            scope,
        )

        api_version = os.getenv("TRAPI_OPENAI_API_VERSION", "2024-10-21")
        instance = os.getenv("TRAPI_INSTANCE", "redmond/interactive")
        endpoint = os.getenv("TRAPI_ENDPOINT", f"https://trapi.research.example.com/{instance}")

        client = AzureOpenAI(
            azure_endpoint=endpoint,
            azure_ad_token_provider=credential,
            api_version=api_version,
            **_get_openai_retry_config(),
        )
        _CLIENT_CACHE[provider] = client
        return client

    if provider == "ollama":
        base_url = os.getenv("OLLAMA_URL")
        if not base_url:
            raise ValueError("OLLAMA_URL is not set")
        client = OpenAI(
            base_url=base_url,
            api_key="ollama",  # placeholder; Ollama ignores key
        )
        _CLIENT_CACHE[provider] = client
        return client

    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is not set")
        if not os.getenv("OPENAI_BASE_URL"):
            raise ValueError("OPENAI_BASE_URL is not set")
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            **_get_openai_retry_config(),
        )
        _CLIENT_CACHE[provider] = client
        return client

    if provider == "azure":
        if not os.getenv("AZURE_OPENAI_API_KEY"):
            raise ValueError("AZURE_OPENAI_API_KEY is not set")
        if not os.getenv("AZURE_OPENAI_API_VERSION"):
            raise ValueError("AZURE_OPENAI_API_VERSION is not set")
        if not os.getenv("AZURE_OPENAI_ENDPOINT"):
            raise ValueError("AZURE_OPENAI_ENDPOINT is not set")
        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            **_get_openai_retry_config(),
        )
        _CLIENT_CACHE[provider] = client
        return client

    if provider == "anthropic":
        if not os.getenv("AZURE_OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
            raise ValueError(
                "Neither ANTHROPIC_API_KEY nor AZURE_OPENAI_API_KEY is set. "
                "Set one of them for the 'anthropic' provider."
            )
        if not os.getenv("AZURE_OPENAI_ENDPOINT") and not os.getenv("ANTHROPIC_BASE_URL"):
            raise ValueError(
                "Neither ANTHROPIC_BASE_URL nor AZURE_OPENAI_ENDPOINT is set. "
                "Set one of them for the 'anthropic' provider."
            )
        api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
        base_url = os.getenv("ANTHROPIC_BASE_URL") or os.getenv("AZURE_OPENAI_ENDPOINT")
        try:
            from anthropic import AnthropicFoundry  # type: ignore
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError(
                "anthropic package not installed. Run: pip install anthropic"
            ) from e
        raw_client = AnthropicFoundry(
            api_key=api_key,
            base_url=base_url,
        )
        client = _AnthropicOpenAIAdapter(raw_client)
        _CLIENT_CACHE[provider] = client
        return client

    if provider == "codex":
        if not os.getenv("CODEX_API_KEY"):
            raise ValueError("CODEX_API_KEY is not set")
        if not os.getenv("CODEX_BASE_URL"):
            raise ValueError("CODEX_BASE_URL is not set")
        try:
            from codex_client import CodexClient  # type: ignore
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError(
                "codex_client package not installed. Install it to use provider 'codex'."
            ) from e
        client = CodexClient(
            api_key=os.getenv("CODEX_API_KEY"),
            base_url=os.getenv("CODEX_BASE_URL"),
            api_version=os.getenv("CODEX_API_VERSION", "2025-04-01-preview"),
        )
        _CLIENT_CACHE[provider] = client
        return client

    raise ValueError(f"Invalid provider: {provider}")


def get_provider_and_model_from_env(
    prefix: str,
) -> Union[Tuple[str, str, float], Tuple[List[str], List[str], List[float]]]:
    """Resolve provider/model(/temperature) triple(s) from environment.

    Order of lookup: <PREFIX>_LLM then DEFAULT_LLM.
    Accepts single: provider:model[:temp]
    Or list: provider:model[:temp];provider2:model2[:temp];...
    Temperature defaults to 1.0 if omitted.
    Returns either a single (provider, model, temperature) OR three lists.
    """
    env_var = f"{prefix.upper()}_LLM"
    value = os.getenv(env_var) or os.getenv("DEFAULT_LLM")
    if not value:
        raise ValueError(f"Neither {env_var} nor DEFAULT_LLM is set in environment.")

    # Multi-entry form
    if ";" in value:
        providers: List[str] = []
        models: List[str] = []
        temperatures: List[float] = []
        for pair in value.split(";"):
            parts = pair.split(":")
            if len(parts) < 2:
                raise ValueError(f"Invalid provider:model pair: {pair}")
            provider = parts[0].strip()
            temperature = 1.0
            model_parts = parts[1:]
            if len(parts) > 2:
                try:
                    temperature = float(parts[-1].strip())
                    model_parts = parts[1:-1]
                except ValueError:
                    model_parts = parts[1:]
            model = ":".join(model_parts).strip()
            providers.append(provider)
            models.append(model)
            temperatures.append(temperature)
        return providers, models, temperatures

    # Single entry
    parts = value.split(":")
    if len(parts) < 2:
        raise ValueError(
            f"Invalid provider:model format in {env_var} or DEFAULT_LLM: {value}"
        )
    provider = parts[0].strip()
    temperature = 1.0
    model_parts = parts[1:]
    if len(parts) > 2:
        try:
            temperature = float(parts[-1].strip())
            model_parts = parts[1:-1]
        except ValueError:
            model_parts = parts[1:]
    model = ":".join(model_parts).strip()
    return provider, model, temperature


def get_model_and_client(prefix: str):
    """Convenience helper returning (client, model, temperature).

    Example:
        client, model, temp = get_model_and_client("spec_checker")
    """
    result = get_provider_and_model_from_env(prefix)
    if isinstance(result[0], list):  # type: ignore[index]
        providers, models, temps = result  # type: ignore[assignment]
        clients = [get_client(p) for p in providers]
        return clients, models, temps
    provider, model, temp = result  # type: ignore[misc]
    client = get_client(provider)
    return client, model, temp

# ------------------------------------------------------------
# Image + LLM Diff Support (stub integration layer)
# ------------------------------------------------------------
import base64
from pathlib import Path
from typing import Optional, Dict, Any

def _encode_image(path: str) -> str:
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def evaluate_image_diff_with_llm(
    img_a: str,
    img_b: str,
    metrics: Dict[str, Any],
    prefix: str = "DEFAULT",
    threshold: float = 0.5,
    model_hint: Optional[str] = None,
) -> Dict[str, Any]:
    """Heuristic significance classification of two images using diff metrics.

    Currently does NOT invoke an LLM; returns a composite heuristic score and
    a boolean 'substantial_change'. This is a placeholder hook where a future
    multimodal call could refine the decision.
    """
    phash = metrics.get('phash_distance', 64)
    ssim = metrics.get('ssim', 0.0)
    pixel_ratio = metrics.get('pixel_change_ratio', 1.0)

    phash_norm = min(phash / 64.0, 1.0)
    ssim_inv = 1.0 - max(min(ssim, 1.0), 0.0)
    composite = (0.4 * phash_norm) + (0.4 * pixel_ratio) + (0.2 * ssim_inv)

    substantial = composite >= threshold
    reason = (
        f"Heuristic composite={composite:.3f} (>= {threshold}) => substantial change"
        if substantial
        else f"Heuristic composite={composite:.3f} (< {threshold}) => minor change"
    )

    return {
        'substantial_change': substantial,
        'reason': reason,
        'metrics': metrics,
        'composite_score': composite,
        'image_a': Path(img_a).name,
        'image_b': Path(img_b).name,
    }

# ---------------------------------------------------------------------------
# Multimodal helper: build messages with image per user provided pattern
# ---------------------------------------------------------------------------
def build_multimodal_user_message(text: str, image_path: str) -> Dict[str, Any]:
    """Return a message dict suitable for OpenAI / Azure-style multimodal chat."""
    import base64
    import mimetypes
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"Image for multimodal message not found: {image_path}")
    mime, _ = mimetypes.guess_type(str(p))
    if not mime:
        mime = "image/png"
    b64 = base64.b64encode(p.read_bytes()).decode('utf-8')
    data_url = f"data:{mime};base64,{b64}"
    content = [
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]
    return {"role": "user", "content": content}


def prepare_semantic_diff_prompt(diff_record: Dict[str, Any]) -> str:
    """Create instruction prompt describing the SIDE-BY-SIDE composite (LEFT=A, RIGHT=B).

    We no longer send a synthetic heat / amplified diff. The model sees an unmodified
    composite image preserving visual fidelity. Provide clear decision rules and
    enforce strict JSON output (reasoning + result boolean)."""
    lines: list[str] = []
    lines.append("You are given a single composite image containing TWO UI screenshots placed side-by-side: LEFT = Screenshot A, RIGHT = Screenshot B (no guaranteed temporal order). They may be the same UI state captured twice or two different states.")
    lines.append("Goal: Determine STATE EQUIVALENCE, i.e. whether A and B represent the SAME underlying functional UI state (equivalent) or DIFFERENT functional UI states (non-equivalent).")
    lines.append("Mapping to result: if they are functionally the SAME state (equivalent) => result=false; if they are DIFFERENT states (non\u2011equivalent) => result=true. This aligns with downstream merge logic that treats result=false as safe to merge.")
    lines.append("CRITICAL: Ignore OS/desktop level cosmetic differences: wallpaper, window chrome, theme (light/dark), translucency, shadows, clock/time, system tray, battery/network indicators, desktop notifications unrelated to the app, window movement/position, minor anti-aliasing, cursor blink, caret blink, subtle color shifts with no functional meaning.")
    lines.append("Focus ONLY on application/task-relevant semantics: new/removed dialogs, modals, panels, menus, toolbars, navigation items, buttons, form fields, list/table/grid content changes, enabled/disabled state, selection meaning change, validation/status messages, progress indicators, data population (e.g., search results appearing), structural layout region repurposing.")
    lines.append("Other domain specific invariants to ignore: Ignore the exact commit message generated, if two commit message talks about the same change in different wording consider them equivalent.")
    lines.append("If ONLY ignored cosmetic changes are present, classify as equivalent => result=false.")
    lines.append("")
    lines.append("Metrics (for context only \u2013 do not rely blindly):")
    lines.append(f"  phash_distance={diff_record.get('phash_distance')}  ssim={diff_record.get('ssim')}  pixel_change_ratio={diff_record.get('pixel_change_ratio')}")
    lines.append("")
    lines.append("Interpretation examples:")
    lines.append("  Non-equivalent (result=true): a dialog appears/disappears; a new pane or modal; a list/table gains or loses meaningful rows; a button becomes enabled/disabled; status indicator color meaningfully changes (success/error/state); new validation or error text; navigation/tab selection changes content; previously empty data region now populated (or vice versa).")
    lines.append("  Equivalent (result=false): only wallpaper/theme/window chrome changes; window moved/resized with identical internal widgets; caret/cursor blink; subtle font rendering/anti-aliasing shift; highlight/focus ring position change without altering available actions or data; clock/time differences; minor spacing/padding tweaks; purely decorative color shade shift not altering meaning.")
    lines.append("")
    lines.append("OUTPUT REQUIREMENTS:")
    lines.append("Return EXACTLY one JSON object with two keys in this order: 'reasoning', 'result'.")
    lines.append(" - reasoning: SHORT justification focusing on 1-2 concrete UI element changes OR stating 'cosmetic only'.")
    lines.append(" - result: boolean true or false (unquoted).")
    lines.append("NO extra keys, no markdown, no surrounding text.")
    lines.append("If unsure or ambiguous, default to result=false (treat as equivalent).")
    return "\n".join(lines)


def run_multimodal_single(user_message: Dict[str, Any], prefix: str = "DEFAULT") -> Dict[str, Any]:
    """Execute a single-turn multimodal chat if provider supports images.

    Integrates optional caching (llm_cache.cached_chat_completion) keyed on
    (model, temperature, message content, prefix). Caching can be disabled via
    environment variables (see llm_cache.py) or if llm_cache import fails.
    """
    try:
        client, model, temp = get_model_and_client(prefix)
    except Exception as e:  # noqa
        return {"success": False, "error": f"Client init failed: {e}"}

    messages = [user_message]

    def _invoke():
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temp,
                messages=messages,
            )
            content = resp.choices[0].message.content if resp.choices else None
            raw_repr = getattr(resp, 'model_dump', lambda: str(resp))()
            return {"success": True, "response": content, "raw": raw_repr}
        except Exception as e:  # noqa
            return {"success": False, "error": f"Invocation failed: {e}"}

    if cached_chat_completion is None:
        return _invoke()

    # Use optional meta to distinguish by prefix.
    result = cached_chat_completion(
        _invoke,
        model=model,
        temperature=temp,
        messages=messages,
        meta={"prefix": prefix, "multimodal": True},
    )
    return result
