#!/usr/bin/env python3
"""
llm_providers.py — One text-completion interface across Anthropic, OpenAI,
Google, and DeepSeek, so the baseline can be run on any of them unchanged.

Provider is inferred from the model ID, or stated explicitly with a
`provider:model` prefix when a name is ambiguous:

    claude-opus-5              -> anthropic
    gpt-5.6-terra, o3          -> openai
    gemini-3.5-flash           -> google
    deepseek-v4-pro            -> deepseek
    openai:some-custom-model   -> openai (explicit)

API keys come from the environment or the project .env file:
    ANTHROPIC_API_KEY, OPENAI_API_KEY,
    GOOGLE_API_KEY (or GEMINI_API_KEY), DEEPSEEK_API_KEY

DeepSeek speaks the OpenAI wire protocol, so it reuses the `openai` SDK
against its own base URL.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROVIDERS = ("anthropic", "openai", "google", "deepseek")

# Env vars checked per provider, in order.
_KEY_ENV = {
    "anthropic": ["ANTHROPIC_API_KEY"],
    "openai": ["OPENAI_API_KEY"],
    "google": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
    "deepseek": ["DEEPSEEK_API_KEY"],
}

_PIP_PACKAGE = {
    "anthropic": "anthropic",
    "openai": "openai",
    "google": "google-genai",
    "deepseek": "openai",  # OpenAI-compatible API
}

DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# OpenAI reasoning families reject `temperature` and require
# `max_completion_tokens` in place of `max_tokens`.
_OPENAI_REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")


# Above this many output tokens, a non-streaming request risks an HTTP
# timeout, so the Anthropic path streams instead.
STREAM_ABOVE_MAX_TOKENS = 16_000

# Per-provider stop reasons that mean "output was cut off at the token cap".
_TRUNCATION_REASONS = {"max_tokens", "length", "MAX_TOKENS"}


@dataclass
class LLMResponse:
    """Provider-agnostic completion result."""

    text: str
    usage: Dict[str, Optional[int]] = field(default_factory=dict)
    model: str = ""
    provider: str = ""
    stop_reason: Optional[str] = None

    @property
    def truncated(self) -> bool:
        """True when the model hit the token cap before finishing.

        Worth checking separately from a parse failure: on models that think
        by default (Claude Sonnet 5, Opus 5, Fable 5) `max_tokens` covers
        thinking *and* visible text, so a budget that is too small returns a
        successful call with empty or half-written output.
        """
        return str(self.stop_reason) in _TRUNCATION_REASONS


def load_env_key(name: str) -> Optional[str]:
    """Read a key from the environment, falling back to the project .env file."""
    value = os.environ.get(name)
    if value:
        return value
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return None
    try:
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith(f"{name}="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    except Exception:
        pass
    return None


def resolve_provider(model: str) -> tuple:
    """Return (provider, model_id) for a model string.

    An explicit `provider:model` prefix wins; otherwise the provider is
    inferred from the model name.
    """
    if ":" in model:
        prefix, rest = model.split(":", 1)
        if prefix in PROVIDERS:
            return prefix, rest

    lowered = model.lower()
    if lowered.startswith("claude"):
        return "anthropic", model
    if lowered.startswith("gemini"):
        return "google", model
    if lowered.startswith("deepseek"):
        return "deepseek", model
    if lowered.startswith(("gpt-", "gpt", "o1", "o3", "o4", "chatgpt")):
        return "openai", model

    raise ValueError(
        f"Cannot infer a provider for model '{model}'. "
        f"Prefix it explicitly, e.g. 'openai:{model}' "
        f"(providers: {', '.join(PROVIDERS)})."
    )


def api_key_for(provider: str, api_key: Optional[str] = None) -> Optional[str]:
    if api_key:
        return api_key
    for name in _KEY_ENV.get(provider, []):
        value = load_env_key(name)
        if value:
            return value
    return None


def missing_key_message(provider: str) -> str:
    names = " or ".join(_KEY_ENV.get(provider, []))
    return (
        f"No API key for provider '{provider}'. Set {names} in the environment "
        f"or add it to the project .env file."
    )


_CLIENT_CACHE: Dict[str, Any] = {}


def get_client(provider: str, api_key: Optional[str] = None) -> Any:
    """Build (and memoize) a client for a provider. Raises on missing key/SDK."""
    if provider in _CLIENT_CACHE and api_key is None:
        return _CLIENT_CACHE[provider]

    key = api_key_for(provider, api_key)
    if not key:
        raise RuntimeError(missing_key_message(provider))

    try:
        if provider == "anthropic":
            import anthropic

            client = anthropic.Anthropic(api_key=key)
        elif provider == "openai":
            from openai import OpenAI

            client = OpenAI(api_key=key)
        elif provider == "deepseek":
            from openai import OpenAI

            client = OpenAI(api_key=key, base_url=DEEPSEEK_BASE_URL)
        elif provider == "google":
            from google import genai

            client = genai.Client(api_key=key)
        else:
            raise ValueError(f"Unknown provider: {provider}")
    except ImportError as e:
        raise RuntimeError(
            f"The SDK for provider '{provider}' is not installed "
            f"(pip install {_PIP_PACKAGE[provider]}): {e}"
        ) from e

    if api_key is None:
        _CLIENT_CACHE[provider] = client
    return client


# ---------------------------------------------------------------------------
# Per-provider completion
# ---------------------------------------------------------------------------


def _anthropic_complete(
    client: Any, model: str, prompt: str, max_tokens: int, temperature: Optional[float]
) -> LLMResponse:
    from model_costs import sampling_kwargs

    kwargs: Dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if temperature is not None:
        kwargs.update(sampling_kwargs(model, temperature))

    if max_tokens > STREAM_ABOVE_MAX_TOKENS:
        # Large budgets must stream or the request can outlive the HTTP timeout.
        with client.messages.stream(**kwargs) as stream:
            resp = stream.get_final_message()
    else:
        resp = client.messages.create(**kwargs)

    # Skip thinking blocks — only TextBlocks carry a .text attribute.
    text = " ".join(b.text for b in resp.content if hasattr(b, "text"))
    u = getattr(resp, "usage", None)
    return LLMResponse(
        stop_reason=getattr(resp, "stop_reason", None),
        text=text,
        usage={
            "input_tokens": getattr(u, "input_tokens", None),
            "output_tokens": getattr(u, "output_tokens", None),
            "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", None),
            "cache_creation_input_tokens": getattr(
                u, "cache_creation_input_tokens", None
            ),
        },
        model=model,
        provider="anthropic",
    )


def _openai_complete(
    client: Any,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: Optional[float],
    provider: str = "openai",
) -> LLMResponse:
    """Chat-completions call for OpenAI and OpenAI-compatible APIs (DeepSeek).

    Reasoning models take `max_completion_tokens` and reject `temperature`.
    The token-limit parameter name is retried on rejection so a newly released
    model that flips convention still works without a code change.
    """
    is_reasoning = model.lower().startswith(_OPENAI_REASONING_PREFIXES)

    def _build(token_param: str) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            token_param: max_tokens,
        }
        if temperature is not None and not is_reasoning:
            kwargs["temperature"] = temperature
        return kwargs

    first = "max_completion_tokens" if is_reasoning else "max_tokens"
    fallback = "max_tokens" if is_reasoning else "max_completion_tokens"

    try:
        resp = client.chat.completions.create(**_build(first))
    except Exception as e:
        msg = str(e).lower()
        if first in msg and ("unsupported" in msg or "not supported" in msg or "use" in msg):
            resp = client.chat.completions.create(**_build(fallback))
        else:
            raise

    text = (resp.choices[0].message.content or "") if resp.choices else ""
    finish = resp.choices[0].finish_reason if resp.choices else None
    u = getattr(resp, "usage", None)
    cached = 0
    details = getattr(u, "prompt_tokens_details", None) if u else None
    if details is not None:
        cached = getattr(details, "cached_tokens", 0) or 0
    prompt_tokens = getattr(u, "prompt_tokens", None) if u else None
    # Report the uncached remainder as input, matching Anthropic's usage shape.
    if isinstance(prompt_tokens, int) and cached:
        prompt_tokens = max(prompt_tokens - cached, 0)

    return LLMResponse(
        text=text,
        stop_reason=finish,
        usage={
            "input_tokens": prompt_tokens,
            "output_tokens": getattr(u, "completion_tokens", None) if u else None,
            "cache_read_input_tokens": cached or None,
            "cache_creation_input_tokens": None,
        },
        model=model,
        provider=provider,
    )


def _google_complete(
    client: Any, model: str, prompt: str, max_tokens: int, temperature: Optional[float]
) -> LLMResponse:
    from google.genai import types

    config_kwargs: Dict[str, Any] = {"max_output_tokens": max_tokens}
    if temperature is not None:
        config_kwargs["temperature"] = temperature

    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(**config_kwargs),
    )

    text = resp.text or ""
    finish = None
    candidates = getattr(resp, "candidates", None)
    if candidates:
        fr = getattr(candidates[0], "finish_reason", None)
        finish = getattr(fr, "name", None) or (str(fr) if fr is not None else None)
    u = getattr(resp, "usage_metadata", None)
    thoughts = getattr(u, "thoughts_token_count", None) if u else None
    cand_tokens = getattr(u, "candidates_token_count", None) if u else None
    # Gemini bills thinking tokens as output; report them together.
    output_tokens = None
    if cand_tokens is not None or thoughts is not None:
        output_tokens = (cand_tokens or 0) + (thoughts or 0)

    return LLMResponse(
        text=text,
        stop_reason=finish,
        usage={
            "input_tokens": getattr(u, "prompt_token_count", None) if u else None,
            "output_tokens": output_tokens,
            "cache_read_input_tokens": (
                getattr(u, "cached_content_token_count", None) if u else None
            ),
            "cache_creation_input_tokens": None,
        },
        model=model,
        provider="google",
    )


def complete(
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: Optional[float] = None,
    api_key: Optional[str] = None,
    client: Optional[Any] = None,
) -> LLMResponse:
    """Send one prompt to `model` and return its text plus token usage.

    `temperature` is a request, not a guarantee — it is dropped for models
    that reject sampling parameters (recent Anthropic models, OpenAI
    reasoning models).
    """
    provider, model_id = resolve_provider(model)
    if client is None:
        client = get_client(provider, api_key)

    if provider == "anthropic":
        return _anthropic_complete(client, model_id, prompt, max_tokens, temperature)
    if provider in ("openai", "deepseek"):
        return _openai_complete(
            client, model_id, prompt, max_tokens, temperature, provider=provider
        )
    if provider == "google":
        return _google_complete(client, model_id, prompt, max_tokens, temperature)
    raise ValueError(f"Unknown provider: {provider}")
