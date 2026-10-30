#!/usr/bin/env python3
"""
model_costs.py — Model metadata: name slugs, cost estimation, parameter support.

Prices live in configs/model_pricing.json so they can be updated without code
changes. A model with no entry there yields a None cost (never a guessed one);
add it to the config to price it.
"""

import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PRICING_PATH = PROJECT_ROOT / "configs" / "model_pricing.json"


# Model families that reject temperature / top_p / top_k with a 400. Sampling
# parameters were removed starting with Opus 4.7; passing one to any of these
# fails the request outright, so callers must omit it rather than send 0.0.
_NO_SAMPLING_PARAMS = (
    "claude-fable-",
    "claude-mythos-",
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-sonnet-5",
)


def model_slug(model: str) -> str:
    """Filesystem- and tag-safe short name for a model ID."""
    return model.replace("/", "_").replace(":", "_").replace(".", "-")


def supports_sampling_params(model: str) -> bool:
    """Whether `temperature` / `top_p` / `top_k` may be sent to this model."""
    return not model.startswith(_NO_SAMPLING_PARAMS)


def sampling_kwargs(model: str, temperature: float) -> Dict[str, float]:
    """`{"temperature": t}` for models that accept it, `{}` for those that don't."""
    return {"temperature": temperature} if supports_sampling_params(model) else {}


def load_pricing(path: Optional[Path] = None) -> Dict[str, Any]:
    p = path or DEFAULT_PRICING_PATH
    if not p.exists():
        return {"models": {}, "cache_multipliers": {}}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def price_for(
    model: str,
    pricing: Dict[str, Any],
    on_date: Optional[date] = None,
) -> Optional[Dict[str, float]]:
    """Return {"input": $/1M, "output": $/1M} for a model, or None if unpriced.

    Matches the model ID exactly, then falls back to the longest configured key
    that the ID starts with (so a date-suffixed snapshot inherits its family's
    price). Introductory pricing applies up to and including `intro_until`.
    """
    # Accept the explicit `provider:model` form used by llm_providers.
    if ":" in model:
        model = model.split(":", 1)[1]

    models = pricing.get("models", {})
    entry = models.get(model)
    if entry is None:
        candidates = [k for k in models if model.startswith(k)]
        if not candidates:
            return None
        entry = models[max(candidates, key=len)]

    today = on_date or date.today()
    intro_until = entry.get("intro_until")
    if intro_until and "intro_input" in entry and "intro_output" in entry:
        try:
            if today <= date.fromisoformat(intro_until):
                return {
                    "input": float(entry["intro_input"]),
                    "output": float(entry["intro_output"]),
                }
        except ValueError:
            pass

    if "input" not in entry or "output" not in entry:
        return None
    return {"input": float(entry["input"]), "output": float(entry["output"])}


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    pricing: Dict[str, Any],
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> Optional[float]:
    """Estimated USD cost for the given token counts, or None if unpriced.

    `input_tokens` is the uncached remainder; cached tokens are billed at the
    configured cache multipliers on top of it.
    """
    prices = price_for(model, pricing)
    if prices is None:
        return None

    mult = pricing.get("cache_multipliers", {})
    read_mult = float(mult.get("cache_read", 0.1))
    write_mult = float(mult.get("cache_write_5m", 1.25))

    per_input = prices["input"] / 1_000_000.0
    per_output = prices["output"] / 1_000_000.0

    return (
        input_tokens * per_input
        + cache_read_tokens * per_input * read_mult
        + cache_write_tokens * per_input * write_mult
        + output_tokens * per_output
    )
