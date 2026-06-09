"""LLM abstraction layer for verifier interactions.

All LLM calls go through this module so the backend can be swapped
between Anthropic, OpenAI, local models, or mock mode for testing.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

from loguru import logger

from src.config import LLMConfig


class LLMError(Exception):
    """Raised when an LLM call fails."""


def _mock_llm_call(
    prompt: str,
    system_prompt: str = "",
    image_path: Optional[str] = None,
    **_kwargs: Any,
) -> str:
    """Mock LLM backend that returns deterministic responses for testing.

    Returns different responses based on prompt content keywords to allow
    basic pipeline testing without API calls.
    """
    prompt_lower = prompt.lower()
    system_lower = system_prompt.lower()

    # Triage mock — produces a deterministic uncertainty map. Equations and
    # anything mentioning suspicious keywords get higher uncertainty.
    if "triage" in system_lower:
        if "excerpt type: equation" in prompt_lower:
            return json.dumps(
                {"uncertainty": 0.72, "route": "math",
                 "reason": "Mock: equation worth a symbolic check."}
            )
        if any(k in prompt_lower for k in ("incorrect", "error", "contradict", "wrong")):
            return json.dumps(
                {"uncertainty": 0.65, "route": "text",
                 "reason": "Mock: surfaced suspicious wording."}
            )
        if "excerpt type: figure" in prompt_lower or "excerpt type: table" in prompt_lower:
            return json.dumps(
                {"uncertainty": 0.4, "route": "vision",
                 "reason": "Mock: visual content."}
            )
        return json.dumps(
            {"uncertainty": 0.08, "route": "none",
             "reason": "Mock: routine prose."}
        )

    # Single-call whole-paper baseline mock. Key off content-specific words only
    # ("error" appears in the prompt's own instruction line, so it is excluded).
    if "single-call whole-paper review" in system_lower:
        if any(k in prompt_lower for k in ("incorrect", "wrong", "contradict", "flawed")):
            return json.dumps(
                {"errors": [{"error_category": "Equation / proof",
                             "error_location": "Section 1",
                             "confidence": 0.8,
                             "supporting_evidence": "Mock: whole-paper review flagged a likely error."}]}
            )
        return json.dumps({"errors": []})

    # Statistical extraction mock — returns a deterministic check.
    if "checkable numeric" in system_lower or "extract checkable" in system_lower:
        if "60" in prompt and "30" in prompt and "100" in prompt:
            # Reported parts (60% + 30%) do not sum to the stated 100% total.
            return json.dumps(
                {"checks": [{"description": "reported percentages sum to stated total",
                             "expr": "60 + 30", "expected": 100, "tolerance": 0.01}],
                 "unit_checks": []}
            )
        return json.dumps({"checks": [], "unit_checks": []})

    # Citation/attribution/novelty mock.
    if "novelty" in system_lower or "attribution" in system_lower:
        if "already" in prompt_lower or "previously established" in prompt_lower:
            return json.dumps(
                {"error_detected": True, "confidence": 0.85,
                 "reasoning": "Mock: novelty over-claim contradicted within the excerpt.",
                 "predicted_error_category": "Equation / proof"}
            )
        return json.dumps(
            {"error_detected": False, "confidence": 0.9,
             "reasoning": "Mock: no attribution contradiction.",
             "predicted_error_category": None}
        )

    # Progressive math verifier — checked BEFORE equation mock because
    # its prompts also contain "sympy" and "latex".
    if "statement_class" in system_lower or "accumulated paper context" in prompt_lower:
        return _mock_progressive_math(prompt)

    # Equation verification mock
    if "sympy" in prompt_lower or "latex" in prompt_lower:
        # Check for specific equation patterns
        if "x^{\\dagger\\dagger}" in prompt or "bidual" in prompt_lower:
            return json.dumps(
                {
                    "sympy_code": (
                        "from sympy import *\n"
                        "# Complex bidual verification\n"
                        "x = symbols('x')\n"
                        "report_valid('bidual isomorphism holds')\n"
                    ),
                    "equation_type": "identity",
                    "explanation": "Bidual isomorphism verified symbolically.",
                }
            )
        if (
            "co2" in prompt_lower
            or "climate" in prompt_lower
            or "lambda" in prompt_lower
        ):
            return json.dumps(
                {
                    "sympy_code": (
                        "from sympy import *\n"
                        "C, C0, Lambda = symbols('C C0 Lambda')\n"
                        "F = 5.35 * log(C/C0)\n"
                        "Delta = Lambda * F\n"
                        "report(simplify(Delta - Lambda * 5.35 * log(C/C0)))\n"
                    ),
                    "equation_type": "definition",
                    "explanation": "Radiative forcing equation derived.",
                }
            )
        return json.dumps(
            {
                "sympy_code": (
                    "from sympy import *\n"
                    "x = symbols('x')\n"
                    "lhs = x**2\n"
                    "rhs = x*x\n"
                    "report(simplify(lhs - rhs))\n"
                ),
                "equation_type": "identity",
                "explanation": "Equation simplified symbolically.",
            }
        )

    # Vision verification mock
    if "figure" in prompt_lower or "image" in prompt_lower or "table" in prompt_lower:
        if "duplicat" in prompt_lower:
            return json.dumps(
                {
                    "error_detected": True,
                    "confidence": 0.85,
                    "reasoning": "Mock: Detected potential figure duplication based on pattern analysis.",
                    "predicted_error_category": "Figure duplication",
                }
            )
        if "inconsist" in prompt_lower:
            return json.dumps(
                {
                    "error_detected": True,
                    "confidence": 0.72,
                    "reasoning": "Mock: Possible data inconsistency detected between figure and caption.",
                    "predicted_error_category": "Data Inconsistency (figure-text)",
                }
            )
        return json.dumps(
            {
                "error_detected": False,
                "confidence": 0.90,
                "reasoning": "Mock: No issues detected in visual content.",
                "predicted_error_category": None,
            }
        )

    # Text verification mock
    if (
        "contradiction" in prompt_lower
        or "inconsist" in prompt_lower
        or "logical" in prompt_lower
    ):
        if "theorem" in prompt_lower and (
            "proof" in prompt_lower or "lemma" in prompt_lower
        ):
            return json.dumps(
                {
                    "error_detected": True,
                    "confidence": 0.78,
                    "reasoning": "Mock: Detected potential gap between theorem statement and proof.",
                    "predicted_error_category": "Equation / proof",
                }
            )
        return json.dumps(
            {
                "error_detected": False,
                "confidence": 0.82,
                "reasoning": "Mock: No logical contradictions found in the text segment.",
                "predicted_error_category": None,
            }
        )

    # Default response
    return json.dumps(
        {
            "error_detected": False,
            "confidence": 0.95,
            "reasoning": "Mock: Default no-error response.",
            "predicted_error_category": None,
        }
    )


def _mock_progressive_math(prompt: str) -> str:
    """Mock responses for ProgressiveMathVerifier's three-statement-class format."""
    prompt_lower = prompt.lower()

    # Declaration: domain membership (check for \mathbb or \in patterns)
    if "\\mathbb" in prompt or "\\in" in prompt:
        if "\\mathbb{r}" in prompt_lower:
            return json.dumps({
                "statement_class": "uncheckable_declaration",
                "sympy_code": "from sympy import *\nx = Symbol('x', real=True)\nreport_assumption_added(['x'], 'x is declared real')",
                "symbols_introduced": ["x"],
                "symbol_domains": {"x": "real"},
                "defines_symbols": {},
                "assumptions_added": ["x ∈ ℝ"],
                "depends_on_symbols": [],
                "explanation": "Mock: symbol x declared as real.",
            })
        if "\\mathbb{n}" in prompt_lower:
            return json.dumps({
                "statement_class": "uncheckable_declaration",
                "sympy_code": "from sympy import *\nn = Symbol('n', integer=True, nonnegative=True)\nreport_assumption_added(['n'], 'n ∈ ℕ')",
                "symbols_introduced": ["n"],
                "symbol_domains": {"n": "natural"},
                "defines_symbols": {},
                "assumptions_added": ["n ∈ ℕ"],
                "depends_on_symbols": [],
                "explanation": "Mock: symbol n declared as natural.",
            })
        if "\\mathbb{z}" in prompt_lower:
            return json.dumps({
                "statement_class": "uncheckable_declaration",
                "sympy_code": "from sympy import *\nn = Symbol('n', integer=True)\nreport_assumption_added(['n'], 'n ∈ ℤ')",
                "symbols_introduced": ["n"],
                "symbol_domains": {"n": "integer"},
                "defines_symbols": {},
                "assumptions_added": ["n ∈ ℤ"],
                "depends_on_symbols": [],
                "explanation": "Mock: symbol n declared as integer.",
            })

    # Declaration: positivity / inequality
    if "> 0" in prompt and "sqrt" not in prompt_lower:
        return json.dumps({
            "statement_class": "uncheckable_declaration",
            "sympy_code": "from sympy import *\nx = Symbol('x', real=True, positive=True)\nreport_assumption_added(['x'], 'x > 0')",
            "symbols_introduced": ["x"],
            "symbol_domains": {"x": "positive_real"},
            "defines_symbols": {},
            "assumptions_added": ["x > 0"],
            "depends_on_symbols": [],
            "explanation": "Mock: x is positive.",
        })

    # Known broken identity: (a+b)^2 = a^2 + b^2 (check BEFORE definition)
    if "(a+b)^2" in prompt_lower or "(a + b)^2" in prompt_lower:
        return json.dumps({
            "statement_class": "checkable_derivation",
            "sympy_code": "from sympy import *\na, b = symbols('a b')\nlhs = (a+b)**2\nrhs = a**2 + b**2\nreport(simplify(lhs - rhs))",
            "symbols_introduced": ["a", "b"],
            "symbol_domains": {"a": "real", "b": "real"},
            "defines_symbols": {},
            "assumptions_added": [],
            "depends_on_symbols": [],
            "explanation": "Mock: claiming (a+b)² = a² + b² (missing 2ab).",
        })

    # Definition: symbol = expression (explicit "let" or "define" in prompt)
    if "=" in prompt and any(kw in prompt_lower for kw in ("define", "let", "definition")):
        return json.dumps({
            "statement_class": "uncheckable_declaration",
            "sympy_code": "from sympy import *\nB, Q, N = symbols('B Q N')\nreport_definition_added('B', 'Q/N')",
            "symbols_introduced": ["B"],
            "symbol_domains": {"B": "unknown"},
            "defines_symbols": {"B": "Q/N"},
            "assumptions_added": ["B is defined as Q/N"],
            "depends_on_symbols": ["Q", "N"],
            "explanation": "Mock: B is defined as Q/N.",
        })

    # Conditional / constraint equation
    if any(kw in prompt_lower for kw in ("constraint", "condition", "circle", "unit")):
        return json.dumps({
            "statement_class": "checkable_constraint",
            "sympy_code": "from sympy import *\ns, t = symbols('s t')\nlhs = s**2 + t**2\nrhs = 1\nreport(simplify(lhs - rhs))",
            "symbols_introduced": ["s", "t"],
            "symbol_domains": {"s": "real", "t": "real"},
            "defines_symbols": {},
            "assumptions_added": ["s² + t² = 1 (unit circle constraint)"],
            "depends_on_symbols": [],
            "explanation": "Mock: constraint equation on the unit circle.",
        })

    # Climate / physics equation
    if any(kw in prompt_lower for kw in ("co2", "climate", "lambda", "radiative")):
        return json.dumps({
            "statement_class": "checkable_derivation",
            "sympy_code": "from sympy import *\nC, C0, Lambda = symbols('C C0 Lambda')\nF = 5.35 * log(C/C0)\nDelta = Lambda * F\nlhs = Delta\nrhs = Lambda * 5.35 * log(C/C0)\nreport(simplify(lhs - rhs))",
            "symbols_introduced": [],
            "symbol_domains": {},
            "defines_symbols": {},
            "assumptions_added": [],
            "depends_on_symbols": [],
            "explanation": "Mock: radiative forcing equation.",
        })

    # Default mock for progressive — simple valid identity
    return json.dumps({
        "statement_class": "checkable_derivation",
        "sympy_code": "from sympy import *\nx, y = symbols('x y')\nlhs = x**2\nrhs = x*x\nreport(simplify(lhs - rhs))",
        "symbols_introduced": ["x", "y"],
        "symbol_domains": {"x": "unknown", "y": "unknown"},
        "defines_symbols": {},
        "assumptions_added": [],
        "depends_on_symbols": [],
        "explanation": "Mock: generic identity x² = x·x.",
    })


def llm_call(
    prompt: str,
    system_prompt: str = "",
    model: Optional[str] = None,
    image_path: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: float = 0.0,
    config: Optional[LLMConfig] = None,
) -> str:
    """Call an LLM with the given prompt and return the response text.

    This is the single entry point for all LLM interactions in the pipeline.
    The backend is determined by config.provider.

    Args:
        prompt: The user/input prompt.
        system_prompt: Optional system-level instructions.
        model: Model name override (uses config if not set).
        image_path: Optional path to an image for multimodal requests.
        max_tokens: Maximum tokens in the response.
        temperature: Sampling temperature.
        config: LLM configuration; uses defaults if None.

    Returns:
        The LLM's text response.

    Raises:
        LLMError: If the LLM call fails.
    """
    if config is None:
        config = LLMConfig()

    if model is None:
        model = config.model

    # Honor the configured output-token budget when the caller doesn't override
    # it. This matters for reasoning-style models: too small a budget can be
    # consumed by chain-of-thought, leaving `message.content` EMPTY — the root
    # cause of the intermittent empty responses on long/dense inputs.
    if max_tokens is None:
        max_tokens = config.max_tokens

    provider = config.provider.lower()

    logger.debug(
        f"LLM call: provider={provider}, model={model}, "
        f"prompt_len={len(prompt)}, has_image={image_path is not None}"
    )

    if provider == "mock":
        return _mock_llm_call(
            prompt=prompt,
            system_prompt=system_prompt,
            image_path=image_path,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    if provider == "anthropic":
        invoke = _anthropic_call
    elif provider == "openai":
        invoke = _openai_call
    elif provider == "deepseek":
        invoke = _deepseek_call
    else:
        raise LLMError(f"Unknown LLM provider: {provider}")

    # Retry with exponential backoff to absorb transient failures (rate limits,
    # network blips) that become more likely under concurrent execution.
    max_retries = max(1, getattr(config, "max_retries", 1))
    backoff = getattr(config, "retry_backoff_seconds", 2.0)
    last_exc: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            result = invoke(
                prompt=prompt,
                system_prompt=system_prompt,
                model=model,
                image_path=image_path,
                max_tokens=max_tokens,
                temperature=temperature,
                api_key_env=config.api_key_env,
            )
            # An empty/whitespace completion is a successful HTTP call but
            # useless to every downstream parser — under concurrency providers
            # (esp. DeepSeek) intermittently return these. Treat as retryable
            # rather than letting a silent "" suppress a finding.
            if result is None or not result.strip():
                raise LLMError("LLM returned an empty response")
            return result
        except Exception as exc:  # provider/network/rate-limit/empty errors
            last_exc = exc
            if attempt >= max_retries:
                break
            sleep_for = backoff * (2 ** (attempt - 1))
            logger.warning(
                f"LLM call failed (attempt {attempt}/{max_retries}): {exc}. "
                f"Retrying in {sleep_for:.1f}s..."
            )
            time.sleep(sleep_for)

    raise LLMError(
        f"LLM call failed after {max_retries} attempt(s): {last_exc}"
    ) from last_exc


def _anthropic_call(
    prompt: str,
    system_prompt: str = "",
    model: str = "claude-sonnet-4-6",
    image_path: Optional[str] = None,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    api_key_env: str = "ANTHROPIC_API_KEY",
) -> str:
    """Call the Anthropic Claude API."""
    try:
        import anthropic
    except ImportError:
        raise LLMError(
            "anthropic package not installed. Install with: pip install anthropic"
        )

    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise LLMError(
            f"Anthropic API key not found. Set {api_key_env} environment variable."
        )

    client = anthropic.Anthropic(api_key=api_key)

    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]

    if image_path:
        import base64
        from pathlib import Path

        img_path = Path(image_path)
        if not img_path.exists():
            raise LLMError(f"Image file not found: {image_path}")

        with open(img_path, "rb") as f:
            img_data = base64.b64encode(f.read()).decode("utf-8")

        # Detect image type from extension
        suffix = img_path.suffix.lower()
        media_type = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }.get(suffix, "image/jpeg")

        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": img_data,
                },
            }
        )

    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_prompt,
        messages=[{"role": "user", "content": content}],
    )

    # Extract text from the first content block
    for block in message.content:
        if block.type == "text":
            return block.text

    raise LLMError("No text content in Anthropic response")


def _openai_call(
    prompt: str,
    system_prompt: str = "",
    model: str = "gpt-5.2",
    image_path: Optional[str] = None,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    api_key_env: str = "OPENAI_API_KEY",
) -> str:
    """Call the OpenAI API."""
    try:
        import openai
    except ImportError:
        raise LLMError("openai package not installed. Install with: pip install openai")

    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise LLMError(
            f"OpenAI API key not found. Set {api_key_env} environment variable."
        )

    client = openai.OpenAI(api_key=api_key)

    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]

    if image_path:
        import base64
        from pathlib import Path

        img_path = Path(image_path)
        if not img_path.exists():
            raise LLMError(f"Image file not found: {image_path}")

        with open(img_path, "rb") as f:
            img_data = base64.b64encode(f.read()).decode("utf-8")

        suffix = img_path.suffix.lower()
        media_type = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }.get(suffix, "image/jpeg")

        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{img_data}"},
            }
        )

    messages.append({"role": "user", "content": content})

    if model.startswith("gpt-5"):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
    else:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    return response.choices[0].message.content or ""


def _deepseek_call(
    prompt: str,
    system_prompt: str = "",
    model: str = "deepseek-chat",
    image_path: Optional[str] = None,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    api_key_env: str = "DEEPSEEK_API_KEY",
) -> str:
    """Call the DeepSeek API (OpenAI-compatible)."""
    try:
        import openai
    except ImportError:
        raise LLMError(
            "openai package not installed (required for DeepSeek compatibility). "
            "Install with: pip install openai"
        )

    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise LLMError(
            f"DeepSeek API key not found. Set {api_key_env} environment variable."
        )

    # Generous client-side timeout: a whole-paper / dense-proof request on a
    # reasoning model can take minutes. Without this the SDK can surface a
    # premature "Connection error." on long generations.
    client = openai.OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        timeout=300.0,
        max_retries=0,  # retries are handled in llm_call with backoff
    )

    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]

    if image_path:
        import base64
        from pathlib import Path

        img_path = Path(image_path)
        if not img_path.exists():
            raise LLMError(f"Image file not found: {image_path}")

        with open(img_path, "rb") as f:
            img_data = base64.b64encode(f.read()).decode("utf-8")

        suffix = img_path.suffix.lower()
        media_type = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }.get(suffix, "image/jpeg")

        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{img_data}"},
            }
        )

    messages.append({"role": "user", "content": content})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        # max_tokens=max_tokens,
        temperature=temperature,
    )

    return response.choices[0].message.content or ""


def parse_json_response(response: str) -> dict[str, Any]:
    """Parse a JSON response from an LLM, handling markdown code blocks.

    Args:
        response: Raw LLM response text.

    Returns:
        Parsed JSON dictionary.

    Raises:
        LLMError: If the response cannot be parsed as JSON.
    """
    text = response.strip()

    # Handle markdown code blocks
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove opening ```json or ```
        if lines[0].startswith("```"):
            lines = lines[1:]
        # Remove closing ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    # Handle case where JSON is embedded in text
    # Try to find the first { and last }
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        text = text[first_brace : last_brace + 1]

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning(f"Failed to parse LLM JSON response: {exc}")
        logger.debug(f"Raw response: {response[:500]}")
        raise LLMError(f"Failed to parse LLM response as JSON: {exc}") from exc
