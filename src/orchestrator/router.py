"""Phase 6: Verifier routing logic.

Maps verification snippets to the appropriate verifier based on snippet type.
Uses the routing table from PipelineConfig with fallback logic.
"""

from __future__ import annotations

from typing import Optional

from loguru import logger

from src.config import PipelineConfig, default_config
from src.models import SnippetType, VerificationSnippet
from src.verifiers.base import BaseVerifier
from src.verifiers.citation_verifier import CitationVerifier
from src.verifiers.llm_only_verifier import LLMOnlyVerifier
from src.verifiers.math_verifier import MathEquationVerifier
from src.verifiers.progressive.progressive_verifier import ProgressiveMathVerifier
from src.verifiers.registry import VerifierRegistry
from src.verifiers.statistical_verifier import StatisticalVerifier
from src.verifiers.text_verifier import TextVerifier
from src.verifiers.triage_verifier import TriageVerifier
from src.verifiers.vision_verifier import VisionVerifier


def create_default_registry() -> VerifierRegistry:
    """Create and populate a VerifierRegistry with the default verifiers.

    This is the standard set of verifiers. Additional verifiers can be
    registered by calling registry.register() on the returned instance.

    Returns:
        A VerifierRegistry populated with math, vision, text, and triage verifiers.
    """
    registry = VerifierRegistry()
    registry.register("math_equation", MathEquationVerifier)
    registry.register("vision", VisionVerifier)
    registry.register("text", TextVerifier)
    registry.register("statistical", StatisticalVerifier)
    registry.register("citation", CitationVerifier)
    registry.register("triage", TriageVerifier)
    registry.register("llm_only", LLMOnlyVerifier)
    registry.register("progressive_math", ProgressiveMathVerifier)
    logger.debug("Default verifier registry created with 8 verifiers")
    return registry


def resolve_route_to_verifier(
    route: str,
    snippet: VerificationSnippet,
    config: Optional[PipelineConfig] = None,
) -> Optional[str]:
    """Map a triage `route` label to a concrete, registered verifier name.

    Falls back to type-based routing when the route is unknown or maps to a
    verifier that cannot handle this snippet. Returns ``None`` for the explicit
    "none" route (no specialist needed).

    Args:
        route: Semantic route label suggested by the triage verifier.
        snippet: The snippet being routed (used for type-based fallback).
        config: Pipeline configuration carrying ``triage_route_map``.

    Returns:
        A registered verifier name, or None to skip specialist verification.
    """
    if config is None:
        config = default_config

    route = (route or "").strip().lower()
    mapped = config.triage_route_map.get(route)

    if mapped == "":
        # Explicit "none" → caller decides; default is to skip.
        return None
    if mapped:
        return mapped

    # Unknown route → fall back to the structural (type-based) router.
    logger.debug(f"Unknown triage route '{route}'; falling back to type routing.")
    return select_verifier_name(snippet, config)


def select_verifier_name(
    snippet: VerificationSnippet,
    config: Optional[PipelineConfig] = None,
) -> str:
    """Select the verifier name for a given snippet.

    Uses the routing table from the pipeline configuration. Falls back
    to "text" for unknown snippet types.

    Args:
        snippet: The verification snippet to route.
        config: Pipeline configuration with routing table.

    Returns:
        The name of the verifier to use.
    """
    if config is None:
        config = default_config

    snippet_type = snippet.snippet_type.value
    routing = config.verifier_routing

    verifier_name = routing.get(snippet_type, "text")
    logger.debug(f"Routing {snippet.snippet_id} ({snippet_type}) → {verifier_name}")

    return verifier_name


def select_verifier(
    snippet: VerificationSnippet,
    registry: VerifierRegistry,
    config: Optional[PipelineConfig] = None,
) -> type[BaseVerifier]:
    """Select and return the verifier class for a snippet.

    Args:
        snippet: The verification snippet.
        registry: The verifier registry to look up classes.
        config: Pipeline configuration.

    Returns:
        The verifier class.

    Raises:
        KeyError: If the routed verifier name is not in the registry.
    """
    verifier_name = select_verifier_name(snippet, config)
    return registry.get(verifier_name)
