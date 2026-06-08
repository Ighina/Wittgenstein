"""Abstract base verifier and shared utilities.

All verifiers inherit from BaseVerifier and implement the verify() method.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Optional

from loguru import logger

from src.config import PipelineConfig, VerifierConfig, default_config
from src.models import (
    BaseVerificationResult,
    VerificationSnippet,
    VerificationStatus,
)
from src.utils.chunking import chunk_text
from src.utils.llm import llm_call, parse_json_response


class BaseVerifier(ABC):
    """Abstract base class for all verifiers.

    Subclasses must:
    1. Set `name` (str) — unique verifier identifier.
    2. Implement `verify(snippet)` → BaseVerificationResult.
    3. Optionally override `can_verify(snippet)` for eligibility checks.
    """

    name: str = "base"

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
    ) -> None:
        self.config = config or default_config
        self.verifier_config: VerifierConfig = (
            self.config.verifiers.get(self.name, VerifierConfig())
        )

    @abstractmethod
    def verify(self, snippet: VerificationSnippet) -> BaseVerificationResult:
        """Verify a single snippet.

        Args:
            snippet: The verification snippet to analyze.

        Returns:
            A verification result with status and findings.
        """
        ...

    def can_verify(self, snippet: VerificationSnippet) -> bool:
        """Check if this verifier can handle the given snippet type.

        Override in subclasses for more specific eligibility checks.
        """
        return True

    def _make_result(
        self,
        snippet_id: str,
        status: VerificationStatus,
        error_detected: bool = False,
        confidence: float = 0.0,
        reasoning: str = "",
        predicted_error_category: Optional[str] = None,
        execution_time_ms: float = 0.0,
        **extra: Any,
    ) -> BaseVerificationResult:
        """Create a verification result with common fields."""
        return BaseVerificationResult(
            snippet_id=snippet_id,
            verifier_name=self.name,
            status=status,
            error_detected=error_detected,
            confidence=confidence,
            reasoning=reasoning,
            predicted_error_category=predicted_error_category,
            execution_time_ms=execution_time_ms,
        )

    def _call_llm(
        self,
        prompt: str,
        system_prompt: str = "",
        image_path: Optional[str] = None,
    ) -> str:
        """Wrapper around the centralized LLM call."""
        return llm_call(
            prompt=prompt,
            system_prompt=system_prompt,
            image_path=image_path,
            config=self.config.llm,
        )

    def _call_llm_json(
        self,
        prompt: str,
        system_prompt: str = "",
        image_path: Optional[str] = None,
    ) -> dict[str, Any]:
        """Call LLM and parse the response as JSON."""
        response = self._call_llm(
            prompt=prompt,
            system_prompt=system_prompt,
            image_path=image_path,
        )
        return parse_json_response(response)

    def _analyze_in_chunks(
        self,
        content: str,
        analyze_chunk,  # Callable[[str], dict] — must return a normalized finding
    ) -> tuple[Optional[dict], int, int]:
        """Verify long ``content`` chunk-by-chunk and aggregate the findings.

        ``analyze_chunk(chunk_text)`` runs the per-chunk LLM analysis and returns
        a dict with at least ``error_detected`` (bool), ``confidence`` (float),
        and ``reasoning`` (str). It may raise — a raising chunk is counted as
        failed (e.g. an empty LLM completion) and skipped, so one bad chunk does
        not sink the whole snippet.

        Aggregation:
          * Any chunk that detects an error → return the highest-confidence such
            finding (prefixed with its chunk position when there are several).
          * Otherwise, if at least one chunk succeeded → return the highest-
            confidence "no error" finding.
          * If every chunk failed → return (None, n_chunks, n_failed); the caller
            should treat this as UNVERIFIABLE.

        Returns:
            (chosen_finding_or_None, n_chunks, n_failed)
        """
        chunks = chunk_text(
            content,
            max_chars=self.config.verify_chunk_chars,
            overlap=self.config.verify_chunk_overlap,
        )
        n_chunks = len(chunks)
        findings: list[tuple[int, dict]] = []
        n_failed = 0

        for idx, chunk in enumerate(chunks):
            try:
                finding = analyze_chunk(chunk)
                findings.append((idx, finding))
            except Exception as exc:  # empty completion, parse error, etc.
                n_failed += 1
                logger.debug(f"Chunk {idx + 1}/{n_chunks} analysis failed: {exc}")

        if not findings:
            return None, n_chunks, n_failed

        errors = [(i, f) for i, f in findings if f.get("error_detected")]
        if errors:
            idx, chosen = max(errors, key=lambda x: x[1].get("confidence", 0.0))
        else:
            idx, chosen = max(findings, key=lambda x: x[1].get("confidence", 0.0))

        if n_chunks > 1:
            chosen = dict(chosen)
            chosen["reasoning"] = f"[chunk {idx + 1}/{n_chunks}] " + str(chosen.get("reasoning", ""))
        return chosen, n_chunks, n_failed

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
