"""Phase 5: Verification orchestrator.

Coordinates the entire verification pipeline for a single paper:
Parse → Segment → Route → Verify → Aggregate → Predictions.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from loguru import logger
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.config import PipelineConfig, default_config
from src.models import (
    BaseVerificationResult,
    NormalizedPaper,
    PaperPrediction,
    PredictedError,
    VerificationSnippet,
    EnrichedPaper,
)
from src.segmentation.segmenter import segment_paper
from src.parser.enriched_segmenter import segment_enriched_paper
from src.parser.llm_content_parser import llm_parse_paper
from src.verifiers.base import BaseVerifier
from src.verifiers.registry import VerifierRegistry
from src.orchestrator.router import (
    create_default_registry,
    select_verifier_name,
    resolve_route_to_verifier,
)


class VerificationOrchestrator:
    """Orchestrates the verification of a scientific paper.

    Coordinates parsing, segmentation, verifier routing, execution,
    and aggregation of findings into paper-level predictions.

    Usage:
        orchestrator = VerificationOrchestrator(config)
        prediction = orchestrator.run(normalized_paper)
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        registry: Optional[VerifierRegistry] = None,
    ) -> None:
        self.config = config or default_config
        self.registry = registry or create_default_registry()

        # Cache verifier instances by name to avoid re-creation
        self._verifier_instances: dict[str, BaseVerifier] = {}

    def run(
        self,
        paper: NormalizedPaper | EnrichedPaper,
        progress: Optional[Progress] = None,
    ) -> PaperPrediction:
        """Run the full verification pipeline on a single paper.

        When config.parser_mode == "regex" (default), ``paper`` should be a
        NormalizedPaper; the internal segmenter is used.

        When config.parser_mode == "llm", ``paper`` should be an EnrichedPaper
        (already parsed by llm_parse_paper); the enriched segmenter is used.

        Args:
            paper: The parsed paper (NormalizedPaper or EnrichedPaper).
            progress: Optional Rich progress bar.

        Returns:
            PaperPrediction with all findings aggregated.
        """
        logger.info(f"Starting verification of paper: {paper.paper_id}")
        t0 = time.monotonic()

        # Step 1: Segment the paper (parser mode chooses the path)
        if self.config.parser_mode == "llm" and isinstance(paper, EnrichedPaper):
            snippets = segment_enriched_paper(paper, config=self.config)
        else:
            snippets = segment_paper(paper, config=self.config.segmentation)  # type: ignore[arg-type]
        logger.info(f"Paper segmented into {len(snippets)} snippets")

        return self._verify_snippets(snippets, paper, t0, progress)

    def run_from_snippets(
        self,
        snippets: list[VerificationSnippet],
        paper_id: str,
        title: str = "",
        paper_category: str = "",
        progress: Optional[Progress] = None,
    ) -> PaperPrediction:
        """Run verification on pre-made snippets (bypasses parsing + segmentation).

        Useful when the caller has already produced snippets from the LLM parser
        or another custom source.

        Args:
            snippets: Pre-made VerificationSnippets.
            paper_id: Paper identifier.
            title: Paper title (for prediction metadata).
            paper_category: Paper category (for prediction metadata).
            progress: Optional Rich progress bar.

        Returns:
            PaperPrediction with all findings aggregated.
        """
        logger.info(f"Verifying {len(snippets)} pre-made snippets for {paper_id}")
        t0 = time.monotonic()

        # Build a lightweight NormalizedPaper for prediction metadata
        paper = NormalizedPaper(
            paper_id=paper_id,
            title=title,
            paper_category=paper_category,
        )
        return self._verify_snippets(snippets, paper, t0, progress)

    def _verify_snippets(
        self,
        snippets: list[VerificationSnippet],
        paper: NormalizedPaper | EnrichedPaper,
        t0: float,
        progress: Optional[Progress] = None,
    ) -> PaperPrediction:
        """Run verification on a list of snippets (shared by all paths)."""
        # Pre-instantiate every verifier we'll need on this (single) thread so
        # the verifier cache is read-only during fan-out.
        for snippet in snippets:
            self._get_verifier(self._route_snippet(snippet))

        task_id = None
        if progress:
            task_id = progress.add_task(
                f"[cyan]Verifying {paper.paper_id}...",
                total=len(snippets),
            )

        num_workers = max(1, self.config.llm.num_workers)
        results: list[BaseVerificationResult] = []

        if num_workers == 1:
            # Sequential path — deterministic, for debugging / reproducibility.
            for snippet in snippets:
                results.append(self._verify_one(snippet))
                if progress and task_id is not None:
                    progress.update(task_id, advance=1)
        else:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(self._verify_one, snippet): snippet
                    for snippet in snippets
                }
                for future in as_completed(futures):
                    results.append(future.result())
                    if progress and task_id is not None:
                        progress.update(task_id, advance=1)

        if progress and task_id is not None:
            progress.remove_task(task_id)

        # Tally verifier usage from the collected results.
        verifier_usage: dict[str, int] = {}
        for result in results:
            verifier_usage[result.verifier_name] = (
                verifier_usage.get(result.verifier_name, 0) + 1
            )

        # Step 3: Aggregate findings
        predicted_errors = self._aggregate_findings(results, paper)

        elapsed = time.monotonic() - t0

        prediction = PaperPrediction(
            paper_id=paper.paper_id,
            title=paper.title,
            paper_category=paper.paper_category,
            predicted_errors=predicted_errors,
            total_snippets=len(snippets),
            snippets_verified=len(results),
            errors_detected=len(predicted_errors),
            verifier_usage=verifier_usage,
            raw_results=[r.model_dump() for r in results],
        )

        logger.info(
            f"Paper {paper.paper_id}: {len(predicted_errors)} errors detected "
            f"from {len(results)} snippets in {elapsed:.1f}s"
        )

        return prediction

    def _verify_one(self, snippet: VerificationSnippet) -> BaseVerificationResult:
        """Route and verify a single snippet, capturing failures as SKIPPED.

        Safe to call concurrently from multiple threads: it only reads the
        (pre-populated) verifier cache and the verifiers are stateless.
        """
        verifier_name = self._route_snippet(snippet)
        verifier = self._get_verifier(verifier_name)
        try:
            return verifier.verify(snippet)
        except Exception as exc:
            logger.error(
                f"Verifier {verifier_name} failed on {snippet.snippet_id}: {exc}"
            )
            return BaseVerificationResult(
                snippet_id=snippet.snippet_id,
                verifier_name=verifier_name,
                status="SKIPPED",
                reasoning=f"Verification error: {exc}",
            )

    def _route_snippet(self, snippet: VerificationSnippet) -> str:
        """Resolve the verifier name for a snippet, honouring llm_only_mode
        and LLM-parser-assigned routes.

        Priority:
        1. ``config.llm_only_mode`` — ALL snippets go to ``llm_only``.
        2. ``snippet.verifier_route`` — set by the LLM parser; resolved through
           the triage route map (e.g., "math" → "math_equation").
        3. Type-based routing via ``select_verifier_name`` (the default).
        """
        if self.config.llm_only_mode is not None:
            return "llm_only"

        # LLM parser assigned an explicit route (e.g., "math", "text", "statistical")
        if snippet.verifier_route:
            resolved = resolve_route_to_verifier(
                snippet.verifier_route, snippet, self.config
            )
            if resolved:
                return resolved

        return select_verifier_name(snippet, self.config)

    def _get_verifier(self, name: str) -> BaseVerifier:
        """Get or create a verifier instance by name."""
        if name not in self._verifier_instances:
            verifier_cls = self.registry.get(name)
            self._verifier_instances[name] = verifier_cls(config=self.config)
        return self._verifier_instances[name]

    def _aggregate_findings(
        self,
        results: list[BaseVerificationResult],
        paper: NormalizedPaper,
    ) -> list[PredictedError]:
        """Aggregate individual verifier results into paper-level predictions.

        Filters out low-confidence results and consolidates related findings.

        Args:
            results: All verifier results for the paper.
            paper: The parsed paper (for context).

        Returns:
            List of PredictedError objects.
        """
        predictions: list[PredictedError] = []

        for result in results:
            # Skip non-errors
            if not result.error_detected:
                continue

            # Apply confidence thresholds per verifier
            verifier_config = self.config.verifiers.get(
                result.verifier_name,
                self.config.verifiers.get("text"),
            )
            threshold = verifier_config.confidence_threshold if verifier_config else 0.5

            if result.confidence < threshold:
                logger.debug(
                    f"Skipping low-confidence finding: {result.snippet_id} "
                    f"(confidence={result.confidence:.2f} < {threshold})"
                )
                continue

            predictions.append(PredictedError(
                error_category=result.predicted_error_category or "Unknown",
                error_location=self._infer_location(result),
                confidence=result.confidence,
                supporting_evidence=result.reasoning,
                verifier_name=result.verifier_name,
                snippet_id=result.snippet_id,
            ))

        # Sort by confidence descending
        predictions.sort(key=lambda p: p.confidence, reverse=True)

        return predictions

    @staticmethod
    def _infer_location(result: BaseVerificationResult) -> str:
        """Infer a human-readable location from the result context."""
        # The snippet_id encodes the paper structure.
        # Try to extract usable location info.
        parts = result.snippet_id.split("_", 2)
        if len(parts) >= 3:
            return parts[2] if len(parts) == 3 else "_".join(parts[2:])
        return result.snippet_id
