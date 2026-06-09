"""Global-Read Orchestrator — baseline-as-triage with context-enriched specialists.

Pipeline:

    Paper
      ↓ Global Read (one LLM call, whole paper)
    Structured Context Map:
      · paper_claims      — every verifiable claim with location & type
      · suspicious_regions — regions warranting deeper specialist checks
      · errors            — high-confidence errors the reader caught directly
      ↓ Segment (same as exhaustive)
    Snippets
      ↓ Enrich: attach relevant global context to each snippet
      ↓ Route: specialists run on flagged snippets with full context
      ↓ Cross-Consistency: one final pass checking for contradictions
      ↓ Aggregate: merge global-read errors + specialist findings

Key difference from the existing modes:
- ``exhaustive`` routes every snippet to a specialist by type, but each specialist
  sees ONLY its own snippet — no whole-paper context. This is why it underperforms
  the single-call baseline on cross-snippet errors.
- ``uncertainty`` adds a per-snippet triage pass, but the triage ALSO sees only
  its own snippet — same fragmentation problem.
- ``global_read`` fixes the root cause: the first stage reads the WHOLE paper and
  produces a structured context map. Every downstream specialist inherits this
  global understanding, so it can catch cross-snippet contradictions, compare
  claims across sections, and verify proof logic with full context.
"""

from __future__ import annotations

import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from loguru import logger
from rich.progress import Progress

from src.baseline.global_reader import (
    GLOBAL_READ_SYSTEM_PROMPT,
    GlobalReader,
    GlobalReadResult,
    PaperClaim,
    SuspiciousRegion,
)
from src.config import PipelineConfig
from src.models import (
    BaseVerificationResult,
    EnrichedPaper,
    NormalizedPaper,
    PaperPrediction,
    PredictedError,
    VerificationSnippet,
    VerificationStatus,
)
from src.orchestrator.orchestrator import VerificationOrchestrator
from src.orchestrator.router import resolve_route_to_verifier, select_verifier_name
from src.parser.enriched_segmenter import segment_enriched_paper
from src.parser.location_parser import fuzzy_match_locations
from src.segmentation.segmenter import segment_paper
from src.utils.llm import llm_call, parse_json_response
from src.verifiers.registry import VerifierRegistry


# ---------------------------------------------------------------------------
# Cross-consistency system prompt
# ---------------------------------------------------------------------------

CROSS_CONSISTENCY_SYSTEM_PROMPT = """You are a scientific-integrity reviewer performing a CROSS-CONSISTENCY CHECK on a paper that has already been analyzed by automated verifiers. You are given:

1. The paper's CLAIMS MAP — every verifiable claim the paper makes, with locations.
2. All FINDINGS from individual verifiers (specialist checks on specific snippets).
3. The GLOBAL READ's suspicious regions and overall assessment.

Your job: identify errors that ONLY become visible when looking ACROSS multiple findings or claims. Individual verifiers each saw only one snippet; you see the whole picture.

## What to check

1. **Cross-snippet contradictions**: Does claim A (from one section) contradict claim B (from another)?
2. **Finding synthesis**: Do two or more individual findings, when combined, reveal a larger problem?
3. **Unverified high-stakes claims**: Are there high-confidence paper claims that NO verifier examined? If so, flag the most concerning unverified claims.
4. **Pattern recognition**: Do multiple suspicious regions point to the same underlying error?
5. **Missing connections**: Is there a finding that should have been escalated to a different specialist?

## Output Format

Return ONLY a JSON object:
```json
{
  "cross_errors": [
    {
      "error_category": "Data Inconsistency (text-text)",
      "error_location": "Abstract vs Section 3.1",
      "confidence": 0.85,
      "supporting_evidence": "The Abstract claims X but Section 3.1 states Y, which are incompatible because..."
    }
  ],
  "synthesis_notes": "Brief summary of how findings relate to each other and what the most important unresolved question is.",
  "unverified_concerns": [
    {
      "location": "...",
      "concern": "..."
    }
  ]
}
```

Be conservative: only flag cross-errors when you are genuinely confident there is a contradiction. A typical paper has 0-1 cross-errors.
"""


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class GlobalReadOrchestrator(VerificationOrchestrator):
    """Baseline-as-triage orchestrator: whole-paper global read → enriched specialists.

    Reuses the parent's verifier cache, aggregation, and location inference.
    Overrides ``run`` to insert the global-read → enrichment → cross-consistency
    stages ahead of specialist verification.
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        registry: Optional[VerifierRegistry] = None,
    ) -> None:
        super().__init__(config=config, registry=registry)
        self._global_reader: Optional[GlobalReader] = None

    @property
    def global_reader(self) -> GlobalReader:
        if self._global_reader is None:
            self._global_reader = GlobalReader(
                config=self.config,
                max_input_chars=getattr(self.config, "global_read_max_chars", 60000),
            )
        return self._global_reader

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------
    def run(
        self,
        paper: NormalizedPaper | EnrichedPaper,
        progress: Optional[Progress] = None,
    ) -> PaperPrediction:
        logger.info(f"[global_read] Starting verification of paper: {paper.paper_id}")
        t0 = time.monotonic()

        # Stage 1: Global Read — structured whole-paper analysis
        logger.info(f"[global_read] Stage 1: Global Read on {paper.paper_id} ...")
        global_result = self.global_reader.run(paper)  # type: ignore[arg-type]
        logger.info(
            f"[global_read] Global Read complete: {len(global_result.errors)} error(s), "
            f"{len(global_result.paper_claims)} claim(s), "
            f"{len(global_result.suspicious_regions)} suspicious region(s)"
        )

        # Stage 2: Segment the paper
        if self.config.parser_mode == "llm" and isinstance(paper, EnrichedPaper):
            snippets = segment_enriched_paper(paper, config=self.config)
        else:
            snippets = segment_paper(paper, config=self.config.segmentation)  # type: ignore[arg-type]
        logger.info(f"[global_read] Paper segmented into {len(snippets)} snippets")

        # Stage 3: Enrich snippets with global context and determine which to verify
        enriched_plan = self._build_enriched_plan(snippets, global_result)
        selected = [s for s, _ in enriched_plan if s is not None]
        skipped_count = len(snippets) - len(selected)
        logger.info(
            f"[global_read] {len(selected)}/{len(snippets)} snippets selected for "
            f"specialist verification ({skipped_count} low-risk)"
        )
        self._log_selection(global_result, enriched_plan)

        # Stage 4: Run specialists on selected snippets (with global context)
        results = self._verify_enriched(enriched_plan, progress, paper.paper_id)

        # Unselected snippets are accepted as low-risk.
        verified_ids = {r.snippet_id for r in results}
        for snippet in snippets:
            if snippet.snippet_id in verified_ids:
                continue
            relevance = self._snippet_relevance(snippet, global_result)
            results.append(
                BaseVerificationResult(
                    snippet_id=snippet.snippet_id,
                    verifier_name="global_read_triage",
                    status=VerificationStatus.NO_ERROR,
                    error_detected=False,
                    confidence=max(0.0, 1.0 - relevance),
                    reasoning=(
                        f"Global Read did not flag this region for specialist review "
                        f"(max relevance={relevance:.2f})."
                    ),
                )
            )

        # Stage 5: Cross-consistency pass
        cross_errors = self._cross_consistency_check(global_result, results, paper)
        logger.info(
            f"[global_read] Cross-consistency: {len(cross_errors)} cross-cutting error(s) found"
        )

        # Stage 6: Aggregate — merge global-read errors + specialist findings + cross-errors
        all_errors = self._merge_all_errors(
            global_read_errors=global_result.errors,
            specialist_results=results,
            cross_errors=cross_errors,
            paper=paper,
        )

        verifier_usage: dict[str, int] = defaultdict(int)
        for r in results:
            verifier_usage[r.verifier_name] += 1
        verifier_usage["global_read"] = 1
        verifier_usage["cross_consistency"] = 1

        elapsed = time.monotonic() - t0
        prediction = PaperPrediction(
            paper_id=paper.paper_id,
            title=paper.title,
            paper_category=paper.paper_category,
            predicted_errors=all_errors,
            total_snippets=len(snippets),
            snippets_verified=len(selected),
            errors_detected=len(all_errors),
            verifier_usage=dict(verifier_usage),
            raw_results=[r.model_dump() for r in results],
            # Store the global read result as uncertainty_map for backward compat
            uncertainty_map=[
                {
                    "snippet_id": f"global_read_{paper.paper_id}",
                    "snippet_type": "GLOBAL",
                    "location": "whole paper",
                    "uncertainty": max(
                        [r.get("confidence", 0.0) for r in global_result.errors]
                        + [s.uncertainty for s in global_result.suspicious_regions]
                        + [0.0]
                    ),
                    "suggested_route": "global_read",
                    "reason": global_result.overall_assessment[:300],
                    "selected": True,
                    "routed_to": "global_read",
                }
            ],
        )

        logger.info(
            f"[global_read] Paper {paper.paper_id}: {len(all_errors)} total error(s) "
            f"({len(global_result.errors)} from global read, "
            f"{len(cross_errors)} cross-consistency, "
            f"rest from specialists) from {len(selected)} specialist checks "
            f"in {elapsed:.1f}s"
        )
        return prediction

    # ------------------------------------------------------------------
    # Stage 3: Enrichment & selection
    # ------------------------------------------------------------------
    def _build_enriched_plan(
        self,
        snippets: list[VerificationSnippet],
        global_result: GlobalReadResult,
    ) -> list[tuple[VerificationSnippet, str]]:
        """Decide which snippets to verify and with which specialist.

        A snippet is selected for verification only if its computed relevance
        to the global read's suspicious_regions (or errors) is ≥ the threshold.
        Paper claims are used for context enrichment, NOT for selection.

        Each selected snippet is enriched with the global context before verification.

        Returns:
            List of (snippet, verifier_name) tuples for snippets to verify.
        """
        plan: list[tuple[VerificationSnippet, str]] = []

        for snippet in snippets:
            relevance = self._snippet_relevance(snippet, global_result)

            if relevance >= self.config.uncertainty_threshold:
                verifier_name = self._resolve_specialist(snippet, global_result)
                if verifier_name:
                    enriched = self._enrich_snippet(snippet, global_result)
                    plan.append((enriched, verifier_name))

        return plan

    @staticmethod
    def _snippet_relevance(
        snippet: VerificationSnippet,
        global_result: GlobalReadResult,
    ) -> float:
        """Compute how relevant a snippet is to the global read's concerns.

        Returns a score in [0, 1] where higher = more likely to contain an error.

        Selection is driven ONLY by suspicious_regions and global-read errors.
        Paper claims are used solely for context enrichment (see _enrich_snippet),
        NOT for selection — otherwise every snippet matches some claim and the
        mode degenerates into exhaustive verification.
        """
        scores: list[float] = []

        # Check suspicious region overlap — this is the primary triage signal
        for region in global_result.suspicious_regions:
            loc_score = fuzzy_match_locations(snippet.location, region.location)
            if loc_score > 0.3:
                scores.append(region.uncertainty * loc_score)

        # Check if any global-read error references this location
        for error in global_result.errors:
            err_loc = error.get("error_location", "")
            if err_loc:
                loc_score = fuzzy_match_locations(snippet.location, err_loc)
                if loc_score > 0.3:
                    scores.append(0.9 * loc_score)

        if not scores:
            return 0.0
        return max(scores)

    @staticmethod
    def _enrich_snippet(
        snippet: VerificationSnippet,
        global_result: GlobalReadResult,
    ) -> VerificationSnippet:
        """Attach relevant global context to a snippet before verification.

        Returns a NEW snippet (does not mutate the original) with enriched
        content and metadata so the specialist sees the paper-level context.
        """
        # Find the most relevant claims for this snippet
        relevant_claims: list[PaperClaim] = []
        for claim in global_result.paper_claims:
            score = fuzzy_match_locations(snippet.location, claim.location)
            if score > 0.2:
                relevant_claims.append(claim)

        # Find overlapping suspicious regions
        relevant_regions: list[SuspiciousRegion] = []
        for region in global_result.suspicious_regions:
            score = fuzzy_match_locations(snippet.location, region.location)
            if score > 0.2:
                relevant_regions.append(region)

        # Build the context prefix
        context_parts: list[str] = []

        if global_result.overall_assessment:
            context_parts.append(
                "=== PAPER-LEVEL CONTEXT (from Global Read) ===\n"
                f"Overall assessment: {global_result.overall_assessment[:500]}"
            )

        if relevant_regions:
            context_parts.append(
                "\n=== WHY THIS SNIPPET WAS FLAGGED ==="
            )
            for r in relevant_regions[:3]:
                context_parts.append(
                    f"· {r.location}: {r.reason} "
                    f"(error likelihood: {r.uncertainty:.0%})"
                )

        if relevant_claims:
            context_parts.append(
                "\n=== RELATED CLAIMS FROM ELSEWHERE IN THE PAPER ==="
            )
            for c in relevant_claims[:5]:
                context_parts.append(
                    f"· [{c.claim_type}] {c.location}: {c.claim_text[:200]}"
                )

        if global_result.errors:
            context_parts.append(
                "\n=== ERRORS ALREADY FOUND BY GLOBAL READ ==="
            )
            for e in global_result.errors[:3]:
                context_parts.append(
                    f"· {e.get('error_location', '?' )}: "
                    f"{e.get('supporting_evidence', '')[:200]}"
                )

        # Also add a broader claims summary (all claims, compact)
        all_claims_summary = "\n".join(
            f"· [{c.claim_type}] {c.location}: {c.claim_text[:150]}"
            for c in global_result.paper_claims[:20]
        )
        if all_claims_summary:
            context_parts.append(
                "\n=== FULL PAPER CLAIMS MAP (for cross-reference) ===\n"
                f"{all_claims_summary}"
            )

        context_prefix = "\n\n".join(context_parts)
        enriched_content = (
            f"{context_prefix}\n\n"
            f"=== SNIPPET TO VERIFY (location: {snippet.location}) ===\n"
            f"{snippet.content}"
        )

        # Create enriched metadata
        enriched_metadata = dict(snippet.metadata)
        enriched_metadata["global_read_claims_count"] = len(relevant_claims)
        enriched_metadata["global_read_regions_count"] = len(relevant_regions)
        enriched_metadata["global_read_has_errors"] = len(global_result.errors) > 0

        return VerificationSnippet(
            snippet_id=snippet.snippet_id,
            snippet_type=snippet.snippet_type,
            paper_id=snippet.paper_id,
            location=snippet.location,
            content=enriched_content,
            image_path=snippet.image_path,
            location_ref=snippet.location_ref,
            verifier_route=snippet.verifier_route,
            dependency_context=snippet.dependency_context,
            metadata=enriched_metadata,
        )

    def _resolve_specialist(
        self,
        snippet: VerificationSnippet,
        global_result: GlobalReadResult,
    ) -> Optional[str]:
        """Resolve which specialist verifier should handle this snippet.

        Priority:
        1. Explicit verifier_route from LLM parser
        2. Suggested verifier from overlapping suspicious_region
        3. Type-based routing (standard)
        """
        # LLM parser explicit route
        if snippet.verifier_route:
            name = resolve_route_to_verifier(snippet.verifier_route, snippet, self.config)
            if name:
                return name

        # Check if any suspicious region suggests a specific verifier
        for region in global_result.suspicious_regions:
            score = fuzzy_match_locations(snippet.location, region.location)
            if score > 0.4 and region.suggested_verifier:
                route = region.suggested_verifier.lower()
                mapped = self.config.triage_route_map.get(route)
                if mapped:
                    try:
                        verifier = self._get_verifier(mapped)
                        if verifier.can_verify(snippet):
                            return mapped
                    except KeyError:
                        pass

        # Fall back to type-based routing
        return select_verifier_name(snippet, self.config)

    # ------------------------------------------------------------------
    # Stage 4: Specialist verification
    # ------------------------------------------------------------------
    def _verify_enriched(
        self,
        plan: list[tuple[VerificationSnippet, str]],
        progress: Optional[Progress],
        paper_id: str,
    ) -> list[BaseVerificationResult]:
        """Run specialist verifiers on the enriched snippets concurrently."""
        if not plan:
            return []

        # Pre-instantiate all needed verifiers
        for _, verifier_name in plan:
            self._get_verifier(verifier_name)

        task_id = None
        if progress:
            task_id = progress.add_task(
                f"[cyan]Specialists on {paper_id} (global-read guided)...",
                total=len(plan),
            )

        results: list[BaseVerificationResult] = []
        num_workers = max(1, self.config.llm.num_workers)

        def _one(item: tuple[VerificationSnippet, str]) -> BaseVerificationResult:
            snippet, name = item
            verifier = self._get_verifier(name)
            try:
                return verifier.verify(snippet)
            except Exception as exc:
                logger.error(f"Specialist {name} failed on {snippet.snippet_id}: {exc}")
                return BaseVerificationResult(
                    snippet_id=snippet.snippet_id,
                    verifier_name=name,
                    status=VerificationStatus.SKIPPED,
                    reasoning=f"Verification error: {exc}",
                )

        if num_workers == 1:
            for item in plan:
                results.append(_one(item))
                if progress and task_id is not None:
                    progress.update(task_id, advance=1)
        else:
            with ThreadPoolExecutor(max_workers=num_workers) as ex:
                futures = {ex.submit(_one, item): item for item in plan}
                for fut in as_completed(futures):
                    results.append(fut.result())
                    if progress and task_id is not None:
                        progress.update(task_id, advance=1)

        if progress and task_id is not None:
            progress.remove_task(task_id)
        return results

    # ------------------------------------------------------------------
    # Stage 5: Cross-consistency check
    # ------------------------------------------------------------------
    def _cross_consistency_check(
        self,
        global_result: GlobalReadResult,
        specialist_results: list[BaseVerificationResult],
        paper: NormalizedPaper | EnrichedPaper,
    ) -> list[dict]:
        """Run a final LLM pass checking for contradictions across all findings.

        This is where cross-snippet errors surface: individual specialists each
        saw only their snippet (enriched with context), but they couldn't compare
        findings against each other. This pass connects the dots.
        """
        # Only run if we have enough substance to check
        if not global_result.paper_claims and not specialist_results:
            return []

        # Collect specialist findings
        findings_summary: list[str] = []
        for r in specialist_results:
            if r.error_detected or r.status in {
                VerificationStatus.INVALID,
                VerificationStatus.ERROR_DETECTED,
            }:
                findings_summary.append(
                    f"[ERROR] {r.verifier_name} on {r.snippet_id}: "
                    f"{r.reasoning[:200]}"
                )
            elif r.status == VerificationStatus.UNVERIFIABLE:
                findings_summary.append(
                    f"[UNVERIFIABLE] {r.verifier_name} on {r.snippet_id}: "
                    f"{r.reasoning[:200]}"
                )

        # Build claims summary
        claims_text = "\n".join(
            f"· [{c.claim_type}] {c.location}: {c.claim_text[:200]}"
            for c in global_result.paper_claims[:30]
        )

        # Global read errors
        global_errors_text = "\n".join(
            f"· {e.get('error_location', '?')}: {e.get('supporting_evidence', '')[:200]}"
            for e in global_result.errors
        )

        prompt = (
            f"Paper: {paper.title} ({paper.paper_id})\n\n"
            f"=== PAPER CLAIMS MAP ===\n{claims_text}\n\n"
            f"=== GLOBAL READ ERRORS ===\n"
            f"{global_errors_text if global_errors_text else '(none)'}\n\n"
            f"=== SPECIALIST FINDINGS ===\n"
            f"{chr(10).join(findings_summary) if findings_summary else '(none)'}\n\n"
            f"=== SUSPICIOUS REGIONS (not yet verified) ===\n"
            + "\n".join(
                f"· {s.location}: {s.reason}" for s in global_result.suspicious_regions[:10]
            )
            + "\n\n"
            "Check for cross-cutting contradictions, unverified high-stakes claims, "
            "and patterns across findings. Return ONLY the JSON object."
        )

        try:
            response = llm_call(
                prompt=prompt,
                system_prompt=CROSS_CONSISTENCY_SYSTEM_PROMPT,
                config=self.config.llm,
                temperature=0.0,
            )
            parsed = parse_json_response(response)
            return parsed.get("cross_errors", []) or []
        except Exception as exc:
            logger.warning(f"[global_read] Cross-consistency check failed: {exc}")
            return []

    # ------------------------------------------------------------------
    # Stage 6: Merge & aggregate
    # ------------------------------------------------------------------
    def _merge_all_errors(
        self,
        global_read_errors: list[dict],
        specialist_results: list[BaseVerificationResult],
        cross_errors: list[dict],
        paper: NormalizedPaper | EnrichedPaper,
    ) -> list[PredictedError]:
        """Merge errors from all three sources, deduplicating by location.

        Sources (in priority order):
        1. Global Read errors (whole-paper, highest context)
        2. Cross-consistency errors (cross-snippet contradictions)
        3. Specialist findings (individual snippet checks)
        """
        seen_locations: set[str] = set()
        all_errors: list[PredictedError] = []

        # 1. Global Read errors
        for e in global_read_errors:
            loc = str(e.get("error_location", ""))
            norm = self._normalize_location(loc)
            if norm not in seen_locations:
                seen_locations.add(norm)
                all_errors.append(
                    PredictedError(
                        error_category=e.get("error_category", "Unknown"),
                        error_location=loc or "Unknown",
                        confidence=float(e.get("confidence", 0.8)),
                        supporting_evidence=str(e.get("supporting_evidence", ""))[:1000],
                        verifier_name="global_read",
                        snippet_id=f"{paper.paper_id}_global",
                    )
                )

        # 2. Cross-consistency errors
        for e in cross_errors:
            loc = str(e.get("error_location", ""))
            norm = self._normalize_location(loc)
            if norm not in seen_locations:
                seen_locations.add(norm)
                all_errors.append(
                    PredictedError(
                        error_category=e.get("error_category", "Unknown"),
                        error_location=loc or "Unknown",
                        confidence=float(e.get("confidence", 0.8)),
                        supporting_evidence=str(e.get("supporting_evidence", ""))[:1000],
                        verifier_name="cross_consistency",
                        snippet_id=f"{paper.paper_id}_cross",
                    )
                )

        # 3. Specialist findings — reuse parent aggregation logic but with
        # dedup against already-seen locations
        specialist_errors = self._aggregate_findings(specialist_results, paper)
        for pe in specialist_errors:
            norm = self._normalize_location(pe.error_location)
            if norm not in seen_locations:
                seen_locations.add(norm)
                all_errors.append(pe)

        # Sort by confidence descending
        all_errors.sort(key=lambda p: p.confidence, reverse=True)
        return all_errors

    @staticmethod
    def _normalize_location(location: str) -> str:
        """Normalize a location string for dedup comparison."""
        return location.strip().lower().replace(" ", "").replace(".", "").replace(",", "")

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def _log_selection(
        self,
        global_result: GlobalReadResult,
        plan: list[tuple[VerificationSnippet, str]],
    ) -> None:
        """Log which snippets were selected and why."""
        logger.info("[global_read] ── Global Read Summary ──")
        logger.info(
            f"[global_read]   Errors: {len(global_result.errors)}, "
            f"Claims: {len(global_result.paper_claims)}, "
            f"Suspicious: {len(global_result.suspicious_regions)}"
        )

        if global_result.suspicious_regions:
            logger.info("[global_read] ── Suspicious Regions ──")
            for s in sorted(
                global_result.suspicious_regions,
                key=lambda r: r.uncertainty,
                reverse=True,
            )[:15]:
                logger.info(
                    f"[global_read]   {s.location:<30} u={s.uncertainty:.2f} "
                    f"→ {s.suggested_verifier}  ({s.reason[:80]})"
                )

        if plan:
            logger.info(
                f"[global_read] ── {len(plan)} Snippet(s) Selected for Specialists ──"
            )
            for snippet, vname in plan[:20]:
                logger.info(
                    f"[global_read]   {snippet.snippet_id:<30} → {vname}"
                )
