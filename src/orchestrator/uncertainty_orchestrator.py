"""Uncertainty-driven orchestration.

An alternative to the exhaustive, type-routed orchestrator. The key shift in
question:

    not  "which verifier is best for this section?"
    but  "where is uncertainty concentrated?"

Pipeline:

    Paper
      ↓ segment
    Snippets
      ↓ triage (general verifier, one cheap call each)
    Uncertainty map        Introduction   0.03
                           Related Work   0.07
                           Methods        0.45
                           Equation Chain 0.78
                           Results        0.18
      ↓ select high-uncertainty nodes (threshold + optional budget)
    Specialized verifiers run ONLY where uncertainty is high
      ↓ math → Equation Chain, stat → Methods, ...
    Aggregate findings

Routing emerges from expected error density rather than document structure,
which directly attacks the sparse-error problem: most sections are correct, so
spending a specialist (and risking a false positive) on every one is wasteful.

This is the single-paper realization of the broader "Adaptive Mixture of
Verifiers" idea: the triage stage is the router predicting verifier utility per
node, and the specialists operate on the high-utility nodes it surfaces.
"""

from __future__ import annotations

import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from loguru import logger
from rich.progress import Progress

from src.config import PipelineConfig
from src.models import (
    BaseVerificationResult,
    EnrichedPaper,
    NormalizedPaper,
    PaperPrediction,
    TriageResult,
    VerificationSnippet,
    VerificationStatus,
)
from src.orchestrator.orchestrator import VerificationOrchestrator
from src.orchestrator.router import resolve_route_to_verifier, select_verifier_name
from src.parser.enriched_segmenter import segment_enriched_paper
from src.segmentation.segmenter import segment_paper
from src.verifiers.registry import VerifierRegistry
from src.verifiers.triage_verifier import TriageVerifier


class UncertaintyOrchestrator(VerificationOrchestrator):
    """Triage-first orchestrator that routes specialists by uncertainty.

    Reuses the parent's verifier cache, aggregation, and location inference;
    overrides ``run`` to insert the triage → uncertainty-map → adaptive-dispatch
    stages ahead of specialist verification.
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        registry: Optional[VerifierRegistry] = None,
    ) -> None:
        super().__init__(config=config, registry=registry)

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------
    def run(
        self,
        paper: NormalizedPaper | EnrichedPaper,
        progress: Optional[Progress] = None,
    ) -> PaperPrediction:
        logger.info(f"[uncertainty] Starting verification of paper: {paper.paper_id}")
        t0 = time.monotonic()

        # Step 1: Segment (parser mode chooses the path)
        if self.config.parser_mode == "llm" and isinstance(paper, EnrichedPaper):
            snippets = segment_enriched_paper(paper, config=self.config)
        else:
            snippets = segment_paper(paper, config=self.config.segmentation)  # type: ignore[arg-type]
        logger.info(f"[uncertainty] Paper segmented into {len(snippets)} snippets")

        # Step 2: Triage pass → uncertainty map
        triage_results = self._triage_all(snippets, progress, paper.paper_id)
        triage_by_id = {t.snippet_id: t for t in triage_results}

        # Step 3: Select the high-uncertainty nodes (threshold, then budget cap)
        selected = self._select_snippets(snippets, triage_by_id)
        logger.info(
            f"[uncertainty] {len(selected)}/{len(snippets)} snippets exceed "
            f"threshold {self.config.uncertainty_threshold:.2f} "
            f"(budget={self.config.uncertainty_budget})"
        )
        self._log_uncertainty_map(triage_results, {s.snippet_id for s in selected})

        # Step 4: Run specialists ONLY on selected snippets
        results = self._verify_selected(selected, triage_by_id, progress, paper.paper_id)

        # Unselected snippets are accepted as low-risk (no specialist call). We
        # still record them so coverage is auditable; confidence is 1 - uncertainty.
        verified_ids = {r.snippet_id for r in results}
        for snippet in snippets:
            if snippet.snippet_id in verified_ids:
                continue
            t = triage_by_id.get(snippet.snippet_id)
            unc = t.uncertainty if t else 0.0
            results.append(
                BaseVerificationResult(
                    snippet_id=snippet.snippet_id,
                    verifier_name="triage",
                    status=VerificationStatus.NO_ERROR,
                    error_detected=False,
                    confidence=max(0.0, 1.0 - unc),
                    reasoning=(
                        f"Triaged low-risk (uncertainty={unc:.2f} < "
                        f"{self.config.uncertainty_threshold:.2f}); no specialist run."
                    ),
                )
            )

        # Step 5: Aggregate (reuses parent thresholding/consolidation)
        predicted_errors = self._aggregate_findings(results, paper)

        verifier_usage: dict[str, int] = defaultdict(int)
        for r in results:
            verifier_usage[r.verifier_name] += 1
        verifier_usage["triage"] = max(verifier_usage.get("triage", 0), len(triage_results))

        elapsed = time.monotonic() - t0
        prediction = PaperPrediction(
            paper_id=paper.paper_id,
            title=paper.title,
            paper_category=paper.paper_category,
            predicted_errors=predicted_errors,
            total_snippets=len(snippets),
            snippets_verified=len(selected),
            errors_detected=len(predicted_errors),
            verifier_usage=dict(verifier_usage),
            raw_results=[r.model_dump() for r in results],
            uncertainty_map=[t.model_dump() for t in triage_results],
        )

        logger.info(
            f"[uncertainty] Paper {paper.paper_id}: {len(predicted_errors)} errors "
            f"from {len(selected)} specialist checks "
            f"(triaged {len(snippets)} snippets) in {elapsed:.1f}s"
        )
        return prediction

    # ------------------------------------------------------------------
    # Stages
    # ------------------------------------------------------------------
    def _triage_all(
        self,
        snippets: list[VerificationSnippet],
        progress: Optional[Progress],
        paper_id: str,
    ) -> list[TriageResult]:
        """Run the general triage pass over every snippet (concurrently)."""
        triage: TriageVerifier = self._get_verifier("triage")  # type: ignore[assignment]

        task_id = None
        if progress:
            task_id = progress.add_task(
                f"[magenta]Triaging {paper_id}...", total=len(snippets)
            )

        results: list[TriageResult] = []
        num_workers = max(1, self.config.llm.num_workers)

        def _one(snippet: VerificationSnippet) -> TriageResult:
            try:
                return triage.triage(snippet)
            except Exception as exc:  # defensive; triage() already fails open
                logger.error(f"Triage crashed on {snippet.snippet_id}: {exc}")
                return TriageResult(
                    snippet_id=snippet.snippet_id,
                    snippet_type=snippet.snippet_type.value,
                    location=snippet.location,
                    uncertainty=max(self.config.uncertainty_threshold, 0.5),
                    suggested_route="text",
                    reason=f"Triage crash (failing open): {exc}",
                )

        if num_workers == 1:
            for snippet in snippets:
                results.append(_one(snippet))
                if progress and task_id is not None:
                    progress.update(task_id, advance=1)
        else:
            with ThreadPoolExecutor(max_workers=num_workers) as ex:
                futures = {ex.submit(_one, s): s for s in snippets}
                for fut in as_completed(futures):
                    results.append(fut.result())
                    if progress and task_id is not None:
                        progress.update(task_id, advance=1)

        if progress and task_id is not None:
            progress.remove_task(task_id)

        # Preserve original snippet order for stable maps/logs.
        order = {s.snippet_id: i for i, s in enumerate(snippets)}
        results.sort(key=lambda t: order.get(t.snippet_id, 0))
        return results

    def _select_snippets(
        self,
        snippets: list[VerificationSnippet],
        triage_by_id: dict[str, TriageResult],
    ) -> list[VerificationSnippet]:
        """Pick snippets above the uncertainty threshold, capped by budget."""
        threshold = self.config.uncertainty_threshold
        above = [
            s for s in snippets
            if triage_by_id.get(s.snippet_id, TriageResult(snippet_id=s.snippet_id)).uncertainty
            >= threshold
        ]
        budget = self.config.uncertainty_budget
        if budget is not None and len(above) > budget:
            above.sort(
                key=lambda s: triage_by_id[s.snippet_id].uncertainty,
                reverse=True,
            )
            dropped = above[budget:]
            above = above[:budget]
            logger.info(
                f"[uncertainty] Budget {budget} exceeded; deferring "
                f"{len(dropped)} lower-uncertainty snippets: "
                f"{[s.snippet_id for s in dropped][:10]}"
            )
        return above

    def _verify_selected(
        self,
        selected: list[VerificationSnippet],
        triage_by_id: dict[str, TriageResult],
        progress: Optional[Progress],
        paper_id: str,
    ) -> list[BaseVerificationResult]:
        """Route each selected snippet to a specialist and verify concurrently."""
        if not selected:
            return []

        # Resolve routes and pre-instantiate the specialists we'll need (so the
        # verifier cache stays read-only during fan-out).
        plan: list[tuple[VerificationSnippet, str]] = []
        for snippet in selected:
            t = triage_by_id.get(snippet.snippet_id)
            route = t.suggested_route if t else "text"
            verifier_name = self._resolve_specialist(route, snippet)
            if t is not None:
                t.selected = True
                t.routed_to = verifier_name
            if verifier_name is None:
                continue
            self._get_verifier(verifier_name)
            plan.append((snippet, verifier_name))

        task_id = None
        if progress:
            task_id = progress.add_task(
                f"[cyan]Specialists on {paper_id}...", total=len(plan)
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

    def _resolve_specialist(
        self,
        route: str,
        snippet: VerificationSnippet,
    ) -> Optional[str]:
        """Resolve a route to a verifier that can actually handle the snippet.

        Order: llm_only_mode → triage route → type-based fallback → None.
        """
        # When llm_only_mode is active, bypass all specialists for the single
        # LLM-only verifier (same for every snippet type).
        if self.config.llm_only_mode is not None:
            return "llm_only"

        name = resolve_route_to_verifier(route, snippet, self.config)
        candidates = []
        if name:
            candidates.append(name)
        type_based = select_verifier_name(snippet, self.config)
        if type_based not in candidates:
            candidates.append(type_based)

        for cand in candidates:
            try:
                verifier = self._get_verifier(cand)
            except KeyError:
                continue
            if verifier.can_verify(snippet):
                return cand

        logger.debug(
            f"No usable specialist for {snippet.snippet_id} "
            f"(route='{route}', candidates={candidates}); accepting as low-risk."
        )
        return None

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    @staticmethod
    def _region_of(triage: TriageResult) -> str:
        """Group snippets into human-readable regions for the uncertainty map."""
        sid = triage.snippet_id
        st = triage.snippet_type
        if st == "EQUATION":
            return "Equation Chain"
        if st in {"FIGURE", "TABLE"}:
            return "Figures / Tables"
        if st in {"THEOREM", "LEMMA", "PROPOSITION", "COROLLARY"}:
            return "Theorems / Proofs"
        # Sections: bucket by the sec_<n> token if present.
        for part in sid.split("_"):
            if part.isdigit():
                return f"Section {part}"
        if "_sec_" in sid:
            return "Sections"
        return "Other"

    def _log_uncertainty_map(
        self,
        triage_results: list[TriageResult],
        selected_ids: set[str],
    ) -> None:
        """Emit the region-level uncertainty map (the table in the docstring)."""
        regions: dict[str, list[float]] = defaultdict(list)
        for t in triage_results:
            regions[self._region_of(t)].append(t.uncertainty)

        logger.info("[uncertainty] ── Uncertainty map (mean / max per region) ──")
        for region, scores in sorted(
            regions.items(), key=lambda kv: max(kv[1]), reverse=True
        ):
            mean = sum(scores) / len(scores)
            logger.info(
                f"[uncertainty]   {region:<22} mean={mean:0.2f}  "
                f"max={max(scores):0.2f}  n={len(scores)}"
            )

        # Show the specific high-uncertainty nodes that earned a specialist.
        escalated = [t for t in triage_results if t.snippet_id in selected_ids]
        escalated.sort(key=lambda t: t.uncertainty, reverse=True)
        if escalated:
            logger.info("[uncertainty] ── Escalated to specialists ──")
            for t in escalated[:20]:
                logger.info(
                    f"[uncertainty]   {t.snippet_id:<28} u={t.uncertainty:0.2f} "
                    f"→ {t.routed_to or t.suggested_route}  ({t.reason[:60]})"
                )
