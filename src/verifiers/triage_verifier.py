"""General triage verifier — the "where is uncertainty concentrated?" pass.

This is the first stage of uncertainty-driven orchestration. Instead of asking
"which verifier fits this section's *type*?", the triage verifier asks "how
likely is this snippet to contain a real, correction-worthy error, and which
specialist should look closer?". Its output is an *uncertainty map* over the
paper; the orchestrator then routes expensive specialized verifiers only to the
high-uncertainty nodes (see UncertaintyOrchestrator).

It deliberately makes ONE cheap call per snippet and never emits a final
error decision — that is the specialists' job. This keeps the triage fast and
keeps error decisions with the tools designed to make them (SymPy for
equations, the strict text reviewer for prose, etc.).
"""

from __future__ import annotations

import time

from loguru import logger

from src.models import TriageResult, VerificationSnippet
from src.verifiers.base import BaseVerifier
from src.utils.llm import parse_json_response


TRIAGE_SYSTEM_PROMPT = """You are the triage stage of a scientific-paper verification pipeline. You are shown ONE excerpt (a section, paragraph, equation, theorem, figure caption, or table). You do NOT decide whether it actually contains an error. Your only job is to estimate WHERE error-checking effort should be spent.

Output two things:

1. `uncertainty` — a number in [0, 1]: your estimate of the probability that this excerpt contains a *correction- or retraction-worthy* error (a wrong derivation/result, an internal contradiction, an invalid proof step, a misreported statistic, a duplicated/mismatched figure, etc.). Calibrate:
   - 0.00–0.15: routine, definitional, or boilerplate content (intro prose, standard definitions, acknowledgements, notation setup). MOST excerpts are here.
   - 0.15–0.40: substantive content with some moving parts but nothing that stands out.
   - 0.40–0.70: a non-trivial derivation, a quantitative claim, a load-bearing proof step, or a statement that *could* be wrong and is worth a specialist's time.
   - 0.70–1.00: something looks off, surprising, internally tense, or makes a strong/atypical quantitative or logical claim.

2. `route` — which specialist should examine it IF escalated. Choose ONE:
   - "math": an equation, derivation, or symbolic identity to check algebraically.
   - "proof": a theorem/lemma/proposition statement or proof to check for logical gaps.
   - "statistics": reported numbers, statistics, p-values, percentages, or quantitative results.
   - "citation": claims about prior work, attributions, or references.
   - "vision": a figure or table (visual content).
   - "text": general prose consistency / factual claims (the default).
   - "none": clearly routine; no specialist needed.

Be discriminating: a typical paper has ZERO or ONE real error, so most excerpts deserve LOW uncertainty. Reserve high scores for genuinely suspicious or high-stakes content.

## Output Format

Return ONLY a JSON object with exactly these fields:
```json
{"uncertainty": 0.0, "route": "text", "reason": "one short sentence"}
```
"""


class TriageVerifier(BaseVerifier):
    """Estimates per-snippet error likelihood for uncertainty-driven routing."""

    name: str = "triage"

    # Types that are inherently visual — triage defaults their route.
    _VISUAL_TYPES = {"FIGURE", "TABLE"}

    def verify(self, snippet: VerificationSnippet):  # pragma: no cover - thin shim
        """BaseVerifier contract shim. Use :meth:`triage` for the real output."""
        raise NotImplementedError(
            "TriageVerifier produces TriageResult via triage(); it is driven by "
            "UncertaintyOrchestrator, not the standard verify() path."
        )

    def can_verify(self, snippet: VerificationSnippet) -> bool:
        return self.verifier_config.enabled and bool(snippet.content.strip() or snippet.metadata.get("latex"))

    def triage(self, snippet: VerificationSnippet) -> TriageResult:
        """Score a single snippet's error likelihood and suggested route."""
        start_time = time.monotonic()

        base = TriageResult(
            snippet_id=snippet.snippet_id,
            snippet_type=snippet.snippet_type.value,
            location=snippet.location,
        )

        if not self.can_verify(snippet):
            base.uncertainty = 0.0
            base.suggested_route = "none"
            base.reason = "Empty or unverifiable snippet."
            base.execution_time_ms = (time.monotonic() - start_time) * 1000
            return base

        prompt = self._build_prompt(snippet)
        try:
            parsed = parse_json_response(
                self._call_llm(prompt=prompt, system_prompt=TRIAGE_SYSTEM_PROMPT)
            )
            base.uncertainty = self._clamp(parsed.get("uncertainty", 0.0))
            base.suggested_route = str(parsed.get("route", "text")).strip().lower() or "text"
            base.reason = str(parsed.get("reason", ""))[:300]
        except Exception as exc:
            # On triage failure, fail OPEN: assume moderate uncertainty so the
            # snippet still gets a specialist look rather than being silently
            # dropped. Routing falls back to type-based selection downstream.
            logger.warning(f"Triage failed for {snippet.snippet_id}: {exc}")
            base.uncertainty = max(self.config.uncertainty_threshold, 0.5)
            base.suggested_route = self._default_route(snippet)
            base.reason = f"Triage error (failing open): {exc}"

        base.execution_time_ms = (time.monotonic() - start_time) * 1000
        return base

    def _default_route(self, snippet: VerificationSnippet) -> str:
        t = snippet.snippet_type.value
        if t == "EQUATION":
            return "math"
        if t in self._VISUAL_TYPES:
            return "vision"
        if t in {"THEOREM", "LEMMA", "PROPOSITION", "COROLLARY"}:
            return "proof"
        return "text"

    @staticmethod
    def _clamp(value) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.0

    def _build_prompt(self, snippet: VerificationSnippet) -> str:
        latex = snippet.metadata.get("latex")
        body = latex if latex else snippet.content
        parts = [
            f"Excerpt type: {snippet.snippet_type.value}",
            f"Location: {snippet.location}",
            "",
            "Content:",
            body[:2500],
            "",
            "Return ONLY the JSON triage object.",
        ]
        return "\n".join(parts)
