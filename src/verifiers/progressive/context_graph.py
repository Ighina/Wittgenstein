"""Context-graph utilities for the progressive math verification pipeline.

Operates on ``SymbolContext`` instances — adding declarations, assumptions,
definitions, and proof obligations; summarising context for LLM prompts; and
resolving pending obligations against accumulated knowledge.
"""

from __future__ import annotations

import re
from typing import Optional

from src.verifiers.progressive.models import (
    DOMAIN_IMPLICATIONS,
    Assumption,
    DependencyEdge,
    ProofObligation,
    SymbolContext,
    SymbolDeclaration,
    SymbolDomain,
)


# ---------------------------------------------------------------------------
# Symbol assumption builders
# ---------------------------------------------------------------------------


def build_sympy_assumptions(declaration: SymbolDeclaration) -> dict[str, bool]:
    """Convert a ``SymbolDeclaration`` into a SymPy-compatible assumptions dict.

    Uses the domain implication chain so, e.g., ``INTEGER`` also sets
    ``rational=True, real=True, complex=True``.
    """
    assumptions: dict[str, bool] = dict(declaration.sympy_assumptions)
    implied = DOMAIN_IMPLICATIONS.get(declaration.domain, {})
    for k, v in implied.items():
        assumptions.setdefault(k, v)
    return assumptions


def build_sympy_symbol_code(
    symbol: str,
    assumptions: dict[str, bool],
) -> str:
    """Build a ``sympy.Symbol(...)`` constructor call for the given symbol.

    Example output: ``x = Symbol('x', real=True, positive=True)``.
    """
    kwargs_parts = [f"'{symbol}'"]
    for k, v in sorted(assumptions.items()):
        if v:
            kwargs_parts.append(f"{k}=True")
    return f"{symbol} = Symbol({', '.join(kwargs_parts)})"


def build_all_symbol_declarations(ctx: SymbolContext) -> str:
    """Generate a SymPy code block that declares all known symbols with assumptions.

    Returns an empty string if the context has no declarations.
    """
    lines: list[str] = []
    for sym_name, decl in sorted(ctx.declarations.items()):
        assumptions = build_sympy_assumptions(decl)
        lines.append(build_sympy_symbol_code(sym_name, assumptions))
        if decl.is_defined and decl.defined_as:
            lines.append(f"{sym_name} = {decl.defined_as}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Domain inference from LaTeX
# ---------------------------------------------------------------------------

# Heuristic patterns that suggest a domain declaration in LaTeX.
# These are intentionally simple — the LLM does the heavy lifting; these
# patterns provide a deterministic fallback for common cases.
_DOMAIN_PATTERNS: list[tuple[re.Pattern[str], SymbolDomain]] = [
    (re.compile("\\\\in\\s*\\\\mathbb\\{N\\}"), SymbolDomain.NATURAL),
    (re.compile("\\\\in\\s*\\\\mathbb\\{Z\\}"), SymbolDomain.INTEGER),
    (re.compile("\\\\in\\s*\\\\mathbb\\{Q\\}"), SymbolDomain.RATIONAL),
    (re.compile("\\\\in\\s*\\\\mathbb\\{R\\}(?!\\^)"), SymbolDomain.REAL),
    (re.compile("\\\\in\\s*\\\\mathbb\\{R\\}\\^"), SymbolDomain.MATRIX),
    (re.compile("\\\\in\\s*\\\\mathbb\\{C\\}"), SymbolDomain.COMPLEX),
    (re.compile(">\\s*0"), SymbolDomain.POSITIVE_REAL),
    (re.compile("\\\\ge\\s*0"), SymbolDomain.NONNEGATIVE_REAL),
    (re.compile("\\\\neq\\s*0"), SymbolDomain.NONZERO),
]


def infer_domain_from_latex(latex: str) -> Optional[SymbolDomain]:
    """Heuristically detect a domain declaration from a LaTeX string.

    Returns ``None`` when no pattern matches — the caller should fall back
    to LLM-based classification.
    """
    for pattern, domain in _DOMAIN_PATTERNS:
        if pattern.search(latex):
            return domain
    return None


# ---------------------------------------------------------------------------
# Context mutation helpers
# ---------------------------------------------------------------------------


def add_declaration_to_context(
    ctx: SymbolContext, decl: SymbolDeclaration,
) -> SymbolDeclaration:
    """Add a ``SymbolDeclaration`` to the context, merging with any existing entry.

    Returns the declaration as it exists in the context after merging.
    """
    ctx.add_declaration(decl)
    return ctx.declarations[decl.symbol]


def add_assumption_to_context(
    ctx: SymbolContext, assumption: Assumption,
) -> None:
    """Record an assumption monotonically."""
    ctx.add_assumption(assumption)


def add_definition_to_context(
    ctx: SymbolContext, symbol: str, rhs_expr: str, snippet_id: str,
) -> None:
    """Record a definition (e.g. B = Q/N) and mark the symbol as defined."""
    ctx.add_definition(symbol, rhs_expr)
    if symbol in ctx.declarations:
        ctx.declarations[symbol].is_defined = True
        ctx.declarations[symbol].defined_as = rhs_expr
    else:
        decl = SymbolDeclaration(
            symbol=symbol,
            domain=SymbolDomain.UNKNOWN,
            is_defined=True,
            defined_as=rhs_expr,
            source_snippet_id=snippet_id,
        )
        ctx.declarations[symbol] = decl


# ---------------------------------------------------------------------------
# Obligation resolution
# ---------------------------------------------------------------------------


def resolve_obligation(
    obligation: ProofObligation, ctx: SymbolContext,
) -> Optional[bool]:
    """Try to resolve a proof obligation against the accumulated context.

    Returns:
        ``True``  — the obligation is satisfied by the context.
        ``False`` — the context contradicts the obligation.
        ``None``  — cannot determine (insufficient information).
    """
    cond = obligation.condition.lower()

    for dep_sym in obligation.depends_on_symbols:
        decl = ctx.declarations.get(dep_sym)
        if decl is None:
            continue

        assumptions = build_sympy_assumptions(decl)

        # Check common conditions
        if "!=" in cond or "≠" in cond or "neq" in cond or "nonzero" in cond:
            if assumptions.get("nonzero") or assumptions.get("positive"):
                obligation.resolved = True
                obligation.resolution_reason = (
                    f"{dep_sym} is declared {decl.domain.value} → nonzero holds"
                )
                obligation.resolution_layer = "side_condition"
                return True

        if "> 0" in cond or "positive" in cond:
            if assumptions.get("positive"):
                obligation.resolved = True
                obligation.resolution_reason = (
                    f"{dep_sym} is declared positive"
                )
                obligation.resolution_layer = "side_condition"
                return True

        if "real" in cond:
            if assumptions.get("real"):
                obligation.resolved = True
                obligation.resolution_reason = (
                    f"{dep_sym} implies real via domain {decl.domain.value}"
                )
                obligation.resolution_layer = "side_condition"
                return True

        if "integer" in cond:
            if assumptions.get("integer"):
                obligation.resolved = True
                obligation.resolution_reason = (
                    f"{dep_sym} implies integer via domain {decl.domain.value}"
                )
                obligation.resolution_layer = "side_condition"
                return True

    # Unresolved
    return None


def resolve_pending_obligations(ctx: SymbolContext) -> list[ProofObligation]:
    """Walk through pending obligations and resolve any that are now satisfiable.

    Resolved obligations are moved from ``pending_obligations`` to
    ``resolved_obligations``. Returns the list of newly-resolved obligations.
    """
    newly_resolved: list[ProofObligation] = []
    still_pending: list[ProofObligation] = []

    for ob in ctx.pending_obligations:
        result = resolve_obligation(ob, ctx)
        if result is True:
            ob.resolved = True
            ctx.resolved_obligations.append(ob)
            newly_resolved.append(ob)
        elif result is False:
            ob.resolved = False
            ob.resolution_reason = "Contradicted by context"
            ctx.resolved_obligations.append(ob)
            newly_resolved.append(ob)
        else:
            ob.resolved = None
            still_pending.append(ob)

    ctx.pending_obligations = still_pending
    return newly_resolved


# ---------------------------------------------------------------------------
# Context summarisation (for LLM prompts)
# ---------------------------------------------------------------------------


def summarize_context(ctx: SymbolContext, max_entries: int = 20) -> str:
    """Build a compact summary of the accumulated context for LLM prompts.

    The summary is bounded so the prompt doesn't blow up for papers with
    dozens of equations.
    """
    parts: list[str] = []

    # Known symbols
    if ctx.declarations:
        parts.append("### Known symbols")
        for sym, decl in sorted(ctx.declarations.items()):
            extra = []
            if decl.domain != SymbolDomain.UNKNOWN:
                extra.append(f"domain={decl.domain.value}")
            if decl.is_defined and decl.defined_as:
                extra.append(f"defined_as={decl.defined_as}")
            suffix = f"  ({', '.join(extra)})" if extra else ""
            parts.append(f"- {sym}{suffix}")
            if len(parts) - 1 >= max_entries:
                parts.append(f"- ... ({len(ctx.declarations) - max_entries} more symbols)")
                break

    # Definitions
    if ctx.definitions:
        parts.append("\n### Definitions")
        for sym, rhs in list(ctx.definitions.items())[:10]:
            parts.append(f"- {sym} = {rhs}")

    # Assumptions
    if ctx.assumptions:
        parts.append("\n### Assumptions")
        for a in ctx.assumptions[-10:]:  # most recent first
            parts.append(f"- [{a.snippet_id}] {a.description}")

    # Pending proof obligations
    if ctx.pending_obligations:
        parts.append("\n### Pending proof obligations")
        for ob in ctx.pending_obligations[-5:]:
            parts.append(f"- [{ob.obligation_id}] {ob.condition} (depends on {ob.depends_on_symbols})")

    return "\n".join(parts) if parts else "No accumulated context yet."


# ---------------------------------------------------------------------------
# Dependency graph helpers
# ---------------------------------------------------------------------------


def get_transitive_dependencies(
    ctx: SymbolContext, snippet_id: str,
) -> set[str]:
    """Return all snippet IDs reachable from ``snippet_id`` via the dependency graph.

    Uses iterative DFS — simple and sufficient for paper-scale graphs.
    """
    visited: set[str] = set()
    stack: list[str] = [snippet_id]

    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        for edge in ctx.dependency_graph.get(current, []):
            if edge.to_snippet not in visited:
                stack.append(edge.to_snippet)

    visited.discard(snippet_id)  # don't include self
    return visited


def trace_symbol_to_source(
    ctx: SymbolContext, symbol: str,
) -> Optional[str]:
    """Find the snippet ID where a symbol was first declared or defined.

    Returns ``None`` if the symbol is unknown.
    """
    decl = ctx.declarations.get(symbol)
    if decl is None:
        return None
    return decl.source_snippet_id
