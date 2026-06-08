"""Data models for the progressive math verification pipeline.

These models represent the accumulated context (symbol declarations, domain
assumptions, definitions, proof obligations) that builds up as the verifier
reads through a paper's equations in order.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Statement classification
# ---------------------------------------------------------------------------


class StatementClass(str, Enum):
    """Three-way classification of a mathematical statement within a paper."""

    UNCHECKABLE_DECLARATION = "uncheckable_declaration"
    CHECKABLE_CONSTRAINT = "checkable_constraint"
    CHECKABLE_DERIVATION = "checkable_derivation"


class SymbolDomain(str, Enum):
    """Known mathematical domains for declared symbols.

    The order here is significant: integer → rational → real → complex is an
    implication chain.  See ``DOMAIN_IMPLICATIONS`` below.
    """

    REAL = "real"
    COMPLEX = "complex"
    INTEGER = "integer"
    NATURAL = "natural"
    RATIONAL = "rational"
    POSITIVE_REAL = "positive_real"
    NONNEGATIVE_REAL = "nonnegative_real"
    NONZERO = "nonzero"
    MATRIX = "matrix"
    FUNCTION_SPACE = "function_space"
    UNKNOWN = "unknown"


# When a symbol is declared with one of these domains, the listed extra
# predicates also hold.  E.g.  integer  →  rational, real, complex, finite.
DOMAIN_IMPLICATIONS: dict[SymbolDomain, dict[str, bool]] = {
    SymbolDomain.INTEGER: {
        "integer": True, "rational": True, "real": True, "complex": True,
    },
    SymbolDomain.NATURAL: {
        "integer": True, "rational": True, "real": True, "complex": True,
        "nonnegative": True,
    },
    SymbolDomain.RATIONAL: {
        "rational": True, "real": True, "complex": True,
    },
    SymbolDomain.REAL: {
        "real": True, "complex": True,
    },
    SymbolDomain.COMPLEX: {
        "complex": True,
    },
    SymbolDomain.POSITIVE_REAL: {
        "real": True, "complex": True, "positive": True,
    },
    SymbolDomain.NONNEGATIVE_REAL: {
        "real": True, "complex": True, "nonnegative": True,
    },
    SymbolDomain.NONZERO: {
        "nonzero": True, "complex": True,
    },
    SymbolDomain.MATRIX: {},
    SymbolDomain.FUNCTION_SPACE: {},
    SymbolDomain.UNKNOWN: {"complex": True},
}


# ---------------------------------------------------------------------------
# Context nodes
# ---------------------------------------------------------------------------


class SymbolDeclaration(BaseModel):
    """A symbol introduced or constrained in the paper."""

    symbol: str
    domain: SymbolDomain = SymbolDomain.UNKNOWN
    sympy_assumptions: dict[str, bool] = Field(default_factory=dict)
    is_defined: bool = False
    defined_as: Optional[str] = None  # SymPy expression string, e.g. "Q/N"
    source_snippet_id: str = ""
    source_location: str = ""
    first_mentioned_at: int = 0  # equation order index


class Assumption(BaseModel):
    """A recorded assumption that is not a full symbol declaration
    (e.g. "x > 0" after x was already declared real)."""

    snippet_id: str = ""
    latex: str = ""
    description: str = ""
    sympy_code: Optional[str] = None
    symbols: list[str] = Field(default_factory=list)
    assumption_type: str = "domain"  # domain | positivity | definition | relation


class ProofObligation(BaseModel):
    """An unmet side-condition that must hold for a verification to be sound."""

    obligation_id: str = ""
    condition: str = ""  # SymPy-readable condition, e.g. "x != 0"
    condition_latex: Optional[str] = None
    source_snippet_id: str = ""
    depends_on_symbols: list[str] = Field(default_factory=list)
    resolved: Optional[bool] = None
    resolution_reason: Optional[str] = None
    resolution_layer: Optional[str] = None


class DependencyEdge(BaseModel):
    """A directed edge in the dependency graph: snippet A depends on snippet B."""

    from_snippet: str
    to_snippet: str
    relation: str = "uses_symbol"  # uses_symbol | references_definition | proved_by


# ---------------------------------------------------------------------------
# Accumulated paper state
# ---------------------------------------------------------------------------


class SymbolContext(BaseModel):
    """Mutable paper-level context built incrementally as equations are verified.

    The context is updated monotonically — new declarations and assumptions are
    added, but never removed (unless contradicted, which would be an error).
    """

    paper_id: str = ""
    equation_order: dict[str, int] = Field(default_factory=dict)

    # symbol name → declaration (including domain & definition info)
    declarations: dict[str, SymbolDeclaration] = Field(default_factory=dict)

    # Ordered list of recorded assumptions
    assumptions: list[Assumption] = Field(default_factory=list)

    # symbol name → SymPy RHS expression string (for definitions like B = Q/N)
    definitions: dict[str, str] = Field(default_factory=dict)

    # Proof obligations — pending and resolved
    pending_obligations: list[ProofObligation] = Field(default_factory=list)
    resolved_obligations: list[ProofObligation] = Field(default_factory=list)

    # Dependency graph: snippet_id → list of outgoing edges
    dependency_graph: dict[str, list[DependencyEdge]] = Field(default_factory=dict)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def record_order(self, snippet_id: str) -> int:
        """Assign an order index to a snippet and return it."""
        if snippet_id not in self.equation_order:
            self.equation_order[snippet_id] = len(self.equation_order)
        return self.equation_order[snippet_id]

    def add_declaration(self, decl: SymbolDeclaration) -> None:
        """Add or update a symbol declaration."""
        existing = self.declarations.get(decl.symbol)
        if existing is not None:
            # Merge: keep the stronger domain.
            merged = self._merge_declarations(existing, decl)
            self.declarations[decl.symbol] = merged
        else:
            self.declarations[decl.symbol] = decl

    def add_assumption(self, assumption: Assumption) -> None:
        """Record an assumption (monotonic — never removed)."""
        self.assumptions.append(assumption)

    def add_definition(self, symbol: str, rhs_expr: str) -> None:
        """Record a definition like B = Q/N."""
        self.definitions[symbol] = rhs_expr

    def add_obligation(self, obligation: ProofObligation) -> None:
        """Track a new proof obligation."""
        self.pending_obligations.append(obligation)

    def add_edge(self, from_snippet: str, to_snippet: str,
                 relation: str = "uses_symbol") -> None:
        """Add a dependency edge."""
        edge = DependencyEdge(from_snippet=from_snippet,
                             to_snippet=to_snippet,
                             relation=relation)
        self.dependency_graph.setdefault(from_snippet, []).append(edge)

    def get_symbol_names(self) -> set[str]:
        """Return all known symbol names."""
        return set(self.declarations.keys())

    def _merge_declarations(
        self,
        existing: SymbolDeclaration,
        new: SymbolDeclaration,
    ) -> SymbolDeclaration:
        """Merge two declarations for the same symbol, keeping the stronger domain."""
        new_domain_is_stronger = (
            existing.domain == SymbolDomain.UNKNOWN
            and new.domain != SymbolDomain.UNKNOWN
        )
        if new_domain_is_stronger:
            existing.domain = new.domain
            existing.sympy_assumptions.update(new.sympy_assumptions)
        if new.is_defined and not existing.is_defined:
            existing.is_defined = True
            existing.defined_as = new.defined_as
        return existing


# ---------------------------------------------------------------------------
# Verification layer results
# ---------------------------------------------------------------------------


class LayerResult(BaseModel):
    """Output from a single verification layer."""

    layer: str
    status: str  # "proved" | "disproved" | "unknown" | "not_applicable"
    confidence: float = 0.0
    condition: Optional[str] = None
    details: str = ""
    residual: Optional[str] = None
    free_symbols: list[str] = Field(default_factory=list)
