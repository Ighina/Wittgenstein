"""Dependency graph for verifiable units in an EnrichedPaper.

Tracks which units depend on which other units (symbol definitions, prior
lemmas, etc.) and can assemble the full context a verifier needs for any
given unit.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Optional

from src.models import SymbolDefinition, VerifiableUnit


class ContextGraph:
    """Tracks dependencies between verifiable units in a paper.

    Each node is a VerifiableUnit. Edges point from a dependent unit to its
    prerequisites (symbol definitions, prior theorems, etc.). The graph is
    used at segmentation time to assemble the full context a verifier needs.
    """

    def __init__(self) -> None:
        # unit_id → VerifiableUnit
        self._units: dict[str, VerifiableUnit] = {}

        # unit_id → list of prerequisite unit_ids
        self._deps: dict[str, list[str]] = defaultdict(list)

        # unit_id → list of unit_ids that depend on it (reverse edges)
        self._dependents: dict[str, list[str]] = defaultdict(list)

        # symbol_name → SymbolDefinition
        self._symbols: dict[str, SymbolDefinition] = {}

        # Accumulated unverifiable text
        self._unverifiable_text_parts: list[str] = []

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def add_unit(self, unit: VerifiableUnit) -> None:
        """Register a verifiable unit in the graph."""
        self._units[unit.unit_id] = unit
        if unit.unit_id not in self._deps:
            self._deps[unit.unit_id] = []

    def add_dependency(self, unit_id: str, depends_on_id: str) -> None:
        """Record that ``unit_id`` depends on ``depends_on_id``."""
        if depends_on_id not in self._deps[unit_id]:
            self._deps[unit_id].append(depends_on_id)
        if unit_id not in self._dependents[depends_on_id]:
            self._dependents[depends_on_id].append(unit_id)

    def add_symbol(self, symbol: SymbolDefinition) -> None:
        """Register a symbol definition."""
        self._symbols[symbol.symbol_name] = symbol

    def add_unverifiable_text(self, text: str) -> None:
        """Append a chunk of unverifiable text to the reference context."""
        if text and text.strip():
            self._unverifiable_text_parts.append(text.strip())

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get_unit(self, unit_id: str) -> Optional[VerifiableUnit]:
        """Return a unit by ID, or None."""
        return self._units.get(unit_id)

    def get_dependencies(self, unit_id: str) -> list[str]:
        """Return the direct dependency IDs for a unit."""
        return list(self._deps.get(unit_id, []))

    def get_symbol(self, name: str) -> Optional[SymbolDefinition]:
        """Look up a symbol definition by name."""
        return self._symbols.get(name)

    def has_unit(self, unit_id: str) -> bool:
        return unit_id in self._units

    @property
    def unit_count(self) -> int:
        return len(self._units)

    @property
    def symbol_count(self) -> int:
        return len(self._symbols)

    @property
    def unverifiable_text(self) -> str:
        return "\n\n".join(self._unverifiable_text_parts)

    # ------------------------------------------------------------------
    # Context assembly
    # ------------------------------------------------------------------

    def resolve_context(
        self,
        unit: VerifiableUnit,
        max_chars: int = 8000,
    ) -> str:
        """Assemble the full dependency context for a unit, bounded by ``max_chars``.

        Walks the dependency graph from ``unit`` backwards, collecting the
        content of all prerequisite units. The result is truncated to
        ``max_chars`` with newer (closer) dependencies kept preferentially.

        Args:
            unit: The unit to resolve context for.
            max_chars: Hard character limit on the assembled context.

        Returns:
            A string containing definitions and prior results that the unit
            depends on, suitable for prepending to a verifier prompt.
        """
        if not unit.dependencies and not self._deps.get(unit.unit_id):
            return ""

        # Collect all transitive dependencies via BFS
        visited: set[str] = set()
        queue: deque[str] = deque()

        for dep_id in self._deps.get(unit.unit_id, []):
            if dep_id not in visited:
                visited.add(dep_id)
                queue.append(dep_id)

        # Also check the unit's own dependency list (which may use symbol names)
        for dep_ref in unit.dependencies:
            # Could be a unit_id or a symbol name
            if dep_ref in self._units and dep_ref not in visited:
                visited.add(dep_ref)
                queue.append(dep_ref)
            elif dep_ref in self._symbols and self._symbols[dep_ref].defining_unit_id:
                def_unit = self._symbols[dep_ref].defining_unit_id
                if def_unit in self._units and def_unit not in visited:
                    visited.add(def_unit)
                    queue.append(def_unit)

        # BFS: collect transitive deps (limited depth to prevent runaway)
        depth = 0
        while queue and depth < 5:
            for _ in range(len(queue)):
                current = queue.popleft()
                for child_dep in self._deps.get(current, []):
                    if child_dep not in visited:
                        visited.add(child_dep)
                        queue.append(child_dep)
            depth += 1

        # Assemble context: order by insertion (dependency order ≈ topological)
        parts: list[str] = []
        total_chars = 0

        for dep_id in sorted(visited):
            dep_unit = self._units.get(dep_id)
            if dep_unit is None:
                continue
            text = f"[{dep_unit.unit_type}] {dep_unit.location}\n{dep_unit.content}"
            if total_chars + len(text) > max_chars:
                # Truncate with indicator
                remaining = max_chars - total_chars
                if remaining > 100:
                    parts.append(text[:remaining] + "\n[...context truncated...]")
                break
            parts.append(text)
            total_chars += len(text) + 1  # +1 for newline separator

        return "\n\n".join(parts)

    def topological_order(self) -> list[str]:
        """Return unit IDs in topological (dependency-first) order.

        Units with no dependencies come first; units that depend on them follow.
        """
        in_degree: dict[str, int] = {}
        for uid in self._units:
            in_degree[uid] = len(self._deps.get(uid, []))

        queue: deque[str] = deque(
            uid for uid, deg in in_degree.items() if deg == 0
        )
        result: list[str] = []

        while queue:
            uid = queue.popleft()
            result.append(uid)
            for dependent in self._dependents.get(uid, []):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        # Any remaining units are in a cycle — append them
        for uid in self._units:
            if uid not in result:
                result.append(uid)

        return result

    def as_dict(self) -> dict[str, list[str]]:
        """Return the dependency graph as a plain dict (for serialization)."""
        return dict(self._deps)

    def symbols_as_list(self) -> list[SymbolDefinition]:
        return list(self._symbols.values())
