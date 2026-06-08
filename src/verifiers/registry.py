"""Plugin registry for verifiers.

Provides a VerifierRegistry that allows verifiers to be registered
and retrieved by name. New verifiers can be added without modifying
the orchestrator — just register them here.
"""

from __future__ import annotations

from typing import Type

from loguru import logger

from src.verifiers.base import BaseVerifier


class VerifierRegistry:
    """A registry of verifier classes keyed by name.

    Supports the plugin architecture: adding a new verifier only requires
    creating the class and registering it here (or via register()).

    Usage:
        registry = VerifierRegistry()
        registry.register("math_equation", MathEquationVerifier)
        verifier_cls = registry.get("math_equation")
    """

    def __init__(self) -> None:
        self._verifiers: dict[str, Type[BaseVerifier]] = {}

    def register(self, name: str, verifier_cls: Type[BaseVerifier]) -> None:
        """Register a verifier class.

        Args:
            name: Unique name for the verifier (e.g., "math_equation").
            verifier_cls: The verifier class (not instance).

        Raises:
            ValueError: If the name is already registered with a different class.
        """
        if name in self._verifiers and self._verifiers[name] is not verifier_cls:
            raise ValueError(
                f"Verifier '{name}' is already registered. "
                f"Use a different name or unregister first."
            )
        self._verifiers[name] = verifier_cls
        logger.debug(f"Registered verifier: {name} → {verifier_cls.__name__}")

    def unregister(self, name: str) -> None:
        """Remove a verifier from the registry."""
        if name in self._verifiers:
            del self._verifiers[name]
            logger.debug(f"Unregistered verifier: {name}")

    def get(self, name: str) -> Type[BaseVerifier]:
        """Retrieve a verifier class by name.

        Args:
            name: The verifier name.

        Returns:
            The verifier class.

        Raises:
            KeyError: If no verifier is registered under the given name.
        """
        if name not in self._verifiers:
            available = list(self._verifiers.keys())
            raise KeyError(
                f"Verifier '{name}' not found. Available: {available}"
            )
        return self._verifiers[name]

    def has(self, name: str) -> bool:
        """Check if a verifier is registered."""
        return name in self._verifiers

    def list_verifiers(self) -> list[str]:
        """Return all registered verifier names."""
        return list(self._verifiers.keys())

    def __contains__(self, name: str) -> bool:
        return self.has(name)

    def __len__(self) -> int:
        return len(self._verifiers)

    def __repr__(self) -> str:
        return f"VerifierRegistry({self.list_verifiers()})"
