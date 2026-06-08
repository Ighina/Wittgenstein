"""Progressive math verification with context accumulation and multi-layer checking.

Exports the ``ProgressiveMathVerifier`` class which replaces single-equation
SymPy checking with a progressive constraint-and-proof pipeline.
"""

from src.verifiers.progressive.progressive_verifier import ProgressiveMathVerifier

__all__ = ["ProgressiveMathVerifier"]
