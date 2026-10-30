TOOL ROUTING POLICY

Default to Python for:
- arithmetic, numerical computation, optimization, statistics;
- data loading, transformation, plotting, simulation;
- empirical validation, unit tests, and counterexample search;
- any task whose output is a value, file, chart, or report.

Use Lean only if the requested output is a proof or formal guarantee.
Specifically, use Lean when the task contains at least one of:
- universal quantification: “for all”, “for every”, “given any”;
- logical connectives or implications that must hold in general;
- equality, equivalence, injectivity, surjectivity, monotonicity,
  continuity, associativity, commutativity, distributivity;
- proof obligations about algorithms, protocols, types, or invariants;
- a theorem, lemma, corollary, or formal specification.

Do not use Lean merely because the task involves mathematics.
If the user asks “calculate”, “evaluate”, “approximate”, “simulate”,
“plot”, or “test”, use Python.

If the user asks “prove”, “show that for all”, “verify formally”,
“derive a theorem”, or “check logical validity”, use Lean.

Hybrid workflow:
1. Use Python to explore examples and search for counterexamples.
2. If no counterexample is found, formulate the claim precisely.
3. Use Lean to prove the precise claim.
4. Report the Python result as empirical evidence and the Lean result
   as a formal proof.

Escalate to Lean only when the statement can be formalized unambiguously.