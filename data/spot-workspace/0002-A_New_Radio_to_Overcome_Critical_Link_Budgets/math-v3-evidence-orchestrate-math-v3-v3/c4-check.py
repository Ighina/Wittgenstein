"""CAND::4  The paper indexes the transition sequences by d in {1,...,|A|^2}
while saying the waveform depends on the data symbols of blocks b-1, b, b+1.

Falsification condition (fixed BEFORE running):
  Literal reading: the waveform is an arbitrary function of the ORDERED TRIPLE
  (x_{b-1}, x_b, x_{b+1}) in A^3.  Then the number of cases is |A|^3, and the
  paper's |A|^2 is too small -> MISMATCH.
  Rotation reading: the waveform shape is determined by the two consecutive
  phase increments only (equivalently, it is equivariant under a common phase
  rotation of all three symbols).  Then the number of cases is the number of
  orbits of A^3 under the diagonal action of A, which we count exhaustively.
  The paper's |A|^2 is right iff that orbit count equals |A|^2.
"""
from itertools import product

for M in (2, 4, 8, 16):          # |A| for BPSK, QPSK, 8-PSK, 16-PSK
    A = list(range(M))           # phases k*2pi/M represented by k
    triples = list(product(A, repeat=3))
    assert len(triples) == M**3

    # orbits of A^3 under the diagonal rotation x -> x + r (mod M)
    orbits = set()
    for t in triples:
        orbit = frozenset(tuple((x + r) % M for x in t) for r in A)
        orbits.add(orbit)

    # equivalently: the set of increment pairs
    increments = {((t[1] - t[0]) % M, (t[2] - t[1]) % M) for t in triples}

    print(f"|A| = {M:2d}:  |A|^3 = {M**3:5d}   |A|^2 = {M*M:4d}   "
          f"orbits under common rotation = {len(orbits):4d}   "
          f"distinct increment pairs = {len(increments):4d}")
    assert len(orbits) == len(increments)

print()
print("Checked |A| in {2,4,8,16} exhaustively over all |A|^3 ordered triples.")
print("orbit count == |A|^2 for every |A| checked:",
      all(len({frozenset(tuple((x + r) % M for x in t) for r in range(M))
               for t in product(range(M), repeat=3)}) == M*M
          for M in (2, 4, 8, 16)))
print()
print("VERDICT: MATCH under the rotation reading -- the number of distinct "
      "phase-transition patterns for three consecutive PSK symbols is exactly "
      "|A|^2, so d in {1,...,|A|^2} is the right index range provided the "
      "waveform depends only on the two phase increments. "
      "VERDICT: MISMATCH under the literal reading -- an arbitrary function of "
      "the ordered triple would need |A|^3 cases. The paper never states the "
      "rotation invariance, so the computation does not settle which reading "
      "was intended.")
