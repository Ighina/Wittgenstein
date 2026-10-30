"""CAND::13  The four QPSK phase-shift sets assigned by subcarrier position.

Falsification condition (fixed BEFORE running):
  A set is unusable iff, reduced mod 2*pi, it is not exactly
  {0, pi/2, pi, 3pi/2} -- i.e. it repeats a residue or omits one, leaving some
  QPSK transition unimplementable.  Additionally the set for the RIGHTMOST
  subcarrier must be entirely non-positive (frequency only lowered) and the set
  for the LEFTMOST entirely non-negative.  Exact arithmetic in units of pi/2.
"""
import sympy as sp
from fractions import Fraction as F

# elements expressed as multiples of pi/2, exactly
sets = {
    "S1 {0, -pi/2, -pi, -3pi/2}":   [F(0), F(-1), F(-2), F(-3)],
    "S2 {+pi/2, 0, -pi/2, -pi}":    [F(1), F(0), F(-1), F(-2)],
    "S3 {+pi, +pi/2, 0, -pi/2}":    [F(2), F(1), F(0), F(-1)],
    "S4 {+3pi/2, +pi, +pi/2, 0}":   [F(3), F(2), F(1), F(0)],
}
target = {F(0), F(1), F(2), F(3)}          # residues mod 4 (units of pi/2)

allok = True
for name, elems in sets.items():
    residues = [e % 4 for e in elems]      # mod 2pi == mod 4 in units of pi/2
    complete = set(residues) == target and len(set(residues)) == 4
    nonpos = all(e <= 0 for e in elems)
    nonneg = all(e >= 0 for e in elems)
    allok &= complete
    print(f"{name}")
    print(f"   elements (units of pi/2): {[str(e) for e in elems]}")
    print(f"   residues mod 2pi         : {sorted(str(r) for r in residues)}")
    print(f"   complete QPSK shift set? {complete}")
    print(f"   all non-positive (rightmost-subcarrier safe)? {nonpos}")
    print(f"   all non-negative (leftmost-subcarrier safe)?  {nonneg}")
    print()

s1 = all(e <= 0 for e in sets["S1 {0, -pi/2, -pi, -3pi/2}"])
s4 = all(e >= 0 for e in sets["S4 {+3pi/2, +pi, +pi/2, 0}"])
# cross-check numerically that each set's complex rotations are the 4th roots of unity
for name, elems in sets.items():
    rots = {sp.simplify(sp.exp(sp.I * sp.pi / 2 * sp.Rational(e.numerator, e.denominator)))
            for e in elems}
    print(f"{name}: distinct unit-circle rotations = {len(rots)} -> {sorted(map(str, rots))}")

print(f"\nall four sets complete mod 2pi: {allok}")
print(f"S1 one-sided negative: {s1}   S4 one-sided positive: {s4}")
print("VERDICT: MATCH" if (allok and s1 and s4) else "VERDICT: MISMATCH")
