"""CAND::14  "+pi/2 at the rightmost subcarrier ... lower it three times as much
as for -pi/2.  This results in a phase shift of -3pi/2 which does not change
the transmitted data symbol."

Falsification condition (fixed BEFORE running):
  Wrong iff (a) 3 * (-pi/2) != -3pi/2, or (b) -3pi/2 is NOT congruent to
  +pi/2 modulo 2pi (which would mean the realised QPSK symbol changes), or
  (c) |-3pi/2| <= |+pi/2|, which would contradict "the overall spectral
  regrowth is even larger".  Exact symbolic arithmetic.
"""
import sympy as sp

half = sp.pi / 2
three = 3 * (-half)
print("3 * (-pi/2)            =", sp.simplify(three), "=", sp.nsimplify(three))
print("is it -3pi/2?          ", sp.simplify(three - (-3 * sp.pi / 2)) == 0)

diff = sp.simplify((-3 * sp.pi / 2) - (sp.pi / 2))
print("(-3pi/2) - (+pi/2)     =", diff, " = -2pi ?", sp.simplify(diff + 2 * sp.pi) == 0)
k = sp.simplify(diff / (2 * sp.pi))
print("that is 2pi * k with k =", k, " integer?", k.is_integer)

print("rotation check: exp(i*(-3pi/2)) =", sp.simplify(sp.exp(sp.I * (-3 * sp.pi / 2))),
      "   exp(i*(+pi/2)) =", sp.simplify(sp.exp(sp.I * sp.pi / 2)))
print("equal rotations?", sp.simplify(sp.exp(sp.I * (-3 * sp.pi / 2)) - sp.exp(sp.I * sp.pi / 2)) == 0)

mag = sp.simplify(sp.Abs(-3 * sp.pi / 2) - sp.Abs(sp.pi / 2))
print("|-3pi/2| - |pi/2| =", mag, " > 0 ?", mag > 0,
      "  ratio =", sp.simplify(sp.Abs(-3*sp.pi/2) / sp.Abs(sp.pi/2)))

# off-by-one guard: what would 2x and 4x give?
for m in (1, 2, 3, 4):
    sh = -m * sp.pi / 2
    cong = sp.simplify(sp.exp(sp.I * sh) - sp.exp(sp.I * sp.pi / 2)) == 0
    print(f"  lowering {m}x: shift {sp.nsimplify(sh)}  == +pi/2 mod 2pi? {cong}")

a = sp.simplify(three - (-3 * sp.pi / 2)) == 0
b = k.is_integer
c = mag > 0
print(f"\n(a)={a} (b)={b} (c)={c}")
print("VERDICT: MATCH" if (a and b and c) else "VERDICT: MISMATCH")
