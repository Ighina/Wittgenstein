"""CAND::1  Eq.(7) exponent uses SN; a T x T DFT-like F_b would require ST.

Falsification condition (fixed BEFORE running):
  The two descriptions disagree iff
     exp(2*pi*j*s*n/(S*N))  !=  exp(2*pi*j*s*n/(S*T))   identically in s,n,
  for the paper's own instantiations (N=16,T=147/8) and (N=32,T=46).
  Additionally: whichever denominator reproduces the paper's printed subcarrier
  spacings (5.74 MHz and 2.88 MHz) is the convention actually in use; the other
  description is then the mistaken one.
"""
import sympy as sp
from fractions import Fraction

s, n, S = sp.symbols('s n S', positive=True)

for (Nv, Tv, spacing_MHz, Tsym_ns) in [(16, sp.Rational(147, 8), 5.74, 200),
                                       (32, sp.Rational(46), 2.88, 500)]:
    print(f"=== N={Nv}, T={Tv} ({float(Tv)}) ===")
    e_SN = sp.exp(2*sp.pi*sp.I*s*n/(S*Nv))
    e_ST = sp.exp(2*sp.pi*sp.I*s*n/(S*Tv))
    diff = sp.simplify(e_SN - e_ST)
    print("  simplify(exp(2pi i s n/(S N)) - exp(2pi i s n/(S T))) =", diff)
    print("  identically zero?", diff == 0)
    # concrete witness: first subcarrier n=1, one sample s=1, S=8
    w_SN = sp.simplify(e_SN.subs({n: 1, s: 1, S: 8}))
    w_ST = sp.simplify(e_ST.subs({n: 1, s: 1, S: 8}))
    print("  witness n=1,s=1,S=8:  SN-form =", sp.N(w_SN, 12), " ST-form =", sp.N(w_ST, 12))
    print("  |difference| =", sp.N(sp.Abs(w_SN - w_ST), 12))

    # which denominator reproduces the printed subcarrier spacing?
    fs_MHz = float(Tv) / (Tsym_ns * 1e-3)          # T samples per symbol period -> MHz
    print(f"  sampling rate f_s = T/Tsym = {fs_MHz:.6f} MHz")
    print(f"  f_s/N = {fs_MHz/Nv:.6f} MHz   (SN convention)")
    print(f"  f_s/T = {fs_MHz/float(Tv):.6f} MHz   (ST convention)")
    print(f"  paper prints        {spacing_MHz} MHz")
    print(f"  SN convention matches paper to 0.01 MHz? {abs(fs_MHz/Nv - spacing_MHz) < 0.01}")
    print(f"  ST convention matches paper to 0.01 MHz? {abs(fs_MHz/float(Tv) - spacing_MHz) < 0.01}")
    print()

print("VERDICT: MISMATCH -- the SN exponent of Eq.(7) and the 'T x T matrix close "
      "to the DFT' description of F_b are different waveforms whenever T != N, and "
      "both numerical examples have T != N. The printed subcarrier spacings follow "
      "the SN convention, so the T x T DFT description is the inconsistent half.")
