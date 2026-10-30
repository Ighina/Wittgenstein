"""CAND::2  Sinusoidal-phase index range: Eq.(7) says s < ST - L_e,
Section IV-A-1 says s < SN - L_e.

Falsification condition (fixed BEFORE running):
  The two stated ranges denote the same index set for all L_s, L_e iff ST == SN,
  i.e. iff T == N. The paper's own instantiations are (N=16, T=147/8) and
  (N=32, T=46).  If T != N in either, the two sentences specify different
  intervals and one of them uses the wrong symbol.
"""
import sympy as sp

S = sp.symbols('S', positive=True, integer=True)
for Nv, Tv in [(16, sp.Rational(147, 8)), (32, sp.Rational(46))]:
    ST, SN = S*Tv, S*Nv
    print(f"=== N={Nv}, T={Tv} ({float(Tv)}) ===")
    print("  ST - SN =", sp.simplify(ST - SN), "  identically zero?", sp.simplify(ST - SN) == 0)
    for Sv in (1, 8, 16):
        a, b = float(Tv*Sv), float(Nv*Sv)
        print(f"  S={Sv:2d}: ST={a:8.3f}  SN={b:8.3f}  difference={a-b:8.3f} samples")
    # how many samples of the block are mis-specified, as a fraction
    print(f"  relative discrepancy |ST-SN|/ST = {abs(float(Tv)-Nv)/float(Tv):.4%}")
    print()

same = all(sp.simplify(sp.Rational(t) - n) == 0
           for n, t in [(16, sp.Rational(147, 8)), (32, sp.Rational(46))])
print("T == N in every example?", same)
print("VERDICT: MISMATCH -- Eq.(7) bounds the sinusoidal phase by ST while "
      "Section IV-A-1 bounds it by SN; with T=18.375!=N=16 and T=46!=N=32 these "
      "are different index sets (a 2.375S and a 14S sample discrepancy "
      "respectively), so the two statements cannot both be right.")
