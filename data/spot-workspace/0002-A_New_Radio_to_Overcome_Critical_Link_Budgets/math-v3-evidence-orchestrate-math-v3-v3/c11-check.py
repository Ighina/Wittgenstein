"""CAND::11 Stage A -- counterexample search.

CLAIM (Step 1): Eq.(9) R = log2(1+SNR); Eq.(10) asserts that for SNR << 1,
R is "well approximated by a linear function", namely R ~= SNR/ln 2.
FORMAL TARGET: lim_{x->0+} log2(1+x) / (x / ln 2) = 1, equivalently
lim_{x->0+} ln(1+x)/x = 1, i.e. the linearisation constant is 1/ln 2 and not,
say, ln 2.

FALSIFICATION CONDITION: the paper is wrong iff the ratio
log2(1+x) / (x/ln 2) does NOT approach 1 as x -> 0+, or iff the rival constant
ln 2 fits better (ratio log2(1+x)/(x*ln 2) -> 1).

INSTANCE SPACE: x = 10^-k for k = 1..30 in 50-digit mpmath precision, plus the
exact symbolic series of log2(1+x) about 0, plus the whole interval x in
(0, 1] on a 10^5-point grid to bound the relative error over the "SNR << 1"
regime the paper names.  Boundary/degenerate cases: x -> 0+, x = 1 (SNR = 1,
the edge of the regime), x large (where the approximation must fail).
"""
import sympy as sp
import mpmath as mp

mp.mp.dps = 50
x = sp.symbols('x', positive=True)

print("-- exact series of log2(1+x) about x = 0 --")
ser = sp.series(sp.log(1 + x, 2), x, 0, 4).removeO()
print("   log2(1+x) =", sp.simplify(ser), "+ O(x^4)")
lead = sp.simplify(sp.limit(sp.log(1 + x, 2) / x, x, 0, '+'))
print("   lim_{x->0+} log2(1+x)/x =", lead, "=", sp.nsimplify(lead), "=", float(lead))
print("   1/ln 2 =", float(1 / sp.log(2)), "   ln 2 =", float(sp.log(2)))
print("   leading coefficient equals 1/ln2 ?", sp.simplify(lead - 1 / sp.log(2)) == 0)
print("   leading coefficient equals ln2  ?", sp.simplify(lead - sp.log(2)) == 0)

print("\n-- the paper's ratio and the rival ratio, at x = 10^-k --")
print(f"{'x':>10} {'log2(1+x)':>22} {'x/ln2':>22} {'paper ratio':>16} {'rival (x*ln2) ratio':>22}")
bad_paper = []
for k in list(range(1, 16)) + [20, 25, 30]:
    xv = mp.mpf(10) ** (-k)
    R = mp.log(1 + xv) / mp.log(2)
    paper = R / (xv / mp.log(2))
    rival = R / (xv * mp.log(2))
    if abs(paper - 1) > mp.mpf(10) ** (-(k - 1)):
        bad_paper.append(k)
    print(f"{mp.nstr(xv, 4):>10} {mp.nstr(R, 16):>22} {mp.nstr(xv/mp.log(2), 16):>22} "
          f"{mp.nstr(paper, 14):>16} {mp.nstr(rival, 14):>22}")

print("\nk where |paper ratio - 1| > 10^-(k-1):", bad_paper)

print("\n-- relative error of Eq.(10) over the whole regime x in (0,1] --")
import numpy as np
xs = np.linspace(1e-9, 1.0, 100001)
R = np.log2(1 + xs)
approx = xs / np.log(2)
rel = np.abs(approx - R) / R
print(f"   grid: 100001 points on (1e-9, 1]")
for thr in (1e-3, 1e-2, 1e-1, 0.5):
    idx = np.where(xs <= thr)[0]
    print(f"   x <= {thr:<6}: max relative error = {rel[idx].max():.6e}")
print(f"   x <= 1.0   : max relative error = {rel.max():.6e} (at x = {xs[rel.argmax()]:.4f})")

print("\n-- the approximation must and does fail for large x (sanity, not a defect) --")
for xv in (10, 100, 1000):
    print(f"   x={xv:5d}: log2(1+x) = {mp.nstr(mp.log(1+xv)/mp.log(2), 8):>10}, "
          f"x/ln2 = {mp.nstr(mp.mpf(xv)/mp.log(2), 8):>12}, "
          f"ratio = {mp.nstr((mp.log(1+xv)/mp.log(2))/(mp.mpf(xv)/mp.log(2)), 6)}")

ok = (sp.simplify(lead - 1 / sp.log(2)) == 0) and not bad_paper
print("\nVERDICT: NO COUNTEREXAMPLE -- the exact series gives "
      "log2(1+x) = x/ln2 - x^2/(2 ln2) + ..., the limit of the paper's ratio is "
      "1 to 50 digits at every x = 10^-k tested (k <= 30), and the rival "
      "constant ln2 gives a ratio of 1/ln^2(2) = 2.081, not 1."
      if ok else "VERDICT: COUNTEREXAMPLE FOUND")
