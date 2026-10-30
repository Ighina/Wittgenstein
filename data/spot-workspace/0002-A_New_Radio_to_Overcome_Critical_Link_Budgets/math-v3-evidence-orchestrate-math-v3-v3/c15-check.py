"""CAND::15 Stage A -- counterexample search.

CLAIM (Step 1): "If ST >= LN, a single D/A-converter can serve all antennas."
Each of N antennas needs one phase transition per block; a transition occupies
L oversampled samples; the block is ST oversampled samples long; transitions at
different antennas may be placed at different times but must not overlap if one
converter is to serve them all.
FORMAL TARGET (sufficiency, which is all the paper claims):
  for every N in N, L W in R with 0 < L and N*L <= W, there exist N pairwise
  disjoint half-open intervals of length L, all contained in [0, W].

FALSIFICATION CONDITION: the paper is wrong iff there are N, L, W with
0 < L and N*L <= W for which NO packing of N disjoint length-L intervals fits
in [0, W].  (The converse direction -- necessity -- is not claimed by the paper
and is not tested.)

INSTANCE SPACE: N = 0..40, L in {1, 1/2, 1/3, 2, pi, 1e-6, 1e6} as exact
Fractions where possible, W = N*L exactly (the tight boundary case, where a
counterexample would have to live) and W = N*L + eps.  Degenerate cases
N = 0, N = 1, and the exact equality N*L = W are the ones that matter.
The constructive witness a_i = i*L is checked for containment and pairwise
disjointness by brute force over all i < j.
"""
from fractions import Fraction as F
from itertools import combinations

def disjoint(a1, a2, L):
    """half-open [a, a+L) intervals"""
    return a1 + L <= a2 or a2 + L <= a1

Ls = [F(1), F(1, 2), F(1, 3), F(2), F(1, 10**6), F(10**6), F(22, 7)]
fails = []
checked = 0
for L in Ls:
    for N in range(0, 41):
        for W in (N * L, N * L + F(1, 1000), N * L + L):   # tight and slack
            if not (N * L <= W):
                continue
            a = [F(i) * L for i in range(N)]
            contained = all(0 <= ai and ai + L <= W for ai in a)
            pw = all(disjoint(a[i], a[j], L) for i, j in combinations(range(N), 2))
            checked += 1
            if not (contained and pw):
                fails.append((str(L), N, str(W), contained, pw))

print(f"instances checked: {checked}")
print(f"   L in {[str(l) for l in Ls]}")
print(f"   N in 0..40,  W in {{N*L, N*L+1/1000, N*L+L}}  (only cases with N*L <= W)")
print(f"   witness a_i = i*L, pairwise disjointness checked over all C(N,2) pairs "
       "(up to 780 pairs per instance)")
print(f"failures found: {fails}")

print("\n-- the tight boundary case in detail, L = 1/3, N = 5, W = 5/3 --")
L, N, W = F(1, 3), 5, F(5, 3)
a = [F(i) * L for i in range(N)]
for i, ai in enumerate(a):
    print(f"   interval {i}: [{ai}, {ai + L})   inside [0, {W}) ? {ai + L <= W}")
print(f"   all pairwise disjoint ? "
      f"{all(disjoint(a[i], a[j], L) for i, j in combinations(range(N), 2))}")

print("\n-- degenerate cases --")
for N, L, W in ((0, F(1), F(0)), (1, F(1), F(1)), (1, F(1), F(2))):
    a = [F(i) * L for i in range(N)]
    print(f"   N={N}, L={L}, W={W}: N*L <= W ? {N*L <= W};  packing fits ? "
          f"{all(0 <= ai and ai + L <= W for ai in a)}")

print("\n-- the necessity direction, NOT claimed by the paper, for information --")
for L, N, W in ((F(1), 3, F(5, 2)), (F(1), 2, F(3, 2))):
    print(f"   L={L}, N={N}, W={W}: N*L = {N*L} > W, so no packing exists "
          f"(total length {N*L} exceeds the window {W})")

print("\nVERDICT: NO COUNTEREXAMPLE" if not fails else f"\nVERDICT: COUNTEREXAMPLE FOUND {fails[:3]}")
