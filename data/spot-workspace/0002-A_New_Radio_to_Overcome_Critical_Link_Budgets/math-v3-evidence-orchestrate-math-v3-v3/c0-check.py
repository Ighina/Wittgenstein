"""CAND::0  T=18.375 used as a matrix dimension.

Falsification condition (fixed BEFORE running):
  The paper is internally inconsistent iff the value assigned to T is not a
  non-negative integer, because T is defined as
    (a) the size of F_b in C^{T x T},
    (b) the number of columns of T_b in C^{N x T},
    (c) the cardinality in "TN entries",
    (d) the upper bound ST of the integer sample index s in Eq. (6)-(7).
  All four require T in Z_{>=0}.
"""
from fractions import Fraction
import sympy as sp

T1 = Fraction("18.375")      # first numerical example
T2 = Fraction(46)            # second numerical example
N1, N2 = 16, 32

print("T (example 1) =", T1, "=", sp.Rational(147, 8), "=", float(T1))
print("T (example 2) =", T2)
print("T1 is an integer?", T1.denominator == 1)
print("T2 is an integer?", T2.denominator == 1)
print("T1 numerator/denominator:", T1.numerator, "/", T1.denominator)

# Derived cardinalities that the text asserts exist.
print("\n-- cardinalities the text asserts --")
print("TN entries, example 1: 18.375 * 16 =", T1 * N1, "integer?", (T1 * N1).denominator == 1)
print("TN entries, example 2: 46 * 32     =", T2 * N2, "integer?", (T2 * N2).denominator == 1)

print("\n-- index bound ST must be an integer (s ranges over 0..ST-1) --")
bad = []
for S in range(1, 17):
    ST = T1 * S
    ok = ST.denominator == 1
    if not ok:
        bad.append(S)
    print(f"  S={S:2d}: ST = {float(ST):9.3f}  integer? {ok}")
print("oversampling factors S in 1..16 giving a NON-integer ST:", bad)
print("smallest S making ST integral:", min(S for S in range(1, 100) if (T1 * S).denominator == 1))

print("\n-- diagonal of an N x T matrix needs min(N,T) positions; T>N required --")
print("example 1: N=16, T=18.375 -> T > N ?", float(T1) > N1)
print("example 2: N=32, T=46     -> T > N ?", float(T2) > N2)

integer_T = (T1.denominator == 1)
if not integer_T:
    print("\nVERDICT: MISMATCH -- T = 147/8 = 18.375 is not an integer, so "
          "F_b in C^{TxT}, T_b in C^{NxT} and the index bound ST are not "
          "well-defined objects for the first numerical example.")
else:
    print("\nVERDICT: MATCH")
