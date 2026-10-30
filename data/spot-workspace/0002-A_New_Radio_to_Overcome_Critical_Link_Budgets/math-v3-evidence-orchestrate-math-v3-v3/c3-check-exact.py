"""CAND::3 Stage A, revision 1 -- exact (sympy) cross-check of the numeric run.

Revision note: the numpy run emitted RuntimeWarnings from complex matmul on
this platform (a BLAS quirk, the printed values were still correct). This
revision redoes the structural cases in EXACT arithmetic so the evidence does
not rest on floating point.
"""
import sympy as sp

for T in (2, 3, 4, 6, 8):
    W = sp.Matrix(T, T, lambda a, b: sp.exp(2*sp.pi*sp.I*a*b/T))   # unit modulus
    F = W / sp.sqrt(T)
    WhW = sp.simplify(W.conjugate().T * W)
    FhF = sp.simplify(F.conjugate().T * F)
    rown_W = sp.simplify(sum(sp.Abs(W[0, s])**2 for s in range(T)))
    rown_F = sp.simplify(sum(sp.Abs(F[0, s])**2 for s in range(T)))
    print(f"T={T}:")
    print(f"   unit-modulus W: ||row||^2 = {rown_W},  W^H W = {T}*I ? "
          f"{sp.simplify(WhW - T*sp.eye(T)) == sp.zeros(T, T)},  "
          f"W unitary? {sp.simplify(WhW - sp.eye(T)) == sp.zeros(T, T)}")
    print(f"   normalised  F: ||row||^2 = {rown_F},  F unitary? "
          f"{sp.simplify(FhF - sp.eye(T)) == sp.zeros(T, T)}")

print()
print("general exact fact, by the diagonal of F^H F = I:")n = sp.symbols('n')
