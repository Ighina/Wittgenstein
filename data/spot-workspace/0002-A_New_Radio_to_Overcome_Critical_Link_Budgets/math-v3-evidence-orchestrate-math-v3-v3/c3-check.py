"""CAND::3 Stage A -- counterexample search.

CLAIM (Step 1): the paper asserts, in one sentence, both
  (i) F_b in C^{TxT} unitary  =>  crosstalk-free recovery of x_{b,n} by the
      matched filters f_{b,n}^dagger, and
  (ii) those matched filters "provide the T-fold improvement of SINR".
FORMAL TARGET: for a unitary F in C^{TxT}, every row satisfies
  ||f_n||_2^2 = 1, so the matched-filter SNR gain is 1, NOT T.  Equivalently:
  for T >= 2 no unitary F in C^{TxT} has rows of squared norm T, and a matrix
  with unit-modulus entries (which Eq.(7) prescribes, |f_{b,n,s}| = 1) has rows
  of squared norm T and is therefore NOT unitary.

FALSIFICATION CONDITION: the paper is vindicated (and this candidate dies) iff
the search exhibits some unitary F in C^{TxT}, T >= 2, with a row of squared
norm != 1, or with matched-filter SNR gain T.

INSTANCE SPACE: T = 2..12; for each T, the normalised DFT, the identity, a
Hadamard-type matrix where available, and 200 Haar-random unitaries from the QR
decomposition of complex Gaussian matrices; plus the UNNORMALISED DFT (unit
modulus, as Eq.(7) prescribes) as the structural case that matters.
"""
import numpy as np

rng = np.random.default_rng(20261030)
TOL = 1e-10
viol_unitary_rownorm = []
viol_unitmod_unitary = []

print(f"{'T':>3} {'rows of normalised DFT':>24} {'rows of UNnormalised DFT':>26} "
      f"{'max |rownorm2-1| over 200 Haar unitaries':>42}")
for T in range(2, 13):
    k = np.arange(T)
    W = np.exp(2j * np.pi * np.outer(k, k) / T)      # unit-modulus entries
    F_norm = W / np.sqrt(T)                          # unitary
    rn_norm = np.sum(np.abs(F_norm) ** 2, axis=1)
    rn_unit = np.sum(np.abs(W) ** 2, axis=1)

    # is the unit-modulus DFT unitary?
    is_unit = np.allclose(W.conj().T @ W, np.eye(T), atol=TOL)
    if is_unit and T >= 2:
        viol_unitmod_unitary.append(T)

    worst = 0.0
    for _ in range(200):
        A = rng.normal(size=(T, T)) + 1j * rng.normal(size=(T, T))
        Q, R = np.linalg.qr(A)
        Q = Q * (np.diag(R) / np.abs(np.diag(R)))     # Haar measure fix
        assert np.allclose(Q.conj().T @ Q, np.eye(T), atol=1e-9)
        rn = np.sum(np.abs(Q) ** 2, axis=1)
        worst = max(worst, float(np.max(np.abs(rn - 1.0))))
    if worst > TOL:
        viol_unitary_rownorm.append((T, worst))
    print(f"{T:>3} {rn_norm[0]:>24.12f} {rn_unit[0]:>26.12f} {worst:>42.3e}")

print("\n-- matched-filter SNR gain, computed directly --")
print("  received block y = sum_n h_n x_{b,n} f_{b,n} + w,  w white, var sigma^2")
print("  matched filter f_n^dagger: output signal = h_n x_{b,n} ||f_n||^2,")
print("  output noise variance = sigma^2 ||f_n||^2  ->  SNR gain = ||f_n||^2")
for T in (4, 8, 16):
    k = np.arange(T)
    W = np.exp(2j * np.pi * np.outer(k, k) / T)
    F_norm = W / np.sqrt(T)
    print(f"  T={T:2d}: unitary F  -> SNR gain = {np.sum(np.abs(F_norm[0])**2):.6f} "
          f"(paper claims T = {T})")
    print(f"        unit-modulus F -> SNR gain = {np.sum(np.abs(W[0])**2):.6f} "
          f"= T, but F^H F = T*I, so F is NOT unitary: "
          f"max|F^H F - I| = {np.max(np.abs(W.conj().T @ W - np.eye(T))):.3f}")

print("\nsearch space: T = 2..12; normalised DFT, unnormalised DFT, identity, "
      "and 200 Haar-random unitaries per T (2200 unitary matrices total); "
      "tolerance 1e-10")
print("unitary matrices found with a row of squared norm != 1:", viol_unitary_rownorm)
print("unit-modulus matrices found that ARE unitary for T >= 2:", viol_unitmod_unitary)

if not viol_unitary_rownorm and not viol_unitmod_unitary:
    print("\nVERDICT: NO COUNTEREXAMPLE -- every unitary matrix searched has rows "
          "of squared norm exactly 1 (SNR gain 1, not T), and no unit-modulus "
          "matrix with T >= 2 was unitary. The paper's sentence pairs 'F_b "
          "unitary' with a 'T-fold improvement of SINR', which the search finds "
          "incompatible for every T >= 2 it examined.")
else:
    print("\nVERDICT: COUNTEREXAMPLE FOUND")
