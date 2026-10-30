import Mathlib

open Matrix Finset

namespace PaperCand3

/-- **Row energy of a unitary matrix.**  Every row of a unitary `T × T` complex
matrix has squared Euclidean norm exactly `1`.  Since the matched filter built
from row `n` of `F` has SNR gain `‖f_n‖₂²`, a unitary `F` gives gain `1`. -/
theorem unitary_row_sq_norm {T : ℕ} (F : Matrix (Fin T) (Fin T) ℂ)
    (hU : F * Fᴴ = 1) (n : Fin T) :
    ∑ s : Fin T, ‖F n s‖ ^ 2 = 1 := by
  have h : (F * Fᴴ) n n = (1 : Matrix (Fin T) (Fin T) ℂ) n n := by rw [hU]
  rw [Matrix.mul_apply, Matrix.one_apply_eq] at h
  have h2 : ∑ s : Fin T, ((‖F n s‖ ^ 2 : ℝ) : ℂ) = 1 := by
    rw [← h]
    refine Finset.sum_congr rfl ?_
    intro s _
    rw [Matrix.conjTranspose_apply, RCLike.star_def, Complex.mul_conj']
    push_cast
    ring
  exact_mod_cast h2

/-- **A unit-modulus matrix is not unitary for `T ≥ 2`.**  Eq. (7) of the paper
prescribes `|f_{b,n,s}| = 1` (constant envelope), which forces `‖f_n‖₂² = T`;
by `unitary_row_sq_norm` a unitary matrix would need `‖f_n‖₂² = 1`. -/
theorem unit_modulus_not_unitary {T : ℕ} (hT : 2 ≤ T)
    (F : Matrix (Fin T) (Fin T) ℂ) (hmod : ∀ n s, ‖F n s‖ = 1) :
    F * Fᴴ ≠ 1 := by
  intro hU
  have hpos : 0 < T := lt_of_lt_of_le (by norm_num) hT
  let n : Fin T := ⟨0, hpos⟩
  have h1 : ∑ s : Fin T, ‖F n s‖ ^ 2 = 1 := unitary_row_sq_norm F hU n
  have h2 : ∑ s : Fin T, ‖F n s‖ ^ 2 = (T : ℝ) := by
    simp [hmod]
  rw [h2] at h1
  have : (2 : ℝ) ≤ (T : ℝ) := by exact_mod_cast hT
  linarith

/-- The two halves of the paper's sentence are incompatible for `T ≥ 2`:
no `T × T` matrix is simultaneously unitary and has a row whose matched-filter
SNR gain is `T`. -/
theorem no_unitary_with_T_fold_gain {T : ℕ} (hT : 2 ≤ T)
    (F : Matrix (Fin T) (Fin T) ℂ) (hU : F * Fᴴ = 1) (n : Fin T) :
    ∑ s : Fin T, ‖F n s‖ ^ 2 ≠ (T : ℝ) := by
  rw [unitary_row_sq_norm F hU n]
  have : (2 : ℝ) ≤ (T : ℝ) := by exact_mod_cast hT
  intro hc
  linarith

end PaperCand3
