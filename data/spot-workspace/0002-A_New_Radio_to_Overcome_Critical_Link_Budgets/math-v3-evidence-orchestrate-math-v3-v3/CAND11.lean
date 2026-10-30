import Mathlib

open Filter Topology Real

namespace PaperCand11

/-- **Eq. (10) of the paper, as a limit.**  The ratio of `log₂(1+x)` to the
paper's linearisation `x / ln 2` tends to `1` as `x → 0⁺`.  This is exactly the
assertion that, for `SNR ≪ 1`, `R = log₂(1+SNR) ≈ SNR / ln 2`, and it pins the
constant to `1 / ln 2`. -/
theorem logb_one_add_div_tendsto_one :
    Tendsto (fun x : ℝ => Real.logb 2 (1 + x) / (x / Real.log 2))
      (𝓝[>] (0 : ℝ)) (𝓝 1) := by
  have hlog2 : Real.log 2 ≠ 0 := ne_of_gt (Real.log_pos (by norm_num))
  have hkey : Tendsto (fun x : ℝ => Real.log (1 + x) / x)
      (𝓝[>] (0 : ℝ)) (𝓝 1) := by
    have h1 : HasDerivAt (fun x : ℝ => 1 + x) 1 0 := by
      simpa using (hasDerivAt_id (0 : ℝ)).const_add (1 : ℝ)
    have hd : HasDerivAt (fun x : ℝ => Real.log (1 + x)) 1 0 := by
      simpa using h1.log (by norm_num : (1 : ℝ) + 0 ≠ 0)
    have hslope := hd.tendsto_slope_zero_right
    refine Tendsto.congr' ?_ hslope
    filter_upwards [self_mem_nhdsWithin] with x hx
    simp [div_eq_inv_mul]
  refine Tendsto.congr' ?_ hkey
  filter_upwards [self_mem_nhdsWithin] with x hx
  have hx0 : x ≠ 0 := ne_of_gt hx
  rw [Real.logb]
  field_simp

/-- The rival constant `ln 2` does **not** work: the ratio of `log₂(1+x)` to
`x * ln 2` tends to `1 / (ln 2)^2 = 2.081… ≠ 1`.  This is what makes the limit
above a statement about the constant and not merely about linearity. -/
theorem rival_constant_fails :
    Tendsto (fun x : ℝ => Real.logb 2 (1 + x) / (x * Real.log 2))
      (𝓝[>] (0 : ℝ)) (𝓝 (1 / (Real.log 2) ^ 2)) := by
  have hlog2 : Real.log 2 ≠ 0 := ne_of_gt (Real.log_pos (by norm_num))
  have h := logb_one_add_div_tendsto_one.mul_const (1 / (Real.log 2) ^ 2)
  rw [one_mul] at h
  refine Tendsto.congr' ?_ h
  filter_upwards [self_mem_nhdsWithin] with x hx
  have hx0 : x ≠ 0 := ne_of_gt hx
  rw [Real.logb]
  field_simp

/-- `1 / (ln 2)^2 ≠ 1`, so the two constants really are distinguishable. -/
theorem rival_limit_ne_one : 1 / (Real.log 2) ^ 2 ≠ (1 : ℝ) := by
  have h1 : Real.log 2 < 1 := by
    have h2e : (2 : ℝ) < Real.exp 1 := by
      have := Real.exp_one_gt_d9
      linarith
    have := Real.log_lt_log (by norm_num : (0:ℝ) < 2) h2e
    simpa using this
  have h0 : 0 < Real.log 2 := Real.log_pos (by norm_num)
  intro hc
  have hsq : (Real.log 2) ^ 2 < 1 := by nlinarith
  rw [div_eq_one_iff_eq (by positivity)] at hc
  linarith

end PaperCand11
