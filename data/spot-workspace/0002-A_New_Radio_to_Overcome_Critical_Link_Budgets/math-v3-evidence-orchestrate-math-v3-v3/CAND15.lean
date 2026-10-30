import Mathlib

open Set

namespace PaperCand15

/-- Two half-open intervals of length `L` are disjoint once the first ends at
or before the second begins. -/
private theorem ico_disj {a b L : ℝ} (h : a + L ≤ b) :
    Disjoint (Ico a (a + L)) (Ico b (b + L)) := by
  rw [Set.disjoint_left]
  intro x hx hx'
  exact absurd (lt_of_lt_of_le hx.2 (le_trans h hx'.1)) (lt_irrefl x)

/-- **Round-robin packing: the paper's `ST ≥ LN` condition is sufficient.**
If `N` transitions of length `L` have total length at most the block length
`W`, they can be placed inside `[0, W]` without overlapping — the witness being
`a i = i * L`.  With `W = S*T` the block length in oversampled samples and `L`
the transition length, this is exactly the paper's "If `ST ≥ LN`, a single
D/A-converter can serve all antennas." -/
theorem roundRobin_packing_of_le {N : ℕ} {L W : ℝ} (hL : 0 < L)
    (h : (N : ℝ) * L ≤ W) :
    ∃ a : Fin N → ℝ,
      (∀ i, 0 ≤ a i ∧ a i + L ≤ W) ∧
      (∀ i j, i ≠ j → Disjoint (Ico (a i) (a i + L)) (Ico (a j) (a j + L))) := by
  refine ⟨fun i => (i : ℕ) * L, ?_, ?_⟩
  · intro i
    refine ⟨by positivity, ?_⟩
    have hi : ((i : ℕ) : ℝ) + 1 ≤ (N : ℝ) := by
      have hnat : (i : ℕ) + 1 ≤ N := i.isLt
      exact_mod_cast hnat
    nlinarith
  · intro i j hij
    have hne : (i : ℕ) ≠ (j : ℕ) := fun hc => hij (Fin.ext hc)
    rcases lt_or_gt_of_ne hne with hlt | hgt
    · refine ico_disj ?_
      have hle : ((i : ℕ) : ℝ) + 1 ≤ ((j : ℕ) : ℝ) := by
        have : (i : ℕ) + 1 ≤ (j : ℕ) := hlt
        exact_mod_cast this
      nlinarith
    · refine (ico_disj ?_).symm
      have hle : ((j : ℕ) : ℝ) + 1 ≤ ((i : ℕ) : ℝ) := by
        have : (j : ℕ) + 1 ≤ (i : ℕ) := hgt
        exact_mod_cast this
      nlinarith

end PaperCand15
