"""CAND::18  The reported dB figures: 9.7 / 18.3 dB (rect pulse) and
9.7 / 20.9 dB (squared sine pulse), with the latter called "a minor improvement".

Falsification condition (fixed BEFORE running):
  Only the ARITHMETIC is computable.  The claim is arithmetically wrong iff the
  printed dB differences are internally impossible, e.g. a negative power ratio
  or a figure exceeding the T-fold combining ceiling.  Whether the repeated
  9.7 dB is a coincidence or a transcription slip, and whether +2.6 dB counts as
  "minor", are NOT decidable by computation -- stated here so the verdict is not
  read as settling them.
"""
import math

print("rect pulse  (RC cut-off 50 MHz):  avg-MF 9.7 dB, FFT-RX 18.3 dB  (called SIR, no noise added)")
print("sine pulse  (RC cut-off 70 MHz):  avg-MF 9.7 dB, FFT-RX 20.9 dB  (called SINR, alpha=15)")
print()
d = 20.9 - 18.3
print(f"FFT-RX improvement = 20.9 - 18.3 = {d:.4f} dB")
print(f"  as a power ratio  = 10^({d:.1f}/10) = {10**(d/10):.4f}x")
print(f"  interference power reduction = {100*(1-10**(-d/10)):.2f}%")
print(f"avg-MF difference  = 9.7 - 9.7 = {9.7-9.7:.4f} dB (identical to the printed precision)")
print()
print("arithmetic sanity: all four figures positive and finite?",
      all(x > 0 and math.isfinite(x) for x in (9.7, 18.3, 9.7, 20.9)))
ceil_ = 10*math.log10(18.375)
print(f"T-fold combining ceiling 10log10(18.375) = {ceil_:.4f} dB "
      "(an SIR is not bounded by it, so no constraint is violated)")
print()
print("NOT DECIDABLE BY COMPUTATION:")
print("  (i) whether the avg-MF figure being bit-for-bit 9.7 dB under two "
      "different pulse shapes and two different RC cut-offs is a coincidence, "
      "a genuine insensitivity, or a transcription slip -- the simulation inputs "
      "(the sequences q_{n,s,d}, the exact windowing) are not given;")
print("  (ii) whether +2.6 dB (a 1.82x reduction of interference power) is "
      "fairly described as 'a minor improvement' -- that is a judgement, not an "
      "arithmetic fact;")
print("  (iii) whether labelling one figure SIR and the other SINR, with no "
      "Gaussian noise added in either case, means the two are the same quantity.")
print()
print("VERDICT: INCONCLUSIVE -- the printed figures are arithmetically "
      "self-consistent, and the suspected defect is not of a kind a computation "
      "can exhibit.")
