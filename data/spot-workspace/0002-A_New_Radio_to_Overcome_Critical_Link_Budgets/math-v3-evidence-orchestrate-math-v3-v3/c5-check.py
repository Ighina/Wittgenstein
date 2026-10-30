"""CAND::5  Second example: "The combining gain is 15 dB, exactly the same as
for transmit beamforming", with N=32 and T=46.

Falsification condition (fixed BEFORE running):
  (a) "exactly the same as for transmit beamforming" is wrong iff
      10*log10(N) = 10*log10(32) differs from 15 dB by more than 0.1 dB.
  (b) the figure contradicts the paper's T-fold argument iff 15 dB EXCEEDS the
      T-fold ceiling 10*log10(T) = 10*log10(46).  (The T-fold statement is an
      upper bound on achievable gain, so falling short of it is not a
      contradiction; exceeding it is.)
  Also cross-check the first example's three printed dB figures the same way.
"""
import math

def dB(x): return 10.0 * math.log10(x)

print("-- example 2: N=32, T=46, FFT receiver with rectangular windowing --")
bf32, ceil46 = dB(32), dB(46)
print(f"  beamforming gain 10log10(32) = {bf32:.4f} dB   (paper: 15 dB)")
print(f"  T-fold ceiling   10log10(46) = {ceil46:.4f} dB")
print(f"  |15 - 10log10(32)| = {abs(15 - bf32):.4f} dB  -> matches 15 dB to 0.1 dB? {abs(15-bf32) < 0.1}")
print(f"  15 dB exceeds the T-fold ceiling? {15 > ceil46}")
print(f"  headroom left unused: {ceil46 - 15:.4f} dB")

print("\n-- example 1: N=16, T=18.375 --")
bf16, ceil18 = dB(16), dB(18.375)
print(f"  beamforming gain 10log10(16)     = {bf16:.4f} dB   (paper: 12.0 dB)")
print(f"  T-fold ceiling   10log10(18.375) = {ceil18:.4f} dB")
for label, g in [("avg matched filter, rect pulse", 12.6),
                 ("FFT receive filter, rect pulse", 12.2),
                 ("avg matched filter, sine pulse", 12.5),
                 ("FFT receive filter, sine pulse", 12.1)]:
    print(f"  {label:34s} {g} dB -> exceeds T-fold ceiling? {g > ceil18}"
          f"   exceeds beamforming? {g > bf16}")

ok_a = abs(15 - bf32) < 0.1
ok_b = 15 <= ceil46
ok_c = all(g <= ceil18 for g in (12.6, 12.2, 12.5, 12.1))
print(f"\n(a) 15 dB == 10log10(32): {ok_a}")
print(f"(b) 15 dB within the T=46 ceiling: {ok_b}")
print(f"(c) all example-1 figures within the T=18.375 ceiling: {ok_c}")
print("VERDICT: MATCH" if (ok_a and ok_b and ok_c) else "VERDICT: MISMATCH")
