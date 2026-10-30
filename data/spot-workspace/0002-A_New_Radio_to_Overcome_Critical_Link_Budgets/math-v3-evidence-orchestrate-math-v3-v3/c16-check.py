"""CAND::16  "coherent combining of N^2 entries at the receiver to N data
symbols can result in an N-fold gain in received power" (transmit beamforming,
Eq.(1)-(2)), versus the generalisation's "T-fold improvement of received power"
for a transmit matrix with TN entries and N symbols.

Falsification condition (fixed BEFORE running):
  Compute the received power gain of Eq.(2) exactly, under both normalisations,
  for N = 2,4,8,16,32,64 and unit-modulus phase-only phi = h^dagger/|h|:
      G_per_antenna = |h phi|^2 / |h_0|^2            (one antenna at power P)
      G_total_power = |h phi|^2 / (N * |h_0|^2)      (same TOTAL radiated power)
  The paper's N-fold figure is wrong under EVERY normalisation iff neither
  equals N.  It is also checked against the dB numbers the paper prints
  (12.0 dB for N=16, 15 dB for N=32) and against the entry-counting rule
  (#entries / #symbols = N^2/N = N for beamforming, TN/N = T in general).
"""
import math
from fractions import Fraction as F

print(f"{'N':>4} {'|h phi|^2':>12} {'G_per_antenna':>14} {'G_total_power':>14} "
      f"{'entries/symbols':>16} {'10log10(N)':>11}")
ok_total, ok_count = True, True
for N in (2, 4, 8, 16, 32, 64):
    # |h_n| = 1 for all n (equal-gain channel), phi_n = conj(h_n) -> h phi = N
    hphi2 = F(N) ** 2
    g_per = hphi2                      # relative to a single antenna at power P
    g_tot = hphi2 / N                  # relative to the same total radiated power
    cnt = F(N * N, N)                  # N^2 entries / N data symbols
    ok_total &= (g_tot == N)
    ok_count &= (cnt == N)
    print(f"{N:>4} {float(hphi2):>12.0f} {float(g_per):>14.0f} {float(g_tot):>14.0f} "
          f"{float(cnt):>16.0f} {10*math.log10(N):>11.4f}")

print()
print("G_total_power == N for every N checked:", ok_total)
print("entries/symbols == N for every N checked:", ok_count)
print("G_per_antenna == N for any N > 1?", any(F(N)**2 == N for N in (2,4,8,16,32,64)))
print()
print("-- consistency with the printed dB figures --")
for N, printed in ((16, 12.0), (32, 15.0)):
    print(f"  N={N}: 10log10(N) = {10*math.log10(N):.4f} dB  printed {printed} dB  "
          f"match to 0.1 dB? {abs(10*math.log10(N)-printed) < 0.1}")
print("  (an N^2-fold reading would print "
      f"{20*math.log10(16):.2f} dB for N=16 and {20*math.log10(32):.2f} dB for N=32, "
      "which the paper does NOT print)")
print()
print("-- the generalisation: TN entries / N symbols = T --")
for N, T in ((16, 18.375), (32, 46)):
    print(f"  N={N}, T={T}: TN/N = {T*N/N} = T,  10log10(T) = {10*math.log10(T):.4f} dB")

print()
print("VERDICT: MATCH -- the N-fold figure is exact under the fixed-total-radiated-"
      "power normalisation, which is the normalisation the paper's own dB figures "
      "(12.0 dB at N=16, 15 dB at N=32) use, and it agrees with the entry-counting "
      "rule that also yields the T-fold generalisation."
      if (ok_total and ok_count) else "VERDICT: MISMATCH")
