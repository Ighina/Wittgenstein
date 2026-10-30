"""CAND::8  Example 1 parameters: N=16, QPSK, 5 Msym/s per subcarrier,
symbol period 200 ns, T=18.375 -> subcarrier spacing 5.74 MHz, 160 Mbit/s.

Falsification condition (fixed BEFORE running):
  The paper is wrong if any of the following fails in exact arithmetic:
    (a) 1/(5 Msym/s) != 200 ns;
    (b) N * 5 Msym/s * log2(4) bits != 160 Mbit/s;
    (c) the spacing implied by Eq.(7)'s SN convention,
        Delta_f = (T / T_sym) / N, differs from 5.74 MHz by more than
        0.005 MHz (the rounding granularity of the printed figure);
    (d) N * Delta_f > 100 MHz (the targeted channel bandwidth).
"""
from fractions import Fraction as F

N = 16
Rs = F(5, 1)                 # Msym/s per subcarrier
bits_per_symbol = 2          # QPSK
T = F(147, 8)                # 18.375
B = F(100)                   # MHz channel bandwidth
printed_spacing = F(574, 100)
printed_rate = F(160)
printed_Tsym_ns = F(200)

Tsym_us = 1 / Rs                              # microseconds
print("(a) symbol period = 1/Rs =", Tsym_us, "us =", Tsym_us * 1000, "ns",
      "  printed:", printed_Tsym_ns, "ns  -> equal?", Tsym_us * 1000 == printed_Tsym_ns)

rate = N * Rs * bits_per_symbol               # Mbit/s
print("(b) total rate = N*Rs*2 =", rate, "Mbit/s   printed:", printed_rate,
      " -> equal?", rate == printed_rate)

fs = T / Tsym_us                              # samples/us = MHz
spacing = fs / N
print("(c) sampling rate f_s = T/T_sym =", fs, "MHz =", float(fs))
print("    Delta_f = f_s/N =", spacing, "=", float(spacing), "MHz   printed:",
      float(printed_spacing), "MHz")
print("    |Delta_f - printed| =", float(abs(spacing - printed_spacing)), "MHz",
      " -> within 0.005 MHz?", abs(spacing - printed_spacing) < F(5, 1000))
print("    rounds to 5.74 at 2 decimals?", round(float(spacing), 2) == 5.74)

occupied = N * spacing
print("(d) occupied bandwidth N*Delta_f =", occupied, "MHz =", float(occupied),
      " -> <= 100 MHz?", occupied <= B)
print("    spectral efficiency = rate/B =", float(rate / B), "bit/s/Hz")

a = Tsym_us * 1000 == printed_Tsym_ns
b = rate == printed_rate
c = abs(spacing - printed_spacing) < F(5, 1000)
d = occupied <= B
print(f"\n(a)={a} (b)={b} (c)={c} (d)={d}")
print("VERDICT: MATCH" if (a and b and c and d) else "VERDICT: MISMATCH")
