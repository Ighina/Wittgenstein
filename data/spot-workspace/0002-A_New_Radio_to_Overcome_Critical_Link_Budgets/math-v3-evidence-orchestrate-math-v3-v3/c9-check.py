"""CAND::9 (revision 1)  Example 2 parameters: N=32, QPSK, 2 Msym/s per
subcarrier, symbol period 500 ns, T=46 -> subcarrier spacing 2.88 MHz,
128 Mbit/s; plus the requirement that ONE spacing formula reproduce BOTH
examples.

Revision note: revision 0 tested |Delta_f - printed| < 0.005 MHz and flagged
example 2 because Delta_f = 2.875 is exactly 0.005 from the printed 2.88.
The printed figure carries two decimals, so the faithful test against a
ROUNDED printed value is round(Delta_f, 2) == printed, which is what this
revision uses.  Both tests are reported.

Falsification condition (unchanged in substance):
  Wrong if (a) 1/(2 Msym/s) != 500 ns; (b) 32*2*2 != 128 Mbit/s;
  (c) Delta_f = T*Rs/N does not round to the printed spacing at the printed
      precision (2 decimals); (d) N*Delta_f > 100 MHz; or (e) the formula that
      reproduces example 2 is not the same one that reproduces example 1.
"""
from fractions import Fraction as F

def spacing(T, N, Rs):          # Rs in Msym/s -> MHz
    return (T * Rs) / N

cases = [("example 1", F(147, 8), 16, F(5), F(574, 100), F(160), F(200)),
         ("example 2", F(46),     32, F(2), F(288, 100), F(128), F(500))]

allok = True
for name, T, N, Rs, printed_sp, printed_rate, printed_Tsym in cases:
    Tsym_ns = 1000 / Rs
    rate = N * Rs * 2
    sp_ = spacing(T, N, Rs)
    occ = N * sp_
    a = Tsym_ns == printed_Tsym
    b = rate == printed_rate
    c = round(float(sp_), 2) == float(printed_sp)
    c_strict = abs(sp_ - printed_sp) < F(5, 1000)
    d = occ <= 100
    allok &= (a and b and c and d)
    print(f"-- {name}: N={N}, T={T} ({float(T)}), Rs={Rs} Msym/s --")
    print(f"   symbol period     = {float(Tsym_ns)} ns  (printed {float(printed_Tsym)})  ok={a}")
    print(f"   total rate        = {float(rate)} Mbit/s (printed {float(printed_rate)}) ok={b}")
    print(f"   f_s = T*Rs        = {float(T*Rs)} MHz")
    print(f"   Delta_f = f_s/N   = {sp_} = {float(sp_)} MHz (printed {float(printed_sp)})")
    print(f"     rounds to printed value at 2 decimals? {c}")
    print(f"     |Delta_f - printed| = {float(abs(sp_ - printed_sp))} MHz "
          f"(< 0.005 strictly? {c_strict})")
    print(f"   occupied = N*Df   = {float(occ)} MHz <= 100  ok={d}")
    print()

print("One and the same formula Delta_f = T*Rs/N reproduces both printed "
      "spacings at the printed precision:", allok)
print("VERDICT: MATCH" if allok else "VERDICT: MISMATCH")
