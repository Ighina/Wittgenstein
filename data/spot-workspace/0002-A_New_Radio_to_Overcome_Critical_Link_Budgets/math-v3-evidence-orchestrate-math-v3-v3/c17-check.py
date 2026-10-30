"""CAND::17  "Each of the N antenna elements operates only at about (1/N)th of
the channel bandwidth."

Falsification condition (fixed BEFORE running):
  The nominal per-antenna bandwidth is the subcarrier spacing Delta_f.
  The claim is wrong iff Delta_f / (B/N) is not "about 1" -- taken as outside
  [0.5, 2] -- for either numerical example (B = 100 MHz).
  SECOND, SEPARATE sub-claim (the one about transition sidelobes): the occupied
  per-antenna bandwidth including the data-dependent transition of Eq.(7) is
  NOT computed here, because it needs the unspecified sequences q_{n,s,d}.
  That part is reported as unchecked.
"""
from fractions import Fraction as F

B = F(100)
cases = [("example 1", 16, F(735, 128)), ("example 2", 32, F(23, 8))]
ok = True
for name, N, df in cases:
    nominal = B / N
    ratio = df / nominal
    inrange = F(1, 2) <= ratio <= 2
    ok &= inrange
    print(f"-- {name}: N={N} --")
    print(f"   subcarrier spacing Delta_f = {df} = {float(df):.6f} MHz")
    print(f"   B/N                        = {nominal} = {float(nominal):.6f} MHz")
    print(f"   Delta_f / (B/N)            = {ratio} = {float(ratio):.6f}")
    print(f"   within [0.5, 2] -> 'about (1/N)th'? {inrange}")
    print(f"   N * Delta_f = {float(N*df):.4f} MHz of the {float(B)} MHz channel "
          f"({float(N*df/B):.2%})")
    print()

print("NOT CHECKED: the occupied bandwidth of a single antenna INCLUDING the "
      "data-dependent transition waveform q_{n,s,d} of Eq.(7); the paper leaves "
      "those sequences unspecified ('to be optimized'), so no faithful "
      "computation of the per-antenna sideband energy is possible here.")
print()
print("VERDICT: MATCH (nominal reading)" if ok else "VERDICT: MISMATCH")
