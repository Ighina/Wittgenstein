"""CAND::10  "At 70 MHz, i.e. 40% above the edge of the channel band".

Falsification condition (fixed BEFORE running):
  The channel bandwidth is 100 MHz centred on the carrier (Fig. 9's axis runs
  -75..+75 MHz), so the band edge is at 50 MHz.  The phrase is wrong iff
  (70 - 50)/50 != 0.40 exactly.  Also report the alternative references in case
  a different one was meant.
"""
from fractions import Fraction as F

B = F(100)          # MHz
edge = B / 2        # 50 MHz
f = F(70)           # MHz

rel_edge = (f - edge) / edge
print("channel bandwidth      =", B, "MHz, centred -> band edge at", edge, "MHz")
print("(70 - 50)/50           =", rel_edge, "=", float(rel_edge) * 100, "%")
print("   equals 40% exactly? ", rel_edge == F(40, 100))
print()
print("alternatives, for completeness:")
print("  (70-50)/100 =", float((f - edge) / B * 100), "%  (relative to full bandwidth)")
print("  70/50       =", float(f / edge), " -> 70 MHz is", float(f/edge*100), "% of the edge")
print("  (70-50)/70  =", float((f - edge) / f * 100), "%")
print()
print("VERDICT: MATCH" if rel_edge == F(40, 100) else "VERDICT: MISMATCH")
