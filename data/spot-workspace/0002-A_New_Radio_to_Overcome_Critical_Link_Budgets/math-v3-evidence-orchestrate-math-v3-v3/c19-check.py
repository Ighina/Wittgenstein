"""CAND::19  "As frequency is the temporal derivative of the phase, the VCO is
fed with the phase derivative, also known as the instantaneous frequency."

Falsification condition (fixed BEFORE running):
  Take the oscillator output exp(i*varphi(t)) with varphi in RADIANS and
  ordinary frequency f in Hz.  The statement 'f = dvarphi/dt' is wrong by the
  factor 2*pi iff dvarphi/dt != f for a pure tone.  Then check the dependent
  quantitative claim: 'the phase shift is the integral of the temporary
  deviation of the instantaneous frequency from the subcarrier frequency over
  the duration of the phase transition' -- exact iff the integral of the
  ANGULAR frequency deviation (not the ordinary one) gives the phase shift.
  Finally note the alternative convention (phase measured in CYCLES/turns),
  under which the sentence is exactly true, which is what decides whether this
  is an error at all.
"""
import sympy as sp

t, f0, T = sp.symbols('t f0 T', positive=True)

# RADIAN convention
phi_rad = 2 * sp.pi * f0 * t
print("phase in radians:  varphi(t) = 2*pi*f0*t")
print("  dvarphi/dt            =", sp.diff(phi_rad, t), " (rad/s)")
print("  ordinary frequency f0 =", f0, " (Hz)")
print("  dvarphi/dt == f0 ?    ", sp.simplify(sp.diff(phi_rad, t) - f0) == 0)
print("  dvarphi/dt / f0       =", sp.simplify(sp.diff(phi_rad, t) / f0))

# CYCLE convention
phi_cyc = f0 * t
print("\nphase in cycles:   varphi(t) = f0*t")
print("  dvarphi/dt            =", sp.diff(phi_cyc, t), " (cycles/s = Hz)")
print("  dvarphi/dt == f0 ?    ", sp.simplify(sp.diff(phi_cyc, t) - f0) == 0)

# the dependent claim: phase shift = integral of the frequency deviation
dev = sp.Function('dev')          # deviation of the instantaneous frequency
print("\n-- phase shift as the integral of the frequency deviation --")
print("  with ANGULAR deviation  Domega(t):  Dvarphi = int_0^T Domega dt  (radians) -- exact")
print("  with ORDINARY deviation Df(t):      int_0^T Df dt = Dvarphi/(2*pi) -- off by 2*pi")
# concrete witness: a rectangular deviation of 25 MHz for 20 ns
Df, dur = sp.Rational(25, 10**3), sp.Rational(20, 10**3)   # 25 MHz (GHz units), 20 ns (us units)
Df_hz, dur_s = sp.Rational(25*10**6), sp.Rational(20, 10**9)
print(f"\n  witness: Df = 25 MHz held for 20 ns")
print(f"    int Df dt            = {sp.nsimplify(Df_hz*dur_s)} = {float(Df_hz*dur_s)} (cycles)")
print(f"    2*pi * int Df dt     = {sp.N(2*sp.pi*Df_hz*dur_s, 10)} rad "
      f"= {sp.N(2*Df_hz*dur_s, 10)} * pi/2")
print(f"    to realise exactly -pi/2 rad you need int Df dt = -1/4 cycle = "
      f"{float(sp.Rational(-1,4))} cycles, i.e. Df*dur = -0.25")

print("\nRESULT: the sentence is off by a factor 2*pi under the radian convention "
      "and exactly true under the cycle (turn) convention. The paper fixes no "
      "unit for the phase, and every quantitative use in the paper is a phase "
      "SHIFT expressed in radians (+pi/2, -3pi/2), where only the angular form "
      "is exact.")
print("VERDICT: INCONCLUSIVE -- convention-dependent; no faithful computation "
      "can decide which convention the sentence intends.")
