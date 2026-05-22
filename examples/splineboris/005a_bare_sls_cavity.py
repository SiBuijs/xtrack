"""
Bare SLS lattice from sls.madx with main RF and third-harmonic cavities.

Cavity parameters follow the SLS / SLS 2.0 TDR nominal values. Placement uses
the original SLS straight labels in sls.madx (short straights ARS02 / ARS08
for the four 500 MHz cavities, ARS07 for the Super-3HC). SLS 2.0 relocates
main RF to X05 and the 3HC to X09; change the anchor markers below if needed.
"""

import xtrack as xt
from pathlib import Path
from scipy.constants import c as clight

E0 = 2.7e9  # eV

# --- RF parameters (TDR nominal) ---
# Main 500 MHz: 4 × 445 kV → 1.78 MV total, split over X02 and X08 (890 kV each)
V_RF_MAIN_STRAIGHT = 890e3  # V per short straight (2 cavities × 445 kV)
F_RF_MAIN = 499.6537e6  # Hz
# TDR φ_s ≈ 57°; xtrack Cavity.lag is offset by +180° from that convention
LAG_RF_MAIN = 57.0 + 180.0  # deg

# Third-harmonic (Super-3HC): passive cavity, design-point effective voltage
V_RF_3HC = 540e3  # V
F_RF_3HC = 1498.95e6  # Hz (= 3 × f_main)
LAG_RF_3HC = 90.0 + 180.0  # deg (TDR φ_3HC,s ≈ π/2 + 180° for xtrack)

# Lattice markers for cavity placement (original SLS in sls.madx)
MAIN_RF_ANCHORS = ('ars02_gsrc_0500', 'ars08_gsrc_0500')  # short straights X02, X08
HC3_ANCHOR = 'ars07_gsrc_0390'  # medium straight X07 (Super-3HC in SLS)

p0 = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1, p0c=E0)

madx_file = (
    Path(__file__).resolve().parent.parent.parent / 'test_data' / 'sls' / 'sls.madx'
)
env = xt.load(str(madx_file))
line = env.ring
line.configure_bend_model(core='mat-kick-mat')
line.particle_ref = p0.copy()

tt = line.get_table()

# Env knobs (shared by all cavities of each type)
env['vrf_main_straight'] = V_RF_MAIN_STRAIGHT
env['frf_main'] = F_RF_MAIN
env['lag_main'] = LAG_RF_MAIN
env['vrf_3hc'] = V_RF_3HC
env['frf_3hc'] = F_RF_3HC
env['lag_3hc'] = LAG_RF_3HC

env.new(
    'cav_main_x02',
    xt.Cavity,
    voltage='vrf_main_straight',
    frequency='frf_main',
    lag='lag_main',
)
env.new(
    'cav_main_x08',
    xt.Cavity,
    voltage='vrf_main_straight',
    frequency='frf_main',
    lag='lag_main',
)
env.new(
    'cav_3hc_x07',
    xt.Cavity,
    voltage='vrf_3hc',
    frequency='frf_3hc',
    lag='lag_3hc',
)

line.insert([
    env.place('cav_main_x02', at=f'{MAIN_RF_ANCHORS[0]}@start'),
    env.place('cav_main_x08', at=f'{MAIN_RF_ANCHORS[1]}@start'),
    env.place('cav_3hc_x07', at=f'{HC3_ANCHOR}@start'),
])

# Check h = 480 from circumference and RF frequency
L = tt['s'][-1]
beta0 = p0.beta0[0]
f_rev = beta0 * clight / L
h_check = F_RF_MAIN / f_rev

print('=' * 80)
print('SLS BARE LATTICE WITH RF CAVITIES')
print('=' * 80)
print(f'Circumference L = {L:.6f} m')
print(f'f_rev = {f_rev / 1e6:.6f} MHz')
print(f'h (from f_RF) = {h_check:.4f}  (TDR: 480)')
print()
print('Cavity placement:')
for name in (*MAIN_RF_ANCHORS, HC3_ANCHOR):
    print(f'  {name:22s}  s = {tt["s", name]:.6f} m')
print()
print('RF settings:')
print(f'  Main: V_straight = {V_RF_MAIN_STRAIGHT / 1e3:.1f} kV × 2  '
      f'(total {2 * V_RF_MAIN_STRAIGHT / 1e6:.3f} MV), f = {F_RF_MAIN / 1e6:.4f} MHz, '
      f'lag = {LAG_RF_MAIN:.1f}°')
print(f'  3HC:  V = {V_RF_3HC / 1e3:.1f} kV, f = {F_RF_3HC / 1e6:.4f} MHz, lag = {LAG_RF_3HC:.1f}°')
print()

line.configure_radiation(model='mean')
line.build_tracker()

tw = line.twiss(radiation_integrals=True, spin=True, polarization=True, radiation_method='full')

print('Tunes:')
print(f'  qx = {tw.qx:.6f}')
print(f'  qy = {tw.qy:.6f}')
print(f'  qs = {tw.qs:.6f}')
print()
print(f'Chromaticity:  dqx = {tw.dqx:.4e},  dqy = {tw.dqy:.4e}')
print()
print('Radiation integrals / damping [1/s]:')
print(f'  alpha_x   = {tw.rad_int_damping_constant_x_s:.4e}')
print(f'  alpha_y   = {tw.rad_int_damping_constant_y_s:.4e}')
print(f'  alpha_z   = {tw.rad_int_damping_constant_zeta_s:.4e}')
print()
print(f'Equilibrium emittances [m]:  ex = {tw.rad_int_eq_gemitt_x:.4e},  '
      f'ey = {tw.rad_int_eq_gemitt_y:.4e}')
print(f'Spin polarization: {tw.spin_polarization_eq:.4e}')
print('=' * 80)

import matplotlib.pyplot as plt

plt.close('all')
tw.plot('x y')
tw.plot('betx bety', 'dx dy')
tw.plot('delta')
plt.show()
