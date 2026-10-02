# File Structure:

## Starting with 000:

These are files that study the local compensation scheme. It's mainly to reproduce the results from [Manuela Boscolo's paper](https://cds.cern.ch/record/2948247/files/document.pdf) for the local scheme.

`000a` generates a `SolenoidField` for the main solenoid and for two compensation solenoids, using the analytic expression for a thin cylindrical solenoid of finite length. It transforms the $x$ and $s$ coordinates to account for the half crossing angle of $15\ \mathrm{mrad}$, evaluates the field along the beam trajectory, and rotates the field back to the beam frame. It then creates a solenoid consisting of 200 `VariableSolenoid` elements, named `sol_slice_{ii}_{ip_name}`, where `ii` runs from 0 to 199 and `ip_name` denotes the IP in question. The fields of the main and compensation solenoids are superimposed on these same slices. It then scales the compensation solenoids such that the net field integral vanishes;
$$\int B_s(s)\ \mathrm{d}s = 0$$
Finally, it installs dipole correctors: one pair spread over the slices at the location of the compensation solenoids, and one thin corrector on each side next to the `dy_match_l_{ip_name}` and `dy_match_r_{ip_name}` markers at $\pm 11.95\ \mathrm{m}$ from the IP.

`000b` uses the dipole correctors from `000a`, plus extra dipole kicks on QD0 and QF1, to correct the orbit and vertical dispersion to zero at $\pm 11.95\ \mathrm{m}$ from the IP. The optics and horizontal dispersion are corrected with trims on the existing quadrupoles QD0 to QD6, and matched to the unperturbed optics at the ends of the straight section (~1.4 km from the IP). The orbit and optics corrections are iterated a few times and stored as knobs, which are all controlled by a single knob `on_sol_corr_{ip_name}`. The script also rotates the doublets, but since the compensation solenoids cancel the net field integral before the doublet, the rotation angle is zero here.

## Starting with 001:

These files study the non-local compensation scheme from the same paper. The structure is the same as for 000, so I only describe the differences.

`001a` generates the main solenoid with a `TiltedSolenoid` (from `tilted_solenoid.py`), which wraps `SolenoidField` and does the same coordinate and field transformation for the $15\ \mathrm{mrad}$ half crossing angle as in `000a`. The main solenoid is now $3\ \mathrm{T}$ with a half length of $1.3\ \mathrm{m}$, and is again sliced into 200 `VariableSolenoid` elements named `sol_slice_{ii}_{ip_name}`. The compensation solenoids are no longer superimposed on these slices. Instead, they are separate solenoids of 50 `VariableSolenoid` slices each, placed between $12$ and $14\ \mathrm{m}$ from the IP on either side, so outside the final doublet. They are again scaled such that the net field integral vanishes;
$$\int B_s(s)\ \mathrm{d}s = 0$$
The dipole correctors and the `dy_match_l_{ip_name}` and `dy_match_r_{ip_name}` markers at $\pm 11.95\ \mathrm{m}$ are installed as in `000a`, except that the correctors overlaid on the main solenoid now start at $1.4\ \mathrm{m}$ instead of $1.23\ \mathrm{m}$.

`001b` is essentially identical to `000b`. The important difference is that the doublet now sits inside the rotated frame of the solenoid, since the compensation solenoids are further out. The final doublet quadrupoles on each side are therefore rotated by half the rotation angle of the main solenoid, to compensate the coupling. The orbit, vertical dispersion, optics and horizontal dispersion are then corrected and matched in the same way as in `000b`, and everything, including the compensation solenoids and the doublet rotation, is again controlled by the single knob `on_sol_corr_{ip_name}`.

`002` analyses the corrected non-local lattice from `001b` (the local lattice from `000b` can be selected by changing `LATTICE_JSON` at the top of the script).

## 003:

`003` is a short exploratory script that looks at the non-linear components of the tilted solenoid field. It builds the same main solenoid as in `000a` ($2\ \mathrm{T}$, half length $1.23\ \mathrm{m}$, radius $13\ \mathrm{cm}$, $15\ \mathrm{mrad}$ half crossing angle) using `TiltedSolenoid`, and evaluates it along the beam axis for $s$ between $-2.5$ and $2.5\ \mathrm{m}$.

It then computes the transverse derivatives of the field with `compute_pure_field_derivatives`, which uses a 9-point finite difference with a step of $0.1\ \mathrm{mm}$. It takes the derivatives of $B_y$ and $B_x$ with respect to $x$ up to fourth order, and normalises them by the beam rigidity at $45.6\ \mathrm{GeV}$ to get the equivalent normal and skew multipole strengths along the solenoid, respectively;
$$k_n(s) = \frac{1}{B\rho}\frac{\partial^n B_y}{\partial x^n}, \qquad k_n^{s}(s) = \frac{1}{B\rho}\frac{\partial^n B_x}{\partial x^n}$$
for $n = 0, \dots, 4$. As a sanity check, it compares the skew quadrupole component $k_1^s$ to a simple central difference.

## Starting with 004:

These files are the current version of the non-local scheme from `001`, with a more accurate model of the solenoid fields. The shared parameters are kept in `solenoid_params.py`, and each script takes a `--b0` argument to choose between the $2\ \mathrm{T}$ case (half length $1.23\ \mathrm{m}$, as in the paper) and the $3\ \mathrm{T}$ case (half length $1.3\ \mathrm{m}$). Most output files are tagged with the field strength, e.g. `_2T` or `_3T`.

`004a` builds the main solenoid (`TiltedSolenoid`, $15\ \mathrm{mrad}$ half crossing angle) and the compensation solenoid (`SolenoidField`, length $1.5\ \mathrm{m}$, radius $3\ \mathrm{cm}$) as in `001a`. Besides the field on the axis, it now also computes the transverse derivatives of $B_x$ and $B_y$ up to fourth order, in the same way as in `003`. All fields are multiplied by a smooth taper of $15\ \mathrm{cm}$ at both ends, so they go smoothly to zero at the edges of the field map, where the solenoid joins the drifts. This is done to preserve symplecticity. The compensation solenoids are again scaled such that the net field integral vanishes;
$$\int B_s(s)\ \mathrm{d}s = 0$$
From this, it builds two models of each solenoid:
- A `SplineBoris` model, with 200 slices for both the main and the compensation solenoid. In each slice, $B_s$ and the multipole components of $B_x$ and $B_y$ are described by polynomials in $s$, of fourth order for $B_s$ and the dipole component and decreasing in order for the higher multipoles, and particles are tracked through the field with a Boris integrator. This model includes the non-linear components of the field, up to the order set by `--max-transverse-order` (default 4, decapole).
- A `VariableSolenoid` model, as in `000` and `001`, which only contains the solenoid field and the dipole components.

Both models are saved to `004_solenoid_lines_{field_tag}.json`. The rest of the script checks the models: it compares the fields of the `SplineBoris` model with the field map, checks that the map is symplectic, and twisses a short line with the main solenoid and the two compensation solenoids to look at the orbit and coupling.

`004b_install_splineboris_solenoids_in_fcc_ring` installs the `SplineBoris` model in the ring at all four IPs. The main solenoid is centred on the IP, and the compensation solenoids are placed between $12$ and $14\ \mathrm{m}$ from the IP, as in `001a`. The main solenoid is switched on by the knob `on_sol_{ip_name}`, and both compensation solenoids by `on_comp_sol_{ip_name}`. The dipole correctors overlaid on the main solenoid, the thin correctors and the `dy_match_l_{ip_name}` and `dy_match_r_{ip_name}` markers at $\pm 11.95\ \mathrm{m}$ are also installed as in `001a`. `004b_install_varsol_solenoids_in_fcc_ring` does the same for the `VariableSolenoid` model.

`004c` does the correction, and works for both models (`--model splineboris` or `--model varsol`). It follows `001b`, with a few important changes:
- The correction is split into two independent halves, one on each side of the IP. Each half starts from the unperturbed optics at the IP, and is matched to the unperturbed optics at the end of the straight section, about $1.4\ \mathrm{km}$ away.
- The doublets are again rotated by half the rotation angle of the main solenoid, and the orbit and vertical dispersion are corrected to zero at $\pm 11.95\ \mathrm{m}$ as before.
- For the optics, each of the six bends closest to the IP (three on each side) is cut in half, and a thin trim quadrupole is placed in the middle. These are used together with trims on the existing quadrupoles QD0 to QD6. Besides the optics and horizontal dispersion at the end of the straight section, the match also targets $\beta_x$ and $\beta_y$ at the chromatic sextupole closest to the IP (`sdm1`).
- There is now also a coupling correction, using skew quadrupole correctors on all quadrupoles in each half of the straight section. It sets $\beta_{x2}$, $\beta_{y1}$, $\alpha_{x2}$, $\alpha_{y1}$, $D_y$ and $D_{py}$ to zero at the end of the straight section.

The orbit, optics and coupling corrections are iterated a few times, and everything is again controlled by the single knob `on_sol_corr_{ip_name}`. At the end, the script prints the first and second order chromaticity of the ring before and after the correction, and saves the corrected lattice as `fccee_z_lcc_{model}_solenoids_coupling_corrected_{field_tag}.json`.

`004d` analyses the corrected `SplineBoris` lattice. It computes the optics with the solenoids off, with the solenoids and corrections on, and with the solenoids on but the correctors in the straight section switched off. It also does a radiation and spin analysis. It plots the orbit, vertical dispersion, coupling ($\beta_{x2}/\beta_x$ and $\beta_{y1}/\beta_y$), beta-beat, energy loss and spin around the IP and over the full straight section, and compares the $2\ \mathrm{T}$ and $3\ \mathrm{T}$ cases. The coupling resonance driving terms $f_{1001}$ and $f_{1010}$ are computed from the Edwards-Teng decoupling of the one-turn map. The figures are saved to `PLOT_DIR`, which can be set with the environment variable `FCC_SOLENOID_PLOT_DIR`.


## Lattices:

All lattices are for FCC-ee at the Z energy, and contain the ring as the line `fccee_p_ring`. The corrected lattices are saved with the main solenoids and all corrections switched on (`on_sol_{ip_name} = 1` and `on_sol_corr_{ip_name} = 1` at all four IPs), and both can be switched off again with these knobs.

`fccee_z_lcc.json` is the bare FCC-ee lattice, without solenoids. It is the starting point for `000a`, `001a` and `004b`.

`fccee_z_lcc_local_solenoid.json` is the lattice with the local compensation scheme, made by `000a` and `000b`.

`fccee_z_lcc_non_local_solenoid.json` is the lattice with the non-local compensation scheme, made by `001a` and `001b`.

`004_solenoid_lines_2T.json` and `004_solenoid_lines_3T.json` only contain the isolated `SplineBoris` and `VariableSolenoid` models of the main and compensation solenoids, made by `004a` and installed in the ring by `004b`.

`fccee_z_lcc_splineboris_solenoids_coupling_corrected_{2T,3T}.json` are the corrected lattices with the `SplineBoris` model, made by `004b_install_splineboris_solenoids_in_fcc_ring` and `004c`. These are the lattices used by `004d`.

`fccee_z_lcc_varsol_solenoids_coupling_corrected_{2T,3T}.json` are the same, but with the `VariableSolenoid` model, made by `004b_install_varsol_solenoids_in_fcc_ring` and `004c --model varsol`.

The installation scripts (`000a`, `001a`, `004b`) also write intermediate lattices starting with `temp_`, which are read by the next script. These are included, so the correction scripts (`000b`, `001b`, `004c`) can be rerun without first rerunning the installation.


# Knobs in the 004 Files

Every knob below exists once per IP, where `{ip_name}` is one of `ipa`, `ipd`, `ipg` or `ipj`, and `{side}` is `left` or `right` (upstream or downstream of the IP). In the corrected lattices all on/off knobs are set to 1.

## Main switches

- `on_sol_{ip_name}` switches the main detector solenoid on (1) or off (0). It scales the field of all slices of the main solenoid, and is independent of the correction.
- `on_sol_corr_{ip_name}` switches the full compensation of that IP on or off: the compensation solenoids, the doublet rotation and all orbit, optics and coupling corrections. It is meant to be switched together with `on_sol_{ip_name}`.

## Knobs driven by `on_sol_corr_{ip_name}`

These are defined as expressions of `on_sol_corr_{ip_name}`, so they follow it automatically. They can be used to switch off one part of the compensation, but note that assigning a number to them breaks the link to `on_sol_corr_{ip_name}`; assigning the string `'on_sol_corr_{ip_name}'` restores it.

- `on_comp_sol_{ip_name}` switches both compensation solenoids. The scaling that cancels the field integral of the main solenoid is already included in the elements.
- `on_rot_doublet_{side}_{ip_name}` switches the rotation of the final doublet on that side. The rotation angle is `phi_rot_doublet_{ip_name}` (in rad), which `004c` sets to half the rotation angle of the main solenoid.
- `on_sol_orbit_corr_{ip_name}`, `on_sol_optics_corr_{ip_name}` and `on_sol_coupling_corr_{ip_name}` switch the orbit, optics and coupling correction on both sides. Each of them in turn drives the per-side knobs `on_sol_orbit_corr_{side}_{ip_name}`, `on_sol_optics_corr_{side}_{ip_name}` and `on_sol_coupling_corr_{side}_{ip_name}`.

## Corrector strengths

These are the knobs varied by the matches in `004c`. Each one is set to its matched value times the per-side knob of its correction, through an intermediate variable with the suffix `_from_on_sol_{orbit,optics,coupling}_corr_{side}_{ip_name}`.

- `acbh{n}_sol_{side}_{ip_name}` and `acbv{n}_sol_{side}_{ip_name}` are the horizontal and vertical dipole correctors (integrated kick, in rad) of the orbit correction:
  - `n = 1` is spread over the outer part of the main solenoid,
  - `n = 2` to `5` are on QD0A, QD0B, QF1A and QF1B,
  - `n = 6` is the thin corrector next to the `dy_match_l_{ip_name}` or `dy_match_r_{ip_name}` marker.
- `k1_{quad}_sol_corr` are the normal quadrupole trims of the optics correction, added to `k1` of the quadrupoles QD0 to QD6.
- `k1_qbmid_{bend}_sol_corr` are the integrated strengths ($k_1 L$, on `knl[1]`) of the thin trim quadrupoles at the centre of the bends, also part of the optics correction.
- `k1s_{quad}_sol_coupling_corr` are the skew quadrupole trims of the coupling correction, added to `k1s` of all quadrupoles in each half of the straight section.