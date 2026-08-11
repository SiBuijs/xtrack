"""Save and reload DA / MA study results (raw data + PDF plots)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np

from solenoid_params import FIELD_TAG

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
# Override on machines where Path.home() isn't a synced CERNBox folder (e.g.
# remote/headless boxes reached over ssh) via FCC_SOLENOID_PLOT_DIR.
PLOT_DIR = Path(
    os.environ.get(
        "FCC_SOLENOID_PLOT_DIR",
        Path.home() / "cernbox" / "Pictures" / "FCC_Solenoid_Studies",
    )
)

ModelTag = Literal["SB", "VarSol"]
StudyTag = Literal["DA", "MA", "EMIT", "POL", "TUNE"]

BUILD_DEFAULTS = {
    "sexamp": 1.0,
    "theta": -0.015,
    "x_offset": 0.0,
    "y_offset": 0.0,
    "extra_sext_strength": 0.0,
    "sext_window_s_min": None,
    "sext_window_s_max": None,
    "sext_window_order": 2,
}


def format_tag_float(value: float) -> str:
    """Format a float for filename tags, e.g. 2.0 -> '2p0'."""
    if value == int(value):
        text = f"{int(value)}.0"
    else:
        text = f"{value:g}"
    return text.replace(".", "p").replace("-", "m")


def variant_suffix(**overrides: Any) -> str:
    """Return '' or a double-underscore variant tag for non-default build knobs."""
    tags: list[str] = []
    sexamp = overrides.get("sexamp", BUILD_DEFAULTS["sexamp"])
    if sexamp != BUILD_DEFAULTS["sexamp"]:
        tags.append(f"sexamp{format_tag_float(sexamp)}")
    x_offset = overrides.get("x_offset", BUILD_DEFAULTS["x_offset"])
    if x_offset != BUILD_DEFAULTS["x_offset"]:
        tags.append(f"xoff{format_tag_float(x_offset)}")
    y_offset = overrides.get("y_offset", BUILD_DEFAULTS["y_offset"])
    if y_offset != BUILD_DEFAULTS["y_offset"]:
        tags.append(f"yoff{format_tag_float(y_offset)}")
    extra_sext_strength = overrides.get(
        "extra_sext_strength", BUILD_DEFAULTS["extra_sext_strength"])
    if extra_sext_strength != BUILD_DEFAULTS["extra_sext_strength"]:
        tags.append(f"xsext{format_tag_float(extra_sext_strength)}")
    sext_window_s_min = overrides.get(
        "sext_window_s_min", BUILD_DEFAULTS["sext_window_s_min"])
    sext_window_s_max = overrides.get(
        "sext_window_s_max", BUILD_DEFAULTS["sext_window_s_max"])
    if sext_window_s_min is not None or sext_window_s_max is not None:
        sext_window_order = overrides.get(
            "sext_window_order", BUILD_DEFAULTS["sext_window_order"])
        order_suffix = (
            "" if sext_window_order == BUILD_DEFAULTS["sext_window_order"]
            else f"o{sext_window_order}")
        tags.append(
            f"swin{format_tag_float(sext_window_s_min)}"
            f"to{format_tag_float(sext_window_s_max)}{order_suffix}")
    if not tags:
        return ""
    return "__" + "__".join(tags)


RadiationModel = Literal["off", "mean", "quantum"]


def radiation_tag(model: RadiationModel) -> str:
    """Filename-safe tag for the radiation model active during the tracking
    run whose results are being saved, e.g. 'mean' -> 'radmean', 'quantum' ->
    'radquantum', 'off' -> 'radoff'. DA/MA (009/010) expose this as a
    --radiation {off,mean,quantum} CLI flag (default mean: deterministic
    damping, no stochastic kicks); EMIT/POL (014/015) always switch to
    quantum right before the tracking loop that produces the saved data,
    since quantum excitation is the physics being measured there -- tagging
    this in filenames keeps the different regimes from silently colliding."""
    if model not in ("off", "mean", "quantum"):
        raise ValueError(
            f"Unknown radiation model {model!r}, expected 'off', 'mean', or 'quantum'.")
    return f"rad{model}"


def global_xy_limit_tag(global_xy_limit: float) -> str:
    if global_xy_limit >= 1.0:
        value = int(global_xy_limit) if global_xy_limit == int(global_xy_limit) else global_xy_limit
        return f"xylim{value}m"
    if global_xy_limit >= 0.01:
        return f"xylim{global_xy_limit * 100:.0f}cm"
    return f"xylim{global_xy_limit * 1000:.0f}mm"


def make_basename(
    *,
    with_solenoids: bool,
    with_correctors: bool,
    model: ModelTag,
    n_part: int,
    n_turns: int,
    global_xy_limit: float,
    radiation: RadiationModel,
    variant: str = "",
    field_tag: str = FIELD_TAG,
) -> str:
    """Build a filename stem (without study prefix) for NPZ/PDF outputs.

    `field_tag` defaults to solenoid_params.FIELD_TAG (the module-level
    MAIN_SOLENOID_B0), but callers that let the field strength be picked at
    runtime (e.g. via solenoid_params.add_b0_argument) should pass the tag
    matching the lattice actually loaded, so saved filenames don't silently
    mislabel the field strength.

    `radiation` records the model active during the saved tracking run --
    see radiation_tag() above.
    """
    sol = "Sol_On" if with_solenoids else "Sol_Off"
    if with_solenoids and not with_correctors:
        sol += "_Cor_Off"
    return (
        f"{sol}_{model}_{field_tag}_{radiation_tag(radiation)}_{n_part}p_{n_turns}t_"
        f"{global_xy_limit_tag(global_xy_limit)}{variant}"
    )


def make_study_stem(study: StudyTag, *, tag: str = "", **basename_kwargs) -> str:
    """Return full filename stem, e.g. DA_Sol_On_SB_750p_10000t_xylim1m."""
    prefix = f"{study}_{tag}" if tag else study
    return f"{prefix}_{make_basename(**basename_kwargs)}"


def _study_npz_path(stem: str) -> Path:
    return DATA_DIR / f"{stem}.npz"


def _study_pdf_path(stem: str) -> Path:
    return PLOT_DIR / f"{stem}.pdf"


def save_figure_pdf(fig: plt.Figure, stem: str) -> Path:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = _study_pdf_path(stem)
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved plot: {pdf_path}")
    return pdf_path


def save_da_study(
    *,
    out: dict[str, Any],
    tt_init,
    fig: plt.Figure,
    model: ModelTag,
    with_solenoids: bool,
    with_correctors: bool,
    n_turns: int,
    global_xy_limit: float,
    sigma_x: float,
    sigma_y: float,
    nn_y_r: int,
    nn_x_theta: int,
    max_amp_sigma_x: float,
    max_amp_sigma_y: float,
    nemitt_x: float,
    nemitt_y: float,
    n_part: int,
    radiation: RadiationModel = "mean",
    variant: str = "",
    sexamp: float = BUILD_DEFAULTS["sexamp"],
    x_offset: float = BUILD_DEFAULTS["x_offset"],
    y_offset: float = BUILD_DEFAULTS["y_offset"],
    extra_sext_strength: float = BUILD_DEFAULTS["extra_sext_strength"],
    sext_window_s_min: float | None = BUILD_DEFAULTS["sext_window_s_min"],
    sext_window_s_max: float | None = BUILD_DEFAULTS["sext_window_s_max"],
    sext_window_order: int = BUILD_DEFAULTS["sext_window_order"],
    field_tag: str = FIELD_TAG,
) -> tuple[Path, Path]:
    stem = make_study_stem(
        "DA",
        with_solenoids=with_solenoids,
        with_correctors=with_correctors,
        model=model,
        n_part=n_part,
        n_turns=n_turns,
        global_xy_limit=global_xy_limit,
        radiation=radiation,
        variant=variant,
        field_tag=field_tag,
    )
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    particles = out["particles"]
    arrays = dict(
        x_hat=np.asarray(tt_init.x_hat),
        y_hat=np.asarray(tt_init.y_hat),
        y_normalized=np.asarray(tt_init.y_normalized),
        at_turn=np.asarray(particles.at_turn),
        state=np.asarray(particles.state),
        lost=np.asarray(out["lost"]),
        frac_lost=float(out["frac_lost"]),
        at_turn_mean=float(out["at_turn_mean"]),
        sigma_x=float(sigma_x),
        sigma_y=float(sigma_y),
        nn_y_r=int(nn_y_r),
        nn_x_theta=int(nn_x_theta),
        max_amp_sigma_x=float(max_amp_sigma_x),
        max_amp_sigma_y=float(max_amp_sigma_y),
        nemitt_x=float(nemitt_x),
        nemitt_y=float(nemitt_y),
        n_turns=int(n_turns),
        global_xy_limit=float(global_xy_limit),
        with_solenoids=with_solenoids,
        with_correctors=with_correctors,
        model=model,
        n_part=int(n_part),
        radiation=radiation,
        sexamp=float(sexamp),
        x_offset=float(x_offset),
        y_offset=float(y_offset),
        extra_sext_strength=float(extra_sext_strength),
        sext_window_s_min=sext_window_s_min,
        sext_window_s_max=sext_window_s_max,
        sext_window_order=int(sext_window_order),
    )
    npz_path = _study_npz_path(stem)
    np.savez(npz_path, **arrays)
    print(f"Saved DA data: {npz_path}")

    pdf_path = save_figure_pdf(fig, stem)
    return npz_path, pdf_path


MaDirection = Literal["x_only", "y_only"]

MA_DIRECTION_TAGS: dict[str, str] = {"x_only": "X", "y_only": "Y"}


def save_ma_study(
    *,
    out: dict[str, Any],
    tt_init,
    fig: plt.Figure,
    model: ModelTag,
    direction: MaDirection,
    with_solenoids: bool,
    with_correctors: bool,
    n_turns: int,
    global_xy_limit: float,
    nemitt_x: float,
    nemitt_y: float,
    nn_y_r: int,
    max_y_r: float,
    energy_spread: float,
    delta_initial_values: np.ndarray,
    n_part: int,
    radiation: RadiationModel = "mean",
    variant: str = "",
    sexamp: float = BUILD_DEFAULTS["sexamp"],
    x_offset: float = BUILD_DEFAULTS["x_offset"],
    y_offset: float = BUILD_DEFAULTS["y_offset"],
    extra_sext_strength: float = BUILD_DEFAULTS["extra_sext_strength"],
    sext_window_s_min: float | None = BUILD_DEFAULTS["sext_window_s_min"],
    sext_window_s_max: float | None = BUILD_DEFAULTS["sext_window_s_max"],
    sext_window_order: int = BUILD_DEFAULTS["sext_window_order"],
    field_tag: str = FIELD_TAG,
) -> tuple[Path, Path]:
    stem = make_study_stem(
        "MA",
        tag=MA_DIRECTION_TAGS[direction],
        with_solenoids=with_solenoids,
        with_correctors=with_correctors,
        model=model,
        n_part=n_part,
        n_turns=n_turns,
        global_xy_limit=global_xy_limit,
        radiation=radiation,
        variant=variant,
        field_tag=field_tag,
    )
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    particles = out["particles"]
    arrays = dict(
        direction=direction,
        delta_init=np.asarray(tt_init.delta_init),
        x_normalized=np.asarray(tt_init.x_normalized),
        y_normalized=np.asarray(tt_init.y_normalized),
        at_turn=np.asarray(particles.at_turn),
        state=np.asarray(particles.state),
        lost=np.asarray(out["lost"]),
        frac_lost=float(out["frac_lost"]),
        at_turn_mean=float(out["at_turn_mean"]),
        nn_y_r=int(nn_y_r),
        max_y_r=float(max_y_r),
        energy_spread=float(energy_spread),
        delta_initial_values=np.asarray(delta_initial_values),
        nemitt_x=float(nemitt_x),
        nemitt_y=float(nemitt_y),
        n_turns=int(n_turns),
        global_xy_limit=float(global_xy_limit),
        with_solenoids=with_solenoids,
        with_correctors=with_correctors,
        model=model,
        n_part=int(n_part),
        radiation=radiation,
        sexamp=float(sexamp),
        x_offset=float(x_offset),
        y_offset=float(y_offset),
        extra_sext_strength=float(extra_sext_strength),
        sext_window_s_min=sext_window_s_min,
        sext_window_s_max=sext_window_s_max,
        sext_window_order=int(sext_window_order),
    )
    npz_path = _study_npz_path(stem)
    np.savez(npz_path, **arrays)
    print(f"Saved MA data: {npz_path}")

    pdf_path = save_figure_pdf(fig, stem)
    return npz_path, pdf_path


def save_emitt_study(
    *,
    fig: plt.Figure,
    model: ModelTag,
    with_solenoids: bool,
    with_correctors: bool,
    n_turns: int,
    global_xy_limit: float,
    n_part: int,
    turns: np.ndarray,
    gemitt_x: np.ndarray,
    gemitt_y: np.ndarray,
    gemitt_z: np.ndarray,
    eq_gemitt_x: float,
    eq_gemitt_y: float,
    eq_gemitt_zeta: float,
    damping_constants_turns: np.ndarray,
    fit_eps_eq: np.ndarray,
    fit_alpha: np.ndarray,
    fit_tau: np.ndarray,
    fit_eps_init: np.ndarray,
    radiation: RadiationModel = "quantum",
    variant: str = "",
    sexamp: float = BUILD_DEFAULTS["sexamp"],
    field_tag: str = FIELD_TAG,
) -> tuple[Path, Path]:
    stem = make_study_stem(
        "EMIT",
        with_solenoids=with_solenoids,
        with_correctors=with_correctors,
        model=model,
        n_part=n_part,
        n_turns=n_turns,
        global_xy_limit=global_xy_limit,
        radiation=radiation,
        variant=variant,
        field_tag=field_tag,
    )
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    arrays = dict(
        turns=np.asarray(turns),
        gemitt_x=np.asarray(gemitt_x),
        gemitt_y=np.asarray(gemitt_y),
        gemitt_z=np.asarray(gemitt_z),
        eq_gemitt_x=float(eq_gemitt_x),
        eq_gemitt_y=float(eq_gemitt_y),
        eq_gemitt_zeta=float(eq_gemitt_zeta),
        damping_constants_turns=np.asarray(damping_constants_turns),
        fit_eps_eq=np.asarray(fit_eps_eq),
        fit_alpha=np.asarray(fit_alpha),
        fit_tau=np.asarray(fit_tau),
        fit_eps_init=np.asarray(fit_eps_init),
        n_part=int(n_part),
        n_turns=int(n_turns),
        global_xy_limit=float(global_xy_limit),
        with_solenoids=with_solenoids,
        with_correctors=with_correctors,
        model=model,
        radiation=radiation,
        sexamp=float(sexamp),
    )
    npz_path = _study_npz_path(stem)
    np.savez(npz_path, **arrays)
    print(f"Saved EMIT data: {npz_path}")

    pdf_path = save_figure_pdf(fig, stem)
    return npz_path, pdf_path


def save_pol_study(
    *,
    fig: plt.Figure,
    model: ModelTag,
    with_solenoids: bool,
    with_correctors: bool,
    n_turns: int,
    global_xy_limit: float,
    n_part: int,
    turns: np.ndarray,
    spin_x_mean: np.ndarray,
    spin_y_mean: np.ndarray,
    spin_z_mean: np.ndarray,
    polarization: np.ndarray,
    p_inf: float,
    p_eq_twiss: float,
    tau_pol_s: float,
    tau_depol_twiss_s: float,
    spin_tune_fractional: float,
    t_rev0: float,
    fit_p0: float,
    fit_slope: float,
    fit_tau_depol_turns: float,
    fit_tau_depol_s: float,
    p_eq_derived: float,
    radiation: RadiationModel = "quantum",
    variant: str = "",
    sexamp: float = BUILD_DEFAULTS["sexamp"],
    field_tag: str = FIELD_TAG,
) -> tuple[Path, Path]:
    stem = make_study_stem(
        "POL",
        with_solenoids=with_solenoids,
        with_correctors=with_correctors,
        model=model,
        n_part=n_part,
        n_turns=n_turns,
        global_xy_limit=global_xy_limit,
        radiation=radiation,
        variant=variant,
        field_tag=field_tag,
    )
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    arrays = dict(
        turns=np.asarray(turns),
        spin_x_mean=np.asarray(spin_x_mean),
        spin_y_mean=np.asarray(spin_y_mean),
        spin_z_mean=np.asarray(spin_z_mean),
        polarization=np.asarray(polarization),
        p_inf=float(p_inf),
        p_eq_twiss=float(p_eq_twiss),
        tau_pol_s=float(tau_pol_s),
        tau_depol_twiss_s=float(tau_depol_twiss_s),
        spin_tune_fractional=float(spin_tune_fractional),
        t_rev0=float(t_rev0),
        fit_p0=float(fit_p0),
        fit_slope=float(fit_slope),
        fit_tau_depol_turns=float(fit_tau_depol_turns),
        fit_tau_depol_s=float(fit_tau_depol_s),
        p_eq_derived=float(p_eq_derived),
        n_part=int(n_part),
        n_turns=int(n_turns),
        global_xy_limit=float(global_xy_limit),
        with_solenoids=with_solenoids,
        with_correctors=with_correctors,
        model=model,
        radiation=radiation,
        sexamp=float(sexamp),
    )
    npz_path = _study_npz_path(stem)
    np.savez(npz_path, **arrays)
    print(f"Saved POL data: {npz_path}")

    pdf_path = save_figure_pdf(fig, stem)
    return npz_path, pdf_path


def _tune_basename(
    *,
    with_solenoids: bool,
    with_correctors: bool,
    model: ModelTag,
    n_delta: int,
    delta_max: float,
    fit_order: int = 2,
    variant: str = "",
    field_tag: str = FIELD_TAG,
) -> str:
    """Filename stem for TUNE outputs -- mirrors make_basename's naming
    spirit (Sol_On/Off[_Cor_Off], model, field_tag) but swaps the
    tracking-specific n_part/n_turns/global_xy_limit tags for this study's
    own n_delta/delta_max, since a tune-vs-delta scan is Twiss-only (no
    tracking, no particle count, no xy limit). fit_order is only tagged when
    non-default (2), same silent-at-default convention as order_tag()."""
    sol = "Sol_On" if with_solenoids else "Sol_Off"
    if with_solenoids and not with_correctors:
        sol += "_Cor_Off"
    order_suffix = "" if fit_order == 2 else f"_fitorder{fit_order}"
    return (
        f"{sol}_{model}_{field_tag}_{n_delta}d_"
        f"dmax{format_tag_float(delta_max)}{order_suffix}{variant}"
    )


def save_tune_study(
    *,
    fig: plt.Figure,
    model: ModelTag,
    with_solenoids: bool,
    with_correctors: bool,
    n_delta: int,
    delta_max: float,
    delta_values: np.ndarray,
    qx: np.ndarray,
    qy: np.ndarray,
    fit_order: int,
    coeffs_x_fit: np.ndarray,
    coeffs_y_fit: np.ndarray,
    dqx_twiss: float,
    ddqx_twiss: float,
    dqy_twiss: float,
    ddqy_twiss: float,
    qx0_twiss: float,
    qy0_twiss: float,
    variant: str = "",
    sexamp: float = BUILD_DEFAULTS["sexamp"],
    field_tag: str = FIELD_TAG,
) -> tuple[Path, Path]:
    """coeffs_x_fit / coeffs_y_fit are Taylor coefficients from the
    delta-scan polynomial fit, indexed [q0, dq, ddq, dddq, ...] up to
    fit_order (see 017_tune_vs_delta.py's _taylor_coeffs_from_polyfit) --
    only dq/ddq (indices 1/2) have a direct-Twiss counterpart to compare
    against (dqx_twiss/ddqx_twiss etc.); any order above 2 is fit-only."""
    stem = "TUNE_" + _tune_basename(
        with_solenoids=with_solenoids,
        with_correctors=with_correctors,
        model=model,
        n_delta=n_delta,
        delta_max=delta_max,
        fit_order=fit_order,
        variant=variant,
        field_tag=field_tag,
    )
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    arrays = dict(
        delta_values=np.asarray(delta_values),
        qx=np.asarray(qx),
        qy=np.asarray(qy),
        fit_order=int(fit_order),
        coeffs_x_fit=np.asarray(coeffs_x_fit),
        coeffs_y_fit=np.asarray(coeffs_y_fit),
        dqx_twiss=float(dqx_twiss),
        ddqx_twiss=float(ddqx_twiss),
        dqy_twiss=float(dqy_twiss),
        ddqy_twiss=float(ddqy_twiss),
        qx0_twiss=float(qx0_twiss),
        qy0_twiss=float(qy0_twiss),
        n_delta=int(n_delta),
        delta_max=float(delta_max),
        with_solenoids=with_solenoids,
        with_correctors=with_correctors,
        model=model,
        sexamp=float(sexamp),
    )
    npz_path = _study_npz_path(stem)
    np.savez(npz_path, **arrays)
    print(f"Saved TUNE data: {npz_path}")

    pdf_path = save_figure_pdf(fig, stem)
    return npz_path, pdf_path


def _plot_da_from_arrays(data: np.lib.npyio.NpzFile) -> plt.Figure:
    x_hat = data["x_hat"]
    y_hat = data["y_normalized"]
    at_turn = data["at_turn"]
    nn_y_r = int(data["nn_y_r"])
    nn_x_theta = int(data["nn_x_theta"])
    sigma_x = float(data["sigma_x"])
    sigma_y = float(data["sigma_y"])
    frac_lost = float(data["frac_lost"])
    at_turn_mean = float(data["at_turn_mean"])

    sol = "Sol On" if bool(data["with_solenoids"]) else "Sol Off"
    model = "SplineBoris" if str(data["model"]) == "SB" else "VariableSolenoid"
    if bool(data["with_solenoids"]) and not bool(data["with_correctors"]):
        title = f"{model}: solenoids powered, correctors off"
    elif bool(data["with_solenoids"]):
        title = f"{model}: solenoids powered + correction scheme"
    else:
        title = f"{model}: solenoids unpowered"

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
    sc = axes[0].scatter(x_hat, y_hat, c=at_turn, s=8, marker="o")
    axes[0].set_xlabel(r"$\hat{x}\,[\sigma_x]$")
    axes[0].set_ylabel(r"$\hat{y}\,[\sigma_y]$")
    axes[0].set_title("scatter")
    fig.colorbar(sc, ax=axes[0], label="lost at turn")

    pcm = axes[1].pcolormesh(
        x_hat.reshape(nn_y_r, nn_x_theta),
        y_hat.reshape(nn_y_r, nn_x_theta),
        at_turn.reshape(nn_y_r, nn_x_theta),
        shading="gouraud",
    )
    axes[1].set_xlabel(r"$\hat{x}\,[\sigma_x]$")
    axes[1].set_ylabel(r"$\hat{y}\,[\sigma_y]$")
    axes[1].set_title("pcolormesh")
    fig.colorbar(pcm, ax=axes[1], label="lost at turn")

    fig.suptitle(
        f"{title}\n"
        f"frac_lost={frac_lost:.4g}, at_turn_mean={at_turn_mean:.4g}\n"
        f"$\\sigma_x={sigma_x*1e6:.2f}\\,\\mu$m, "
        f"$\\sigma_y={sigma_y*1e6:.2f}\\,\\mu$m "
        f"($\\sigma_x/\\sigma_y={sigma_x/sigma_y:.0f}$)"
    )
    fig.tight_layout()
    return fig


def _plot_ma_from_arrays(data: np.lib.npyio.NpzFile) -> plt.Figure:
    delta_init = data["delta_init"]
    # Older MA files (pre direction-split) only ever scanned x_only.
    direction = str(data["direction"]) if "direction" in data.files else "x_only"
    amp_key, amp_symbol = (
        ("y_normalized", r"\hat{y}") if direction == "y_only" else ("x_normalized", r"\hat{x}")
    )
    amplitude = data[amp_key]
    at_turn = data["at_turn"]
    lost = data["lost"].astype(bool)
    frac_lost = float(data["frac_lost"])
    at_turn_mean = float(data["at_turn_mean"])

    sol = "Sol On" if bool(data["with_solenoids"]) else "Sol Off"
    model = "SplineBoris" if str(data["model"]) == "SB" else "VariableSolenoid"
    if bool(data["with_solenoids"]) and not bool(data["with_correctors"]):
        title = f"{model}: solenoids powered, correctors off"
    elif bool(data["with_solenoids"]):
        title = f"{model}: solenoids powered + correction scheme"
    else:
        title = f"{model}: solenoids unpowered"
    title = f"{title} ({'y' if direction == 'y_only' else 'x'}-only)"

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot(
        delta_init[~lost],
        amplitude[~lost],
        ".",
        ms=3,
        label="survived",
    )
    sc = ax.scatter(
        delta_init[lost],
        amplitude[lost],
        c=at_turn[lost],
        marker="o",
        s=18,
        label="lost",
    )
    ax.set_xlabel(r"$\delta$")
    ax.set_ylabel(fr"${amp_symbol}$")
    ax.set_title(
        f"{title}\n"
        f"frac_lost={frac_lost:.4g}, at_turn_mean={at_turn_mean:.4g}"
    )
    fig.colorbar(sc, ax=ax, label="lost at turn")
    return fig


def replot_from_npz(npz_path: Path | str, *, show: bool = False) -> Path:
    npz_path = Path(npz_path)
    with np.load(npz_path, allow_pickle=True) as data:
        study = npz_path.name.split("_", 1)[0]
        if study == "DA":
            fig = _plot_da_from_arrays(data)
        elif study == "MA":
            fig = _plot_ma_from_arrays(data)
        else:
            raise ValueError(f"Unrecognized study prefix in {npz_path.name}")

        pdf_path = save_figure_pdf(fig, npz_path.stem)

    if show:
        plt.show()
    else:
        plt.close(fig)
    return pdf_path
