"""Save and reload DA / MA study results (raw data + PDF plots)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
PLOT_DIR = Path("/home/simonfan/cernbox/Pictures/FCC_Solenoid_Studies")

ModelTag = Literal["SB", "VarSol"]
StudyTag = Literal["DA", "MA", "EMIT"]

BUILD_DEFAULTS = {
    "sexamp": 1.0,
    "theta": -0.015,
}


def format_tag_float(value: float) -> str:
    """Format a float for filename tags, e.g. 2.0 -> '2p0'."""
    if value == int(value):
        text = f"{int(value)}.0"
    else:
        text = f"{value:g}"
    return text.replace(".", "p").replace("-", "m")


def variant_suffix(**overrides: float) -> str:
    """Return '' or a double-underscore variant tag for non-default build knobs."""
    tags: list[str] = []
    sexamp = overrides.get("sexamp", BUILD_DEFAULTS["sexamp"])
    if sexamp != BUILD_DEFAULTS["sexamp"]:
        tags.append(f"sexamp{format_tag_float(sexamp)}")
    if not tags:
        return ""
    return "__" + tags[0] if len(tags) == 1 else "__" + "__".join(tags)


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
    variant: str = "",
) -> str:
    """Build a filename stem (without study prefix) for NPZ/PDF outputs."""
    sol = "Sol_On" if with_solenoids else "Sol_Off"
    if with_solenoids and not with_correctors:
        sol += "_Cor_Off"
    return (
        f"{sol}_{model}_{n_part}p_{n_turns}t_"
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
    nemitt_x: float,
    nemitt_y: float,
    n_part: int,
    variant: str = "",
    sexamp: float = BUILD_DEFAULTS["sexamp"],
) -> tuple[Path, Path]:
    stem = make_study_stem(
        "DA",
        with_solenoids=with_solenoids,
        with_correctors=with_correctors,
        model=model,
        n_part=n_part,
        n_turns=n_turns,
        global_xy_limit=global_xy_limit,
        variant=variant,
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
        nemitt_x=float(nemitt_x),
        nemitt_y=float(nemitt_y),
        n_turns=int(n_turns),
        global_xy_limit=float(global_xy_limit),
        with_solenoids=with_solenoids,
        with_correctors=with_correctors,
        model=model,
        n_part=int(n_part),
        sexamp=float(sexamp),
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
    variant: str = "",
    sexamp: float = BUILD_DEFAULTS["sexamp"],
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
        variant=variant,
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
        sexamp=float(sexamp),
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
    variant: str = "",
    sexamp: float = BUILD_DEFAULTS["sexamp"],
) -> tuple[Path, Path]:
    stem = make_study_stem(
        "EMIT",
        with_solenoids=with_solenoids,
        with_correctors=with_correctors,
        model=model,
        n_part=n_part,
        n_turns=n_turns,
        global_xy_limit=global_xy_limit,
        variant=variant,
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
        sexamp=float(sexamp),
    )
    npz_path = _study_npz_path(stem)
    np.savez(npz_path, **arrays)
    print(f"Saved EMIT data: {npz_path}")

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
