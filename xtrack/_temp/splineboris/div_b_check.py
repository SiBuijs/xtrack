"""
Shared div(B) sanity check for FieldFitter/TubeFitter raw input data.

Both fitters consume a field map on a rectangular (X, Y, Z) grid; before any
fitting happens, div(B) = dBx/dx + dBy/dy + dBs/dz can already be estimated
directly from the raw grid via finite differences, independent of whichever
fitting method is used downstream. A real (source-free) field satisfies
div(B) = 0 everywhere -- any nonzero value here reflects noise/discretization
error already present in the input data, not a fitting artifact, and sets a
floor on how "Maxwellian" any fit of it can be judged to be.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def report_raw_div_b(df_raw_data: pd.DataFrame, label: str) -> float | None:
    """
    Print the RMS deviation of div(B) from zero, estimated from raw (X, Y, Z)
    grid data via finite differences (``np.gradient``, actual coordinate
    spacing). ``df_raw_data`` must have a full rectangular ``('X', 'Y', 'Z')``
    MultiIndex and either ``('Bx', 'By', 'Bs')`` or ``('Bskew', 'Bnorm',
    'Bs')`` columns (FieldFitter and TubeFitter use different raw-column
    names for the same physical quantities).

    Returns the relative RMS (fraction of the typical individual-term
    magnitude, i.e. RMS of sqrt(dBx/dx^2 + dBy/dy^2 + dBs/dz^2)), or None if
    there isn't enough grid resolution in some direction to estimate a
    derivative there (fewer than 2 unique values).
    """
    cols = df_raw_data.columns
    x_col = "Bx" if "Bx" in cols else "Bskew"
    y_col = "By" if "By" in cols else "Bnorm"
    s_col = "Bs"

    idx = df_raw_data.index
    x_vals = np.sort(idx.get_level_values("X").unique().to_numpy(dtype=float))
    y_vals = np.sort(idx.get_level_values("Y").unique().to_numpy(dtype=float))
    z_vals = np.sort(idx.get_level_values("Z").unique().to_numpy(dtype=float))
    nx, ny, nz = len(x_vals), len(y_vals), len(z_vals)

    if min(nx, ny, nz) < 2:
        print(
            f"[{label}] div(B) check skipped: need >=2 unique values along "
            f"X, Y, and Z (got {nx}, {ny}, {nz})."
        )
        return None

    full_index = pd.MultiIndex.from_product([x_vals, y_vals, z_vals], names=["X", "Y", "Z"])
    df_grid = df_raw_data[[x_col, y_col, s_col]].reindex(full_index)

    bx_grid = df_grid[x_col].to_numpy(dtype=float).reshape(nx, ny, nz)
    by_grid = df_grid[y_col].to_numpy(dtype=float).reshape(nx, ny, nz)
    bs_grid = df_grid[s_col].to_numpy(dtype=float).reshape(nx, ny, nz)

    if np.any(np.isnan(bx_grid)) or np.any(np.isnan(by_grid)) or np.any(np.isnan(bs_grid)):
        print(f"[{label}] div(B) check skipped: raw data isn't a full rectangular X/Y/Z grid.")
        return None

    dbx_dx = np.gradient(bx_grid, x_vals, axis=0)
    dby_dy = np.gradient(by_grid, y_vals, axis=1)
    dbs_dz = np.gradient(bs_grid, z_vals, axis=2)
    div_b = dbx_dx + dby_dy + dbs_dz

    div_rms = float(np.sqrt(np.mean(div_b ** 2)))
    term_rms = float(np.sqrt(np.mean(dbx_dx ** 2 + dby_dy ** 2 + dbs_dz ** 2)))
    rel = div_rms / term_rms if term_rms > 0 else 0.0

    print(
        f"[{label}] raw-data div(B) deviation: RMS = {div_rms:.3e} T/m "
        f"({rel * 100:.2f}% of typical |dB_i/dx_i| term, from finite "
        f"differences on the raw X/Y/Z grid)"
    )
    return rel
