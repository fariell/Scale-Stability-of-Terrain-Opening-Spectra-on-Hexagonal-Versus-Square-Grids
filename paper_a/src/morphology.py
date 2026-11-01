"""Grey-scale hexagonal / square opening and morphological pattern spectra."""
from __future__ import annotations

from typing import Literal

import numpy as np
from scipy import ndimage

try:
    import cupy as cp  # type: ignore

    HAS_CUPY = True
except Exception:
    cp = None
    HAS_CUPY = False


Array = np.ndarray


def _xp(device: Literal["cpu", "cuda"]):
    if device == "cuda":
        if not HAS_CUPY:
            raise RuntimeError("cupy is not installed; use device=cpu or pip install cupy-cuda12x")
        return cp
    return np


def _to_xp(z: Array, device: str):
    xp = _xp(device)  # type: ignore
    if device == "cuda":
        return xp.asarray(z)
    return z


def _to_np(z) -> Array:
    if HAS_CUPY and isinstance(z, cp.ndarray):
        return cp.asnumpy(z)
    return np.asarray(z)


def square_disk(radius: int, connectivity: int = 8) -> np.ndarray:
    r = int(radius)
    yy, xx = np.ogrid[-r : r + 1, -r : r + 1]
    if connectivity == 4:
        return (np.abs(xx) + np.abs(yy)) <= r
    return xx * xx + yy * yy <= r * r + 1e-9


def grey_open_square(z: Array, radius: int, connectivity: int = 8, device: str = "cpu") -> Array:
    """Opening on a square raster. NaNs are treated as fill (no leakage)."""
    se = square_disk(radius, connectivity)
    fill = np.nanmin(z) - 1.0
    x = np.where(np.isnan(z), fill, z)
    if device == "cuda" and HAS_CUPY:
        # CuPy has no grey_opening; run scipy on CPU for square (small cost).
        device = "cpu"
    opened = ndimage.grey_opening(x, footprint=se, mode="nearest")
    opened[np.isnan(z)] = np.nan
    return opened.astype(np.float64)


def _hex_neighbor_max(x, fill):
    """One hex dilation step on odd-r offset grid. x has fill at invalid cells."""
    xp = type(x) if HAS_CUPY and isinstance(x, cp.ndarray) else np
    if HAS_CUPY and isinstance(x, cp.ndarray):
        xp = cp
    else:
        xp = np
    p = xp.pad(x, 1, constant_values=fill)
    c = p[1:-1, 1:-1]
    e = p[1:-1, 2:]
    w = p[1:-1, :-2]
    n = p[:-2, 1:-1]
    s = p[2:, 1:-1]
    # odd-r: even rows use NE/SE = (row±1, col+1); odd rows use NW/SW = (row±1, col-1)
    ne = p[:-2, 2:]
    se = p[2:, 2:]
    nw = p[:-2, :-2]
    sw = p[2:, :-2]
    nrows = c.shape[0]
    even = (xp.arange(nrows) % 2) == 0
    even = even[:, None]
    n_diag = xp.where(even, ne, nw)
    s_diag = xp.where(even, se, sw)
    return xp.maximum.reduce([c, e, w, n, s, n_diag, s_diag])


def grey_open_hex(z: Array, radius: int, device: str = "cpu") -> Array:
    """Hex opening = dilate^r ( erode^r (z) ) with 6-neighborhood."""
    radius = int(radius)
    fill_low = -1e30
    fill_high = 1e30
    inv = np.isnan(z)
    x = np.where(inv, fill_high, z)  # erosion uses min → pad with +inf
    x = _to_xp(x, device)
    for _ in range(radius):
        # erosion = -dilate(-z) but with +inf fill; neighbor min
        x = -_hex_neighbor_max(-x, fill_low)
    x = _to_np(x)
    x = np.where(inv, fill_low, x)  # dilation uses max → pad with -inf
    x = _to_xp(x, device)
    for _ in range(radius):
        x = _hex_neighbor_max(x, fill_low)
    opened = _to_np(x)
    opened[inv] = np.nan
    return opened.astype(np.float64)


def volume(z: Array, cell_area: float) -> float:
    v = z[np.isfinite(z)]
    if v.size == 0:
        return 0.0
    return float(v.sum() * cell_area)


def pattern_spectrum(
    z: Array,
    cell_area: float,
    n_scales: int,
    kind: Literal["hex", "square"] = "hex",
    connectivity: int = 8,
    device: str = "cpu",
) -> dict:
    """Normalized opening pattern spectrum PS(r) = (V(γ_{r-1}) - V(γ_r)) / sum."""
    radii = np.arange(1, n_scales + 1, dtype=int)
    vols = []
    z_work = z.astype(np.float64)
    vols.append(volume(z_work, cell_area))
    for r in radii:
        if kind == "hex":
            z_work = grey_open_hex(z, r, device=device)
        else:
            z_work = grey_open_square(z, r, connectivity=connectivity, device=device)
        vols.append(volume(z_work, cell_area))
    vols = np.asarray(vols, dtype=np.float64)
    ps = vols[:-1] - vols[1:]
    ps = np.clip(ps, 0, None)
    s = ps.sum()
    ps_n = ps / s if s > 0 else ps
    peak = int(radii[int(np.argmax(ps_n))]) if ps_n.size else 0
    return {
        "radii": radii,
        "ps": ps_n,
        "ps_raw": ps,
        "volumes": vols,
        "lambda_peak": peak,
        "volume0": float(vols[0]),
    }


def l1_spectrum(ps_a: Array, ps_b: Array) -> float:
    n = min(len(ps_a), len(ps_b))
    return float(np.abs(ps_a[:n] - ps_b[:n]).sum())


def align_peak_to_meters(peak_r: int, spacing_m: float) -> float:
    return float(peak_r * spacing_m)


def resample_ps_to_meters(
    radii: Array,
    ps: Array,
    spacing_m: float,
    meter_axis: Array,
) -> Array:
    """Interpolate a cell-index spectrum onto a common physical-radius axis (m)."""
    r_m = np.asarray(radii, dtype=np.float64) * float(spacing_m)
    y = np.interp(meter_axis, r_m, np.asarray(ps, dtype=np.float64), left=0.0, right=0.0)
    s = float(y.sum())
    return y / s if s > 0 else y


def l1_spectrum_meters(
    spec_a: dict,
    spacing_a: float,
    spec_b: dict,
    spacing_b: float,
    n_axis: int = 64,
    wmin: float | None = None,
    wmax: float | None = None,
) -> float:
    """L1 after both spectra are aligned in meters (not cell counts)."""
    rmax = max(float(spec_a["radii"][-1]) * spacing_a, float(spec_b["radii"][-1]) * spacing_b)
    rmin = min(spacing_a, spacing_b)
    lo = float(wmin) if wmin is not None else rmin
    hi = float(wmax) if wmax is not None else rmax
    axis = np.linspace(lo, hi, n_axis)
    pa = resample_ps_to_meters(spec_a["radii"], spec_a["ps"], spacing_a, axis)
    pb = resample_ps_to_meters(spec_b["radii"], spec_b["ps"], spacing_b, axis)
    return float(np.abs(pa - pb).sum())


def hex_se_cells(radius: int) -> int:
    r = int(radius)
    return 3 * r * r + 3 * r + 1


def square_se_cells(radius: int, connectivity: int = 8) -> int:
    return int(square_disk(radius, connectivity).sum())


def se_area_m2(kind: str, radius: int, spacing_m: float, connectivity: int = 8) -> float:
    if kind == "hex":
        a = (np.sqrt(3.0) / 2.0) * float(spacing_m) ** 2
        return hex_se_cells(radius) * a
    a = float(spacing_m) ** 2
    return square_se_cells(radius, connectivity) * a


def equal_area_square_radius(hex_r: int, hex_spacing: float, sq_spacing: float, connectivity: int = 8) -> int:
    target = se_area_m2("hex", hex_r, hex_spacing)
    best_r, best_d = 1, 1e99
    for r in range(1, 24):
        d = abs(se_area_m2("square", r, sq_spacing, connectivity) - target)
        if d < best_d:
            best_d, best_r = d, r
    return int(best_r)


def volumes_monotone(spec: dict, atol: float = 1e-6) -> bool:
    v = np.asarray(spec["volumes"], dtype=np.float64)
    return bool(np.all(np.diff(v) <= atol))


def spectral_stats(spec: dict, spacing_m: float) -> dict:
    """Continuous scale statistics on the native cell axis mapped to meters."""
    r = np.asarray(spec["radii"], dtype=np.float64)
    ps = np.asarray(spec["ps"], dtype=np.float64)
    w = r * float(spacing_m)
    mass = float(ps.sum())
    if mass <= 0:
        return {"mean_m": np.nan, "median_m": np.nan, "subbin_peak_m": np.nan, "argmax_m": np.nan}
    p = ps / mass
    mean_m = float((w * p).sum())
    cdf = np.cumsum(p)
    median_m = float(np.interp(0.5, cdf, w))
    i = int(np.argmax(ps))
    argmax_m = float(w[i])
    sub = argmax_m
    if 0 < i < len(ps) - 1:
        y0, y1, y2 = float(ps[i - 1]), float(ps[i]), float(ps[i + 1])
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) > 1e-12:
            delta = 0.5 * (y0 - y2) / denom
            delta = float(np.clip(delta, -0.5, 0.5))
            sub = (r[i] + delta) * float(spacing_m)
    return {"mean_m": mean_m, "median_m": median_m, "subbin_peak_m": sub, "argmax_m": argmax_m}


def wasserstein1_meters(
    spec_a: dict,
    spacing_a: float,
    spec_b: dict,
    spacing_b: float,
    wmin: float,
    wmax: float,
    n_axis: int = 128,
) -> float:
    """1-D Wasserstein-1 on a common physical support after renormalization."""
    axis = np.linspace(float(wmin), float(wmax), int(n_axis))
    pa = resample_ps_to_meters(spec_a["radii"], spec_a["ps"], spacing_a, axis)
    pb = resample_ps_to_meters(spec_b["radii"], spec_b["ps"], spacing_b, axis)
    dx = float(axis[1] - axis[0]) if len(axis) > 1 else 1.0
    cdf_a = np.cumsum(pa)
    cdf_b = np.cumsum(pb)
    return float(np.abs(cdf_a - cdf_b).sum() * dx)
