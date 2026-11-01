"""E0–E5 runner with study-box clip, native square grid, meter-aligned spectra."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import rasterio
import yaml
from affine import Affine
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds
from scipy import ndimage

from .hex_grid import build_hex_grid
from .morphology import (
    align_peak_to_meters,
    l1_spectrum_meters,
    pattern_spectrum,
    resample_ps_to_meters,
)


def load_config(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def pick_device(requested: str) -> str:
    if requested in ("cpu", "cuda"):
        if requested == "cuda":
            try:
                import cupy  # noqa: F401

                return "cuda"
            except Exception:
                print("[warn] cuda requested but cupy missing; CPU fallback")
                return "cpu"
        return "cpu"
    try:
        import cupy  # noqa: F401

        cupy.cuda.Device(0).compute_capability
        return "cuda"
    except Exception:
        return "cpu"


def clip_reproject_dem(
    src_path: Path,
    west: float,
    south: float,
    east: float,
    north: float,
    proj: str,
    res_m: float,
    dst_path: Path,
) -> Path:
    """Warp only the WGS84 study box to an equal-area grid at res_m."""
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    left, bottom, right, top = transform_bounds(
        "EPSG:4326", proj, west, south, east, north, densify_pts=21
    )
    if top < bottom:
        bottom, top = top, bottom
    res_m = float(res_m)
    width = max(16, int(np.ceil((right - left) / res_m)))
    height = max(16, int(np.ceil((top - bottom) / res_m)))
    dst_transform = Affine(res_m, 0.0, left, 0.0, -res_m, top)
    with rasterio.open(src_path) as src:
        nodata = src.nodata if src.nodata is not None else -9999.0
        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": "float32",
            "crs": proj,
            "transform": dst_transform,
            "nodata": nodata,
            "compress": "lzw",
        }
        with rasterio.open(dst_path, "w", **profile) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=proj,
                resampling=Resampling.average,
                src_nodata=src.nodata,
                dst_nodata=nodata,
            )
    return dst_path


def read_band(path: Path):
    with rasterio.open(path) as src:
        z = src.read(1).astype(np.float64)
        nodata = src.nodata
        if nodata is not None:
            z[z == nodata] = np.nan
        z[~np.isfinite(z)] = np.nan
        z[z < -100] = np.nan
        return z, src.transform, nodata, str(src.crs)


def rotate_array(z: np.ndarray, angle: float) -> np.ndarray:
    """Rotate in-place (same shape) so hex/square share the same domain."""
    if abs(angle) < 1e-9:
        return z
    finite = z[np.isfinite(z)]
    fill = float(np.min(finite) - 1.0) if finite.size else 0.0
    x = np.where(np.isfinite(z), z, fill)
    y = ndimage.rotate(x, angle, reshape=False, order=1, mode="constant", cval=fill)
    y[y <= fill + 0.5] = np.nan
    y[np.isnan(z)] = np.nan
    return y


def _ps(z, area, n_scales, kind, device, spacing, connectivity=8):
    d = pattern_spectrum(
        z, area, n_scales, kind=kind, connectivity=connectivity, device=device
    )
    d["lambda_m"] = align_peak_to_meters(d["lambda_peak"], spacing)
    d["spacing_m"] = float(spacing)
    return d


def _row(exp, tag, setting, hps, sps, dh=0.0, ds=0.0, passed="", note=""):
    return {
        "exp": exp,
        "tag": tag,
        "setting": setting,
        "lambda_hex_m": hps if np.isscalar(hps) else hps.get("lambda_m", ""),
        "lambda_sq_m": sps if np.isscalar(sps) else sps.get("lambda_m", ""),
        "dL1_hex": dh,
        "dL1_sq": ds,
        "pass": passed,
        "note": note,
    }


def run_one_scene(
    tif_wgs84: Path,
    proj: str,
    work: Path,
    resolutions: list[int],
    rotations: list[int],
    n_scales: int,
    device: str,
    tag: str,
    bbox: dict | None = None,
) -> dict:
    if bbox is None:
        raise ValueError("bbox is required so 1° COP30 tiles are clipped to the study box")
    work.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    base_res = int(min(resolutions))
    hex_r = lambda res: float(res) / np.sqrt(3.0)

    def warp(res: float, name: str) -> Path:
        return clip_reproject_dem(
            tif_wgs84,
            bbox["west"],
            bbox["south"],
            bbox["east"],
            bbox["north"],
            proj,
            res,
            work / name,
        )

    base_tif = warp(base_res, f"{tag}_{base_res}m.tif")
    z0, transform0, _, _ = read_band(base_tif)
    print(f"[scene] {tag} {base_res}m raster {z0.shape[1]}x{z0.shape[0]}")
    hex0 = build_hex_grid(z0, transform0, proj, size_m=hex_r(base_res))
    sq_area = float(base_res) * float(base_res)
    e0_hex = _ps(hex0.values, hex0.cell_area, n_scales, "hex", device, hex0.spacing)
    e0_sq = _ps(z0, sq_area, n_scales, "square", device, float(base_res))
    vol_rel = abs(e0_hex["volume0"] - e0_sq["volume0"]) / max(e0_hex["volume0"], 1.0)
    mean_h = float(np.nanmean(hex0.values))
    mean_s = float(np.nanmean(z0))
    relief = float(np.nanmax(z0) - np.nanmin(z0))
    mean_rel = abs(mean_h - mean_s) / max(relief, 1.0)
    area_h = float(np.isfinite(hex0.values).sum() * hex0.cell_area)
    area_s = float(np.isfinite(z0).sum() * sq_area)
    area_rel = abs(area_h - area_s) / max(area_s, 1.0)
    e0_pass = (vol_rel < 0.05 or (mean_rel < 0.05 and area_rel < 0.08)) and e0_hex["lambda_peak"] > 0
    rows.append(
        _row(
            "E0",
            tag,
            f"{base_res}m, theta=0",
            e0_hex,
            e0_sq,
            passed=bool(e0_pass),
            note=(
                f"vol_rel={vol_rel:.4f} mean_rel={mean_rel:.4f} area_rel={area_rel:.4f} "
                f"shape={z0.shape[1]}x{z0.shape[0]} hex={hex0.values.shape[1]}x{hex0.values.shape[0]}"
            ),
        )
    )

    e1_hex_dl1, e1_sq_dl1, e1_hex_rel, e1_sq_rel = [], [], [], []
    e1_hex_specs = []
    e1_sq_specs = []
    for res in resolutions:
        if int(res) == base_res:
            continue
        tif = warp(res, f"{tag}_{int(res)}m.tif")
        z, transform, _, _ = read_band(tif)
        hx = build_hex_grid(z, transform, proj, size_m=hex_r(res))
        hps = _ps(hx.values, hx.cell_area, n_scales, "hex", device, hx.spacing)
        sps = _ps(z, float(res) * float(res), n_scales, "square", device, float(res))
        dh = l1_spectrum_meters(hps, hps["spacing_m"], e0_hex, e0_hex["spacing_m"])
        ds = l1_spectrum_meters(sps, sps["spacing_m"], e0_sq, e0_sq["spacing_m"])
        rh = abs(hps["lambda_m"] - e0_hex["lambda_m"]) / max(e0_hex["lambda_m"], 1e-6)
        rs = abs(sps["lambda_m"] - e0_sq["lambda_m"]) / max(e0_sq["lambda_m"], 1e-6)
        e1_hex_dl1.append(dh)
        e1_sq_dl1.append(ds)
        e1_hex_rel.append(rh)
        e1_sq_rel.append(rs)
        e1_hex_specs.append(hps)
        e1_sq_specs.append(sps)
        rows.append(
            _row("E1", tag, f"{int(res)} vs {base_res}", hps, sps, dh, ds, note=f"rel hex={rh:.3f} sq={rs:.3f}")
        )
    e1_pass = (np.mean(e1_hex_rel) < np.mean(e1_sq_rel)) or (np.mean(e1_hex_dl1) < np.mean(e1_sq_dl1))
    rows.append(
        _row(
            "E1_summary",
            tag,
            "mean 60-120 vs 30",
            float(np.mean(e1_hex_rel)),
            float(np.mean(e1_sq_rel)),
            float(np.mean(e1_hex_dl1)),
            float(np.mean(e1_sq_dl1)),
            passed=bool(e1_pass),
            note="meter-aligned L1; pass if hex drift < square on peak OR L1",
        )
    )

    ai_h, ai_s = [], []
    for ang in rotations:
        if int(ang) == 0:
            continue
        zr = rotate_array(z0, float(ang))
        hx = build_hex_grid(zr, transform0, proj, size_m=hex_r(base_res))
        hps = _ps(hx.values, hx.cell_area, n_scales, "hex", device, hx.spacing)
        sps = _ps(zr, sq_area, n_scales, "square", device, float(base_res))
        dh = l1_spectrum_meters(hps, hps["spacing_m"], e0_hex, e0_hex["spacing_m"])
        ds = l1_spectrum_meters(sps, sps["spacing_m"], e0_sq, e0_sq["spacing_m"])
        ai_h.append(dh)
        ai_s.append(ds)
        rows.append(_row("E2", tag, f"rot {ang}", hps, sps, dh, ds))
    e2_pass = (max(ai_h) if ai_h else 1.0) < (max(ai_s) if ai_s else 0.0)
    rows.append(
        _row(
            "E2_summary",
            tag,
            "max AI 15-45 vs 0",
            float(max(ai_h) if ai_h else 0),
            float(max(ai_s) if ai_s else 0),
            float(max(ai_h) if ai_h else 0),
            float(max(ai_s) if ai_s else 0),
            passed=bool(e2_pass),
            note="pass if max AI hex < max AI square (meter-aligned)",
        )
    )

    tif60 = warp(60, f"{tag}_60m.tif")
    z60, _, _, _ = read_band(tif60)
    ps8_30 = _ps(z0, sq_area, n_scales, "square", device, float(base_res), 8)
    ps8_60 = _ps(z60, 60.0 * 60.0, n_scales, "square", device, 60.0, 8)
    ps4_30 = _ps(z0, sq_area, n_scales, "square", device, float(base_res), 4)
    ps4_60 = _ps(z60, 60.0 * 60.0, n_scales, "square", device, 60.0, 4)
    d8 = l1_spectrum_meters(ps8_60, 60.0, ps8_30, float(base_res))
    d4 = l1_spectrum_meters(ps4_60, 60.0, ps4_30, float(base_res))
    dh60 = float(e1_hex_dl1[0]) if e1_hex_dl1 else 0.0
    sign_8 = dh60 < d8
    sign_4 = dh60 < d4
    e3_pass = bool(sign_8 == sign_4)
    rows.append(
        _row(
            "E3",
            tag,
            "square 4 vs 8 neigh, 30-60 L1",
            d4,
            d8,
            d4,
            d8,
            passed=e3_pass,
            note=(
                f"hex L1={dh60:.4f}; hex<sq8={sign_8} hex<sq4={sign_4}; "
                "sign must not flip"
            ),
        )
    )

    finite = z0[np.isfinite(z0)]
    preview = np.nan_to_num(z0, nan=float(np.min(finite) if finite.size else 0.0)).astype(np.float32)
    rmax_m = float(n_scales) * max(float(base_res), float(hex0.spacing))
    meter_axis = np.linspace(min(float(base_res), float(hex0.spacing)), rmax_m, 64)
    return {
        "rows": rows,
        "z_preview": preview,
        "ps_hex": e0_hex["ps"],
        "ps_sq": e0_sq["ps"],
        "radii": e0_hex["radii"],
        "meter_axis": meter_axis,
        "ps_hex_m": resample_ps_to_meters(e0_hex["radii"], e0_hex["ps"], e0_hex["spacing_m"], meter_axis),
        "ps_sq_m": resample_ps_to_meters(e0_sq["radii"], e0_sq["ps"], e0_sq["spacing_m"], meter_axis),
        "e0_pass": bool(e0_pass),
        "e1_pass": bool(e1_pass),
        "e2_pass": bool(e2_pass),
        "vol_rel": float(vol_rel),
        "mean_rel": float(mean_rel),
        "area_rel": float(area_rel),
        "shape": (int(z0.shape[1]), int(z0.shape[0])),
    }


def write_table(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["exp", "tag", "setting", "lambda_hex_m", "lambda_sq_m", "dL1_hex", "dL1_sq", "pass", "note"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_figures(bundle: dict, out: Path, tag: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print("[warn] skip figures:", exc)
        return

    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.imshow(bundle["z_preview"], cmap="terrain")
    ax.set_title(f"{tag} DEM (study box)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / f"fig2_{tag}_dem.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    if "meter_axis" in bundle:
        ax.plot(bundle["meter_axis"], bundle["ps_hex_m"], marker="o", label="hex")
        ax.plot(bundle["meter_axis"], bundle["ps_sq_m"], marker="s", label="square")
        ax.set_xlabel("opening radius (m)")
    else:
        ax.plot(bundle["radii"], bundle["ps_hex"], marker="o", label="hex")
        ax.plot(bundle["radii"], bundle["ps_sq"], marker="s", label="square")
        ax.set_xlabel("opening radius (cells)")
    ax.set_ylabel("normalized pattern spectrum")
    ax.legend()
    ax.set_title("E0 pattern spectrum")
    fig.tight_layout()
    fig.savefig(out / f"fig2_{tag}_spectrum.png", dpi=200)
    plt.close(fig)
