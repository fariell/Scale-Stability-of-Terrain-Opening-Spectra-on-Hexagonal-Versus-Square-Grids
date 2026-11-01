#!/usr/bin/env python3
"""GRSL revision experiments: isolate tessellation, robust metrics, uncertainty."""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def _fix_windows_proj() -> None:
    prefix = Path(sys.prefix)
    for cand in (prefix / "Library" / "share" / "proj", prefix / "share" / "proj"):
        if (cand / "proj.db").exists():
            os.environ.setdefault("PROJ_DATA", str(cand))
            os.environ.setdefault("PROJ_LIB", str(cand))
            break


_fix_windows_proj()

from src.experiments import clip_reproject_dem, load_config, pick_device, read_band  # noqa: E402
from src.hex_grid import build_hex_grid  # noqa: E402
from src.morphology import (  # noqa: E402
    equal_area_square_radius,
    grey_open_square,
    l1_spectrum_meters,
    pattern_spectrum,
    se_area_m2,
    spectral_stats,
    square_disk,
    volumes_monotone,
    wasserstein1_meters,
)


def hex_r_of_spacing(spacing_m: float) -> float:
    return float(spacing_m) / np.sqrt(3.0)


def apply_mask(z: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = z.copy()
    out[~mask] = np.nan
    return out


def interior_square(z: np.ndarray, rmax: int) -> np.ndarray:
    v = np.isfinite(z)
    se = square_disk(int(rmax), 8)
    return ndimage.binary_erosion(v, structure=se)


def resample_square_lattice(z: np.ndarray, angle_deg: float, order: int = 1) -> np.ndarray:
    """Rotate the sampling lattice; every angle including 0 uses the same interpolator."""
    h, w = z.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.mgrid[0:h, 0:w]
    rad = np.deg2rad(float(angle_deg))
    c, s = np.cos(rad), np.sin(rad)
    dx, dy = xx - cx, yy - cy
    xs = c * dx + s * dy + cx
    ys = -s * dx + c * dy + cy
    finite = z[np.isfinite(z)]
    fill = float(np.min(finite) - 1.0) if finite.size else 0.0
    zf = np.where(np.isfinite(z), z, fill)
    out = ndimage.map_coordinates(
        zf, np.vstack([ys.ravel(), xs.ravel()]), order=int(order), mode="constant", cval=fill
    ).reshape(h, w)
    out[out <= fill + 0.5] = np.nan
    return out


def ps_hex(z, area, n_scales, device, spacing):
    d = pattern_spectrum(z, area, n_scales, kind="hex", device=device)
    d["spacing_m"] = float(spacing)
    d["stats"] = spectral_stats(d, spacing)
    d["mono"] = volumes_monotone(d)
    return d


def ps_sq(z, area, n_scales, device, spacing, connectivity=8):
    d = pattern_spectrum(z, area, n_scales, kind="square", connectivity=connectivity, device=device)
    d["spacing_m"] = float(spacing)
    d["stats"] = spectral_stats(d, spacing)
    d["mono"] = volumes_monotone(d)
    return d


def ps_sq_matched_area(z, n_scales, device, hex_spacing, sq_spacing, connectivity=8):
    """Square openings whose SE area tracks the hexagonal SE at each hex radius."""
    from src.morphology import volume

    radii = np.arange(1, n_scales + 1, dtype=int)
    area = float(sq_spacing) ** 2
    vols = [volume(z, area)]
    for r in radii:
        rs = equal_area_square_radius(int(r), hex_spacing, sq_spacing, connectivity)
        opened = grey_open_square(z, rs, connectivity=connectivity, device=device)
        vols.append(volume(opened, area))
    vols = np.asarray(vols, dtype=np.float64)
    ps = np.clip(vols[:-1] - vols[1:], 0, None)
    s = ps.sum()
    ps_n = ps / s if s > 0 else ps
    peak = int(radii[int(np.argmax(ps_n))]) if ps_n.size else 0
    d = {
        "radii": radii,
        "ps": ps_n,
        "volumes": vols,
        "lambda_peak": peak,
        "volume0": float(vols[0]),
        "spacing_m": float(sq_spacing),
    }
    d["stats"] = spectral_stats(d, sq_spacing)
    d["mono"] = volumes_monotone(d)
    return d


def drift_pack(coarse, fine, wmin, wmax):
    st_c, st_f = coarse["stats"], fine["stats"]
    return {
        "d_peak": abs(st_c["argmax_m"] - st_f["argmax_m"]) / max(st_f["argmax_m"], 1e-6),
        "d_mean": abs(st_c["mean_m"] - st_f["mean_m"]) / max(st_f["mean_m"], 1e-6),
        "d_med": abs(st_c["median_m"] - st_f["median_m"]) / max(st_f["median_m"], 1e-6),
        "d_sub": abs(st_c["subbin_peak_m"] - st_f["subbin_peak_m"]) / max(st_f["subbin_peak_m"], 1e-6),
        "l1_full": l1_spectrum_meters(coarse, coarse["spacing_m"], fine, fine["spacing_m"]),
        "l1_win": l1_spectrum_meters(
            coarse, coarse["spacing_m"], fine, fine["spacing_m"], wmin=wmin, wmax=wmax
        ),
        "w1": wasserstein1_meters(
            coarse, coarse["spacing_m"], fine, fine["spacing_m"], wmin, wmax
        ),
        "peak_c": st_c["argmax_m"],
        "peak_f": st_f["argmax_m"],
        "mean_c": st_c["mean_m"],
        "mean_f": st_f["mean_m"],
        "mono": bool(coarse["mono"] and fine["mono"]),
    }


def warp(cfg, tif, bbox, proj, res, work, tag):
    return clip_reproject_dem(
        tif, bbox["west"], bbox["south"], bbox["east"], bbox["north"], proj, res, work / f"{tag}_{int(res)}m.tif"
    )


def scene_stack(tif, proj, bbox, work, tag, n_scales, device, method="area", spacing_mode="match_s"):
    """Build 30/60/90/120 m spectra for hex (method) and square-8."""
    out = {}
    for res in (30, 60, 90, 120):
        path = warp({"x": 0}, tif, bbox, proj, res, work, tag)
        z, transform, _, _ = read_band(path)
        if spacing_mode == "equal_area":
            hex_spacing = float(res) * np.sqrt(2.0 / np.sqrt(3.0))
        else:
            hex_spacing = float(res)
        hx = build_hex_grid(z, transform, proj, size_m=hex_r_of_spacing(hex_spacing), method=method)
        hps = ps_hex(hx.values, hx.cell_area, n_scales, device, hx.spacing)
        sps = ps_sq(z, float(res) ** 2, n_scales, device, float(res), 8)
        out[int(res)] = {
            "z": z,
            "transform": transform,
            "hex": hx,
            "hps": hps,
            "sps": sps,
            "n_valid_hex": int(np.isfinite(hx.values).sum()),
            "n_valid_sq": int(np.isfinite(z).sum()),
            "relief": float(np.nanmax(z) - np.nanmin(z)),
            "slope": float(np.nanmean(np.degrees(np.arctan(np.hypot(*np.gradient(np.nan_to_num(z))))))),
        }
        print(
            f"  [{tag} {res}m {method}/{spacing_mode}] "
            f"hex_peak={hps['stats']['argmax_m']:.0f} mean={hps['stats']['mean_m']:.1f} "
            f"sq_peak={sps['stats']['argmax_m']:.0f} mean={sps['stats']['mean_m']:.1f} mono={hps['mono'] and sps['mono']}"
        )
    return out


def e1_rows(stack, tag, method, spacing_mode):
    rows = []
    fine = stack[30]
    for res in (60, 90, 120):
        wmin, wmax = float(res), 360.0
        ph = drift_pack(stack[res]["hps"], fine["hps"], wmin, wmax)
        ps = drift_pack(stack[res]["sps"], fine["sps"], wmin, wmax)
        rows.append(
            {
                "exp": "E1",
                "tag": tag,
                "method": method,
                "spacing": spacing_mode,
                "setting": f"{res} vs 30",
                **{f"hex_{k}": v for k, v in ph.items()},
                **{f"sq_{k}": v for k, v in ps.items()},
            }
        )
    return rows


def summarize_e1(rows_e1, tag, method, spacing_mode):
    hex_peak = np.mean([r["hex_d_peak"] for r in rows_e1])
    sq_peak = np.mean([r["sq_d_peak"] for r in rows_e1])
    hex_mean = np.mean([r["hex_d_mean"] for r in rows_e1])
    sq_mean = np.mean([r["sq_d_mean"] for r in rows_e1])
    hex_w1 = np.mean([r["hex_w1"] for r in rows_e1])
    sq_w1 = np.mean([r["sq_w1"] for r in rows_e1])
    hex_l1w = np.mean([r["hex_l1_win"] for r in rows_e1])
    sq_l1w = np.mean([r["sq_l1_win"] for r in rows_e1])
    return {
        "exp": "E1_summary",
        "tag": tag,
        "method": method,
        "spacing": spacing_mode,
        "setting": "mean 60-120 vs 30",
        "hex_d_peak": hex_peak,
        "sq_d_peak": sq_peak,
        "hex_d_mean": hex_mean,
        "sq_d_mean": sq_mean,
        "hex_w1": hex_w1,
        "sq_w1": sq_w1,
        "hex_l1_win": hex_l1w,
        "sq_l1_win": sq_l1w,
        "hex_mono": all(r["hex_mono"] for r in rows_e1),
        "sq_mono": all(r["sq_mono"] for r in rows_e1),
    }


def rotation_ai(z0, transform, proj, n_scales, device, method, angles):
    rmax = int(n_scales)
    mask = interior_square(z0, rmax)
    z_int = apply_mask(z0, mask)
    rows = []
    ref_h = None
    ref_s = None
    for ang in angles:
        zs = resample_square_lattice(z_int, ang, order=1)
        zs = apply_mask(zs, np.isfinite(zs) & mask)
        hx = build_hex_grid(
            z_int, transform, proj, size_m=hex_r_of_spacing(30.0), method=method, angle_deg=ang
        )
        # restrict hex centers to interior by masking cells whose value came from exterior: already from z_int
        hps = ps_hex(hx.values, hx.cell_area, n_scales, device, hx.spacing)
        sps = ps_sq(zs, 30.0 ** 2, n_scales, device, 30.0, 8)
        if ang == 0:
            ref_h, ref_s = hps, sps
        dh = wasserstein1_meters(hps, hps["spacing_m"], ref_h, ref_h["spacing_m"], 30.0, 360.0)
        ds = wasserstein1_meters(sps, sps["spacing_m"], ref_s, ref_s["spacing_m"], 30.0, 360.0)
        dh_l1 = l1_spectrum_meters(hps, hps["spacing_m"], ref_h, ref_h["spacing_m"], wmin=30, wmax=360)
        ds_l1 = l1_spectrum_meters(sps, sps["spacing_m"], ref_s, ref_s["spacing_m"], wmin=30, wmax=360)
        rows.append(
            {
                "exp": "E2",
                "setting": f"rot {ang}",
                "hex_w1": dh,
                "sq_w1": ds,
                "hex_l1_win": dh_l1,
                "sq_l1_win": ds_l1,
                "method": method,
            }
        )
        print(f"  [rot {ang:5.1f}] W1 hex={dh:.4f} sq={ds:.4f}  L1 hex={dh_l1:.4f} sq={ds_l1:.4f}")
    return rows


def tile_uncertainty(z0, transform, proj, n_scales, device, method, n=3):
    h, w = z0.shape
    hs, ws = h // n, w // n
    d_peak_h, d_peak_s, d_mean_h, d_mean_s, w1_h, w1_s = [], [], [], [], [], []
    floor = []
    for i in range(n):
        for j in range(n):
            sl = (slice(i * hs, (i + 1) * hs), slice(j * ws, (j + 1) * ws))
            z = z0[sl]
            if np.isfinite(z).sum() < 400:
                continue
            # fake local transform
            from affine import Affine

            tloc = Affine(transform.a, transform.b, transform.c + j * ws * transform.a, transform.d, transform.e, transform.f + i * hs * transform.e)
            hx30 = build_hex_grid(z, tloc, proj, size_m=hex_r_of_spacing(30.0), method=method)
            # coarsen tile by 2x average
            z60 = ndimage.zoom(np.nan_to_num(z, nan=np.nanmean(z)), 0.5, order=1)
            # mask
            z60 = z60.astype(np.float64)
            hx60 = build_hex_grid(z60, Affine(60, 0, tloc.c, 0, -60, tloc.f), proj, size_m=hex_r_of_spacing(60.0), method=method)
            h30 = ps_hex(hx30.values, hx30.cell_area, n_scales, device, hx30.spacing)
            h60 = ps_hex(hx60.values, hx60.cell_area, n_scales, device, hx60.spacing)
            s30 = ps_sq(z, 30.0 ** 2, n_scales, device, 30.0)
            s60 = ps_sq(z60, 60.0 ** 2, n_scales, device, 60.0)
            ph = drift_pack(h60, h30, 60.0, 360.0)
            ps = drift_pack(s60, s30, 60.0, 360.0)
            d_peak_h.append(ph["d_peak"])
            d_peak_s.append(ps["d_peak"])
            d_mean_h.append(ph["d_mean"])
            d_mean_s.append(ps["d_mean"])
            w1_h.append(ph["w1"])
            w1_s.append(ps["w1"])
    # noise floor: half-cell hex offset at 30 m on full window, W1 between dq=0 and dq=0.5
    h0 = build_hex_grid(z0, transform, proj, size_m=hex_r_of_spacing(30.0), method=method, dq=0.0)
    h1 = build_hex_grid(z0, transform, proj, size_m=hex_r_of_spacing(30.0), method=method, dq=0.5)
    p0 = ps_hex(h0.values, h0.cell_area, n_scales, device, h0.spacing)
    p1 = ps_hex(h1.values, h1.cell_area, n_scales, device, h1.spacing)
    floor.append(wasserstein1_meters(p0, p0["spacing_m"], p1, p1["spacing_m"], 30.0, 360.0))
    h2 = build_hex_grid(z0, transform, proj, size_m=hex_r_of_spacing(30.0), method=method, dr=0.5)
    p2 = ps_hex(h2.values, h2.cell_area, n_scales, device, h2.spacing)
    floor.append(wasserstein1_meters(p0, p0["spacing_m"], p2, p2["spacing_m"], 30.0, 360.0))
    return {
        "n_tiles": len(d_mean_h),
        "hex_d_mean_med": float(np.median(d_mean_h)) if d_mean_h else np.nan,
        "hex_d_mean_iqr": float(np.subtract(*np.percentile(d_mean_h, [75, 25]))) if d_mean_h else np.nan,
        "sq_d_mean_med": float(np.median(d_mean_s)) if d_mean_s else np.nan,
        "sq_d_mean_iqr": float(np.subtract(*np.percentile(d_mean_s, [75, 25]))) if d_mean_s else np.nan,
        "hex_d_peak_med": float(np.median(d_peak_h)) if d_peak_h else np.nan,
        "sq_d_peak_med": float(np.median(d_peak_s)) if d_peak_s else np.nan,
        "hex_w1_med": float(np.median(w1_h)) if w1_h else np.nan,
        "sq_w1_med": float(np.median(w1_s)) if w1_s else np.nan,
        "hex_w1_iqr": float(np.subtract(*np.percentile(w1_h, [75, 25]))) if w1_h else np.nan,
        "sq_w1_iqr": float(np.subtract(*np.percentile(w1_s, [75, 25]))) if w1_s else np.nan,
        "noise_floor_w1": float(np.mean(floor)),
        "tiles_hex_d_mean": d_mean_h,
        "tiles_sq_d_mean": d_mean_s,
    }


def se_area_table(spacing=30.0):
    rows = []
    for r in (1, 2, 3, 5, 8, 12):
        rows.append(
            {
                "r": r,
                "hex_m2": se_area_m2("hex", r, spacing),
                "sq8_m2": se_area_m2("square", r, spacing, 8),
                "sq4_m2": se_area_m2("square", r, spacing, 4),
                "matched_sq8_r": equal_area_square_radius(r, spacing, spacing, 8),
            }
        )
    return rows


def write_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    cfg = load_config(ROOT / "config.yaml")
    data = ROOT / "data"
    out = ROOT / "outputs" / "revision"
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device("cpu")
    n_scales = 12
    study = cfg["study"]
    t0 = time.time()

    windows = [
        {
            "tag": "yanhe_loess",
            "tif": data / "yanhe_loess_cop30.tif",
            "proj": study["proj"],
            "bbox": {"west": study["west"], "south": study["south"], "east": study["east"], "north": study["north"]},
        },
        {
            "tag": "yanhe_north",
            "tif": data / "yanhe_loess_cop30.tif",
            "proj": study["proj"],
            "bbox": {"west": study["west"], "south": 36.65, "east": study["east"], "north": study["north"]},
        },
        {
            "tag": "sichuan_hills",
            "tif": data / "sichuan_hills_cop30.tif",
            "proj": study["confirm_proj"],
            "bbox": {
                "west": study["confirm_west"],
                "south": study["confirm_south"],
                "east": study["confirm_east"],
                "north": study["confirm_north"],
            },
        },
    ]

    all_rows = []
    stacks = {}
    print("=== SE areas ===")
    se_rows = se_area_table(30.0)
    write_csv(se_rows, out / "se_areas.csv")
    for r in se_rows:
        print(r)

    for win in windows:
        work = out / "work" / win["tag"]
        work.mkdir(parents=True, exist_ok=True)
        print(f"=== stack area-weighted {win['tag']} ===")
        st = scene_stack(win["tif"], win["proj"], win["bbox"], work, win["tag"], n_scales, device, method="area")
        stacks[win["tag"]] = st
        e1 = e1_rows(st, win["tag"], "area", "match_s")
        all_rows.extend(e1)
        all_rows.append(summarize_e1(e1, win["tag"], "area", "match_s"))
        all_rows.append(
            {
                "exp": "valid_counts",
                "tag": win["tag"],
                "n_valid_hex": st[30]["n_valid_hex"],
                "n_valid_sq": st[30]["n_valid_sq"],
                "relief_m": st[30]["relief"],
                "mean_slope_deg": st[30]["slope"],
                "frac_hex_valid": st[30]["n_valid_hex"] / max(st[30]["hex"].values.size, 1),
            }
        )

    # bilinear confound on main window only
    win0 = windows[0]
    work = out / "work" / "yanhe_bilinear"
    print("=== stack bilinear yanhe ===")
    st_bi = scene_stack(win0["tif"], win0["proj"], win0["bbox"], work, "yanhe_bilinear", n_scales, device, method="bilinear")
    e1b = e1_rows(st_bi, "yanhe_loess", "bilinear", "match_s")
    all_rows.extend(e1b)
    all_rows.append(summarize_e1(e1b, "yanhe_loess", "bilinear", "match_s"))

    print("=== nearest-neighbour hex yanhe ===")
    st_nn = scene_stack(win0["tif"], win0["proj"], win0["bbox"], out / "work" / "yanhe_nn", "yanhe_nn", n_scales, device, method="bilinear")
    # rebuild 30/60 with order=0 by extra calls
    # (scene_stack bilinear is order=1; add explicit 30 vs 120 nearest)
    path30 = list((out / "work" / "yanhe_bilinear").glob("*_30m.tif"))
    # skip dedicated nn stack if too costly; sample 30 and 120 only
    z30, t30, _, _ = read_band(out / "work" / "yanhe_loess" / "yanhe_loess_30m.tif")
    z120, t120, _, _ = read_band(out / "work" / "yanhe_loess" / "yanhe_loess_120m.tif")
    hx_nn30 = build_hex_grid(z30, t30, win0["proj"], size_m=hex_r_of_spacing(30.0), method="bilinear", order=0)
    hx_nn120 = build_hex_grid(z120, t120, win0["proj"], size_m=hex_r_of_spacing(120.0), method="bilinear", order=0)
    pnn30 = ps_hex(hx_nn30.values, hx_nn30.cell_area, n_scales, device, hx_nn30.spacing)
    pnn120 = ps_hex(hx_nn120.values, hx_nn120.cell_area, n_scales, device, hx_nn120.spacing)
    pack_nn = drift_pack(pnn120, pnn30, 120.0, 360.0)
    all_rows.append({"exp": "resample_nn", "tag": "yanhe_loess", "setting": "120 vs 30", **{f"hex_{k}": v for k, v in pack_nn.items()}})

    print("=== equal-area spacing yanhe ===")
    st_ea = scene_stack(
        win0["tif"], win0["proj"], win0["bbox"], out / "work" / "yanhe_eqarea", "yanhe_eqarea", n_scales, device, method="area", spacing_mode="equal_area"
    )
    e1e = e1_rows(st_ea, "yanhe_loess", "area", "equal_area")
    all_rows.extend(e1e)
    all_rows.append(summarize_e1(e1e, "yanhe_loess", "area", "equal_area"))

    print("=== equal-area SE square vs hex yanhe 30-120 ===")
    st = stacks["yanhe_loess"]
    matched = ps_sq_matched_area(st[30]["z"], n_scales, device, 30.0, 30.0, 8)
    matched120 = ps_sq_matched_area(st[120]["z"], n_scales, device, 120.0, 120.0, 8)
    pm = drift_pack(matched120, matched, 120.0, 360.0)
    all_rows.append({"exp": "E3_matchedSE", "tag": "yanhe_loess", "setting": "120 vs 30", **{f"sq_{k}": v for k, v in pm.items()}})
    ph = drift_pack(st[120]["hps"], st[30]["hps"], 120.0, 360.0)
    all_rows.append({"exp": "E3_matchedSE_hexref", "tag": "yanhe_loess", **{f"hex_{k}": v for k, v in ph.items()}})

    print("=== 4 vs 8 neighbourhood, W1 common support ===")
    for tag in ("yanhe_loess", "sichuan_hills"):
        st = stacks[tag]
        s4_30 = ps_sq(st[30]["z"], 30.0 ** 2, n_scales, device, 30.0, 4)
        s4_60 = ps_sq(st[60]["z"], 60.0 ** 2, n_scales, device, 60.0, 4)
        p4 = drift_pack(s4_60, s4_30, 60.0, 360.0)
        p8 = drift_pack(st[60]["sps"], st[30]["sps"], 60.0, 360.0)
        ph = drift_pack(st[60]["hps"], st[30]["hps"], 60.0, 360.0)
        all_rows.append(
            {
                "exp": "E3",
                "tag": tag,
                "hex_w1": ph["w1"],
                "sq8_w1": p8["w1"],
                "sq4_w1": p4["w1"],
                "hex_d_mean": ph["d_mean"],
                "sq8_d_mean": p8["d_mean"],
                "sq4_d_mean": p4["d_mean"],
            }
        )

    print("=== rotation lattice + interior mask ===")
    st = stacks["yanhe_loess"]
    rot_rows = rotation_ai(st[30]["z"], st[30]["transform"], windows[0]["proj"], n_scales, device, "area", [0, 10, 20, 30, 45, 60])
    for r in rot_rows:
        r["tag"] = "yanhe_loess"
        all_rows.append(r)
    ai_h = [r["hex_w1"] for r in rot_rows]
    ai_s = [r["sq_w1"] for r in rot_rows]
    all_rows.append(
        {
            "exp": "E2_summary",
            "tag": "yanhe_loess",
            "hex_max_w1": float(np.max(ai_h)),
            "sq_max_w1": float(np.max(ai_s)),
            "hex_mean_w1": float(np.mean(ai_h)),
            "sq_mean_w1": float(np.mean(ai_s)),
        }
    )

    print("=== 3x3 tiles + offset noise floor ===")
    unc = tile_uncertainty(st[30]["z"], st[30]["transform"], windows[0]["proj"], n_scales, device, "area", n=3)
    unc_row = {k: v for k, v in unc.items() if not k.startswith("tiles_")}
    unc_row["exp"] = "M6_tiles"
    unc_row["tag"] = "yanhe_loess"
    all_rows.append(unc_row)
    print(unc_row)

    write_csv(all_rows, out / "revision_table.csv")
    meta = {"seconds": round(time.time() - t0, 1), "device": device, "n_rows": len(all_rows)}
    (out / "revision_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    # dump tile lists
    (out / "tile_means.json").write_text(
        json.dumps({"hex": unc["tiles_hex_d_mean"], "sq": unc["tiles_sq_d_mean"]}, indent=2), encoding="utf-8"
    )
    print(json.dumps(meta, indent=2))
    print("->", out / "revision_table.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
