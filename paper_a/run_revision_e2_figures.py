#!/usr/bin/env python3
"""Fix-up: lattice rotation / offset floor + GRSL figures (uses existing warps)."""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYTHONUNBUFFERED", "1")


def _fix_windows_proj() -> None:
    prefix = Path(sys.prefix)
    for cand in (prefix / "Library" / "share" / "proj", prefix / "share" / "proj"):
        if (cand / "proj.db").exists():
            os.environ.setdefault("PROJ_DATA", str(cand))
            os.environ.setdefault("PROJ_LIB", str(cand))
            break


_fix_windows_proj()

from src.experiments import load_config, pick_device, read_band  # noqa: E402
from src.hex_grid import build_hex_grid  # noqa: E402
from src.morphology import (  # noqa: E402
    l1_spectrum_meters,
    pattern_spectrum,
    resample_ps_to_meters,
    spectral_stats,
    wasserstein1_meters,
)

from run_revision import (  # noqa: E402
    apply_mask,
    hex_r_of_spacing,
    interior_square,
    ps_hex,
    ps_sq,
    resample_square_lattice,
)


def dump_spec(d):
    return {
        "radii": d["radii"].tolist(),
        "ps": d["ps"].tolist(),
        "spacing_m": float(d["spacing_m"]),
        "stats": d["stats"],
        "lambda_peak": int(d["lambda_peak"]),
    }


def rotation_block(z0, transform, proj, n_scales, device, method, angles):
    mask = interior_square(z0, n_scales)
    z_int = apply_mask(z0, mask)
    ref_h = ref_s = None
    rows = []
    for ang in angles:
        zs = resample_square_lattice(z_int, ang, order=1)
        zs = apply_mask(zs, np.isfinite(zs) & mask)
        hx = build_hex_grid(
            z_int, transform, proj, size_m=hex_r_of_spacing(30.0), method=method, angle_deg=ang
        )
        hps = ps_hex(hx.values, hx.cell_area, n_scales, device, hx.spacing)
        sps = ps_sq(zs, 30.0**2, n_scales, device, 30.0, 8)
        if ang == 0:
            ref_h, ref_s = hps, sps
        rows.append(
            {
                "method": method,
                "ang": ang,
                "hex_w1": wasserstein1_meters(hps, hps["spacing_m"], ref_h, ref_h["spacing_m"], 30, 360),
                "sq_w1": wasserstein1_meters(sps, sps["spacing_m"], ref_s, ref_s["spacing_m"], 30, 360),
                "hex_l1": l1_spectrum_meters(hps, hps["spacing_m"], ref_h, ref_h["spacing_m"], wmin=30, wmax=360),
                "sq_l1": l1_spectrum_meters(sps, sps["spacing_m"], ref_s, ref_s["spacing_m"], wmin=30, wmax=360),
                "hex_peak": hps["stats"]["argmax_m"],
                "sq_peak": sps["stats"]["argmax_m"],
            }
        )
        print(
            f"  [{method} rot {ang:5.1f}] W1 hex={rows[-1]['hex_w1']:.4f} sq={rows[-1]['sq_w1']:.4f} "
            f"L1 hex={rows[-1]['hex_l1']:.4f} sq={rows[-1]['sq_l1']:.4f}"
        )
    return rows


def dem_rotate_block(z0, transform, proj, n_scales, device, angles):
    """Original protocol: rotate the square DEM, then sample both operators."""
    rows = []
    h, w = z0.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    finite = z0[np.isfinite(z0)]
    fill = float(np.min(finite) - 1.0)
    zf = np.where(np.isfinite(z0), z0, fill)
    ref_h = ref_s = None
    for ang in angles:
        zrot = ndimage.rotate(zf, ang, reshape=False, order=1, cval=fill)
        zrot[zrot <= fill + 0.5] = np.nan
        hx = build_hex_grid(zrot, transform, proj, size_m=hex_r_of_spacing(30.0), method="bilinear")
        hps = ps_hex(hx.values, hx.cell_area, n_scales, device, hx.spacing)
        sps = ps_sq(zrot, 30.0**2, n_scales, device, 30.0, 8)
        if ang == 0:
            ref_h, ref_s = hps, sps
        rows.append(
            {
                "method": "dem_rotate_bilinear",
                "ang": ang,
                "hex_w1": wasserstein1_meters(hps, hps["spacing_m"], ref_h, ref_h["spacing_m"], 30, 360),
                "sq_w1": wasserstein1_meters(sps, sps["spacing_m"], ref_s, ref_s["spacing_m"], 30, 360),
                "hex_l1": l1_spectrum_meters(hps, hps["spacing_m"], ref_h, ref_h["spacing_m"], wmin=30, wmax=360),
                "sq_l1": l1_spectrum_meters(sps, sps["spacing_m"], ref_s, ref_s["spacing_m"], wmin=30, wmax=360),
            }
        )
        print(
            f"  [DEM-rot {ang:5.1f}] W1 hex={rows[-1]['hex_w1']:.4f} sq={rows[-1]['sq_w1']:.4f} "
            f"L1 hex={rows[-1]['hex_l1']:.4f} sq={rows[-1]['sq_l1']:.4f}"
        )
    return rows


def ieee_style():
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.6,
            "figure.dpi": 150,
        }
    )


def hillshade(z, cell=30.0, az=315.0, alt=45.0):
    dy, dx = np.gradient(np.nan_to_num(z, nan=np.nanmean(z)), cell)
    slope = np.pi / 2.0 - np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    azr, altr = np.radians(az), np.radians(alt)
    hs = np.sin(altr) * np.sin(slope) + np.cos(altr) * np.cos(slope) * np.cos(azr - aspect)
    hs[~np.isfinite(z)] = np.nan
    return hs


def main():
    cfg = load_config(ROOT / "config.yaml")
    device = pick_device("cpu")
    n_scales = 12
    proj = cfg["study"]["proj"]
    out = ROOT / "outputs" / "revision"
    figdir = Path(r"C:\Users\Administrator\Desktop\地形认知\grsl_hex_spectrum\figures")
    figdir.mkdir(parents=True, exist_ok=True)

    z30, t30, _, _ = read_band(out / "work" / "yanhe_loess" / "yanhe_loess_30m.tif")
    print("=== lattice rotation (area) ===")
    rot_area = rotation_block(z30, t30, proj, n_scales, device, "area", [0, 10, 20, 30, 45, 60])
    print("=== lattice rotation (bilinear) ===")
    rot_bi = rotation_block(z30, t30, proj, n_scales, device, "bilinear", [0, 10, 20, 30, 45, 60])
    print("=== DEM rotation (original protocol) ===")
    rot_dem = dem_rotate_block(z30, t30, proj, n_scales, device, [0, 15, 30, 45])

    print("=== offset noise floor ===")
    floors = []
    for method in ("area", "bilinear"):
        h0 = build_hex_grid(z30, t30, proj, size_m=hex_r_of_spacing(30.0), method=method, dq=0.0)
        p0 = ps_hex(h0.values, h0.cell_area, n_scales, device, h0.spacing)
        for key, kw in (("dq0.5", {"dq": 0.5}), ("dr0.5", {"dr": 0.5})):
            h1 = build_hex_grid(z30, t30, proj, size_m=hex_r_of_spacing(30.0), method=method, **kw)
            p1 = ps_hex(h1.values, h1.cell_area, n_scales, device, h1.spacing)
            w1 = wasserstein1_meters(p0, p0["spacing_m"], p1, p1["spacing_m"], 30, 360)
            floors.append({"method": method, "shift": key, "w1": w1})
            print(f"  floor {method} {key} W1={w1:.4f}")

    # spectra for figures
    specs = {"hex_area": {}, "hex_bilinear": {}, "square": {}}
    means = {"hex_area": {}, "hex_bilinear": {}, "square": {}}
    peaks = {"hex_area": {}, "hex_bilinear": {}, "square": {}}
    for res, tag in [(30, "yanhe_loess"), (60, "yanhe_loess"), (90, "yanhe_loess"), (120, "yanhe_loess")]:
        z, tr, _, _ = read_band(out / "work" / "yanhe_loess" / f"yanhe_loess_{res}m.tif")
        hx_a = build_hex_grid(z, tr, proj, size_m=hex_r_of_spacing(float(res)), method="area")
        hx_b = build_hex_grid(z, tr, proj, size_m=hex_r_of_spacing(float(res)), method="bilinear")
        ha = ps_hex(hx_a.values, hx_a.cell_area, n_scales, device, hx_a.spacing)
        hb = ps_hex(hx_b.values, hx_b.cell_area, n_scales, device, hx_b.spacing)
        sq = ps_sq(z, float(res) ** 2, n_scales, device, float(res), 8)
        specs["hex_area"][res] = dump_spec(ha)
        specs["hex_bilinear"][res] = dump_spec(hb)
        specs["square"][res] = dump_spec(sq)
        peaks["hex_area"][res] = ha["stats"]["argmax_m"]
        peaks["hex_bilinear"][res] = hb["stats"]["argmax_m"]
        peaks["square"][res] = sq["stats"]["argmax_m"]
        means["hex_area"][res] = ha["stats"]["mean_m"]
        means["hex_bilinear"][res] = hb["stats"]["mean_m"]
        means["square"][res] = sq["stats"]["mean_m"]
        print(f"  spec {res}m area_peak={ha['stats']['argmax_m']:.0f} bi_peak={hb['stats']['argmax_m']:.0f} sq={sq['stats']['argmax_m']:.0f}")

    payload = {
        "rot_area": rot_area,
        "rot_bilinear": rot_bi,
        "rot_dem": rot_dem,
        "floors": floors,
        "peaks": peaks,
        "means": means,
    }
    (out / "e2_rotation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out / "spectra_yanhe.json").write_text(json.dumps(specs, indent=2), encoding="utf-8")

    ieee_style()
    # Fig 2: hillshade + 30 m spectra
    hs = hillshade(z30, 30.0)
    fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.55), gridspec_kw={"width_ratios": [1.15, 1]})
    im = ax[0].imshow(hs, cmap="gray", vmin=-0.2, vmax=1.0)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_title("(a) Yan River, Copernicus GLO-30")
    w = np.arange(1, 13) * 30.0
    ax[1].plot(w, specs["hex_area"][30]["ps"], "-o", ms=3.5, color="#1b9e77", label="hex, area-weighted")
    ax[1].plot(w, specs["hex_bilinear"][30]["ps"], "--s", ms=3.5, color="#d95f02", label="hex, bilinear")
    ax[1].plot(w, specs["square"][30]["ps"], ":^", ms=3.5, color="#7570b3", label="square, 8-neigh.")
    ax[1].set_xlabel("Opening width $w=rs$ (m)")
    ax[1].set_ylabel("Opening spectrum $PS(r)$")
    ax[1].legend(frameon=False, loc="upper right")
    ax[1].set_title("(b) 30 m pattern spectra")
    ax[1].set_xlim(30, 360)
    fig.tight_layout()
    fig.savefig(figdir / "fig2_dem_spectrum.pdf")
    fig.savefig(figdir / "fig2_dem_spectrum.png", dpi=200)
    plt.close(fig)

    # Fig 3: peak, mean, Nyquist
    res = np.array([30, 60, 90, 120], dtype=float)
    fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.6))
    ax[0].plot(res, [peaks["hex_area"][int(r)] for r in res], "-o", color="#1b9e77", label="hex, area")
    ax[0].plot(res, [peaks["hex_bilinear"][int(r)] for r in res], "--s", color="#d95f02", label="hex, bilinear")
    ax[0].plot(res, [peaks["square"][int(r)] for r in res], ":^", color="#7570b3", label="square 8-neigh.")
    ax[0].plot(res, res, "k--", lw=0.8, label="Nyquist cell $s$")
    ax[0].set_xlabel("Cell size $s$ (m)")
    ax[0].set_ylabel(r"Peak scale $\lambda$ (m)")
    ax[0].set_title("(a) Discrete peak")
    ax[0].legend(frameon=False, fontsize=6)
    ax[1].plot(res, [means["hex_area"][int(r)] for r in res], "-o", color="#1b9e77", label="hex, area")
    ax[1].plot(res, [means["hex_bilinear"][int(r)] for r in res], "--s", color="#d95f02", label="hex, bilinear")
    ax[1].plot(res, [means["square"][int(r)] for r in res], ":^", color="#7570b3", label="square 8-neigh.")
    ax[1].set_xlabel("Cell size $s$ (m)")
    ax[1].set_ylabel(r"Spectral mean $\bar{w}$ (m)")
    ax[1].set_title("(b) Spectral mean")
    fig.tight_layout()
    fig.savefig(figdir / "fig3_scale.pdf")
    fig.savefig(figdir / "fig3_scale.png", dpi=200)
    plt.close(fig)

    # Fig 4: rotation
    fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.55))
    aa = [r["ang"] for r in rot_area]
    ax[0].plot(aa, [r["hex_w1"] for r in rot_area], "-o", color="#1b9e77", label="hex lattice, area")
    ax[0].plot(aa, [r["hex_w1"] for r in rot_bi], "--s", color="#d95f02", label="hex lattice, bilinear")
    ax[0].plot(aa, [r["sq_w1"] for r in rot_area], ":^", color="#7570b3", label="square lattice")
    ax[0].set_xlabel(r"Lattice azimuth $\theta$ ($^\circ$)")
    ax[0].set_ylabel(r"$W_1$ to $0^\circ$ (m)")
    ax[0].set_title("(a) Lattice rotation")
    ax[0].legend(frameon=False, fontsize=6)
    ad = [r["ang"] for r in rot_dem]
    ax[1].plot(ad, [r["hex_w1"] for r in rot_dem], "-o", color="#d95f02", label="hex from rotated DEM")
    ax[1].plot(ad, [r["sq_w1"] for r in rot_dem], ":^", color="#7570b3", label="square rotated DEM")
    ax[1].set_xlabel(r"DEM rotation $\theta$ ($^\circ$)")
    ax[1].set_ylabel(r"$W_1$ to $0^\circ$ (m)")
    ax[1].set_title("(b) DEM rotation (shared mask)")
    ax[1].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(figdir / "fig4_rotation.pdf")
    fig.savefig(figdir / "fig4_rotation.png", dpi=200)
    plt.close(fig)

    # compact e2 csv
    keys = ["protocol", "method", "ang", "hex_w1", "sq_w1", "hex_l1", "sq_l1"]
    with (out / "e2_rotation.csv").open("w", newline="", encoding="utf-8") as f:
        wri = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        wri.writeheader()
        for r in rot_area:
            wri.writerow({"protocol": "lattice", **r})
        for r in rot_bi:
            wri.writerow({"protocol": "lattice", **r})
        for r in rot_dem:
            wri.writerow({"protocol": "dem", **r})
        for r in floors:
            wri.writerow({"protocol": "offset", "method": r["method"], "ang": r["shift"], "hex_w1": r["w1"]})
    print("wrote", out / "e2_rotation.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
