#!/usr/bin/env python3
"""Nine-tile 30-to-60 m W1 bootstrap (supplementary; not in the 5-page letter)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
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

from affine import Affine  # noqa: E402
from src.experiments import load_config, pick_device, read_band  # noqa: E402
from src.hex_grid import build_hex_grid  # noqa: E402
from run_revision import drift_pack, hex_r_of_spacing, ps_hex, ps_sq  # noqa: E402


def main():
    cfg = load_config(ROOT / "config.yaml")
    device = pick_device("cpu")
    proj = cfg["study"]["proj"]
    z0, transform, _, _ = read_band(
        ROOT / "outputs" / "revision" / "work" / "yanhe_loess" / "yanhe_loess_30m.tif"
    )
    n, n_scales = 3, 12
    h, w = z0.shape
    hs, ws = h // n, w // n
    hex_w1, sq_w1 = [], []
    for i in range(n):
        for j in range(n):
            z = z0[i * hs : (i + 1) * hs, j * ws : (j + 1) * ws]
            tloc = Affine(
                transform.a,
                transform.b,
                transform.c + j * ws * transform.a,
                transform.d,
                transform.e,
                transform.f + i * hs * transform.e,
            )
            hx30 = build_hex_grid(z, tloc, proj, size_m=hex_r_of_spacing(30.0), method="area")
            z60 = ndimage.zoom(np.nan_to_num(z, nan=np.nanmean(z)), 0.5, order=1).astype(np.float64)
            hx60 = build_hex_grid(
                z60, Affine(60, 0, tloc.c, 0, -60, tloc.f), proj, size_m=hex_r_of_spacing(60.0), method="area"
            )
            h30 = ps_hex(hx30.values, hx30.cell_area, n_scales, device, hx30.spacing)
            h60 = ps_hex(hx60.values, hx60.cell_area, n_scales, device, hx60.spacing)
            s30 = ps_sq(z, 30.0**2, n_scales, device, 30.0)
            s60 = ps_sq(z60, 60.0**2, n_scales, device, 60.0)
            ph = drift_pack(h60, h30, 60.0, 360.0)
            ps = drift_pack(s60, s30, 60.0, 360.0)
            hex_w1.append(ph["w1"])
            sq_w1.append(ps["w1"])
            print(f"  tile {i},{j} W1 hex={ph['w1']:.3f} sq={ps['w1']:.3f}")

    hex_w1 = np.asarray(hex_w1, dtype=np.float64)
    sq_w1 = np.asarray(sq_w1, dtype=np.float64)
    rng = np.random.default_rng(20261101)
    B = 10000
    idx = rng.integers(0, len(hex_w1), size=(B, len(hex_w1)))
    boot_h = np.median(hex_w1[idx], axis=1)
    boot_s = np.median(sq_w1[idx], axis=1)
    boot_d = boot_h - boot_s
    out = {
        "n_tiles": int(len(hex_w1)),
        "B": B,
        "hex_w1": hex_w1.tolist(),
        "sq_w1": sq_w1.tolist(),
        "hex_median": float(np.median(hex_w1)),
        "sq_median": float(np.median(sq_w1)),
        "hex_iqr": float(np.subtract(*np.percentile(hex_w1, [75, 25]))),
        "sq_iqr": float(np.subtract(*np.percentile(sq_w1, [75, 25]))),
        "hex_median_ci95": [float(np.percentile(boot_h, 2.5)), float(np.percentile(boot_h, 97.5))],
        "sq_median_ci95": [float(np.percentile(boot_s, 2.5)), float(np.percentile(boot_s, 97.5))],
        "diff_median_ci95": [float(np.percentile(boot_d, 2.5)), float(np.percentile(boot_d, 97.5))],
        "note": "Spatial-block bootstrap of nine 30-to-60 m tiles (resample tiles, not a whole-window 30-120 m bootstrap).",
    }
    path = ROOT / "outputs" / "revision" / "tile_w1_bootstrap.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in out if k not in ("hex_w1", "sq_w1")}, indent=2))
    print("->", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
