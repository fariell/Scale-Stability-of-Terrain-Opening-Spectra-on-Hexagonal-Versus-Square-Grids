"""Download Copernicus GLO-30 (COP30) via OpenTopography, with NASADEM fallback."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import requests


OT_URL = "https://portal.opentopography.org/API/globaldem"


class DownloadError(RuntimeError):
    pass


def _api_key() -> str:
    key = os.environ.get("OPENTOPO_API_KEY", "").strip()
    if not key:
        raise DownloadError(
            "Missing OPENTOPO_API_KEY. Register at https://portal.opentopography.org "
            "→ MyOpenTopo → Request API key, then: export OPENTOPO_API_KEY=..."
        )
    return key


def download_dem(
    west: float,
    south: float,
    east: float,
    north: float,
    out_tif: Path,
    demtype: str = "COP30",
    timeout: int = 600,
) -> Path:
    """Write a GeoTIFF clipped to the WGS84 box. OpenTopography limit is ~1e6 km²."""
    out_tif = Path(out_tif)
    out_tif.parent.mkdir(parents=True, exist_ok=True)
    if out_tif.exists() and out_tif.stat().st_size > 1024:
        return out_tif

    params = {
        "demtype": demtype,
        "south": f"{south:.6f}",
        "north": f"{north:.6f}",
        "west": f"{west:.6f}",
        "east": f"{east:.6f}",
        "outputFormat": "GTiff",
        "API_KEY": _api_key(),
    }
    print(f"[download] {demtype} {west},{south} – {east},{north}")
    r = requests.get(OT_URL, params=params, timeout=timeout, stream=True)
    if r.status_code != 200:
        raise DownloadError(f"OpenTopography HTTP {r.status_code}: {r.text[:400]}")
    tmp = out_tif.with_suffix(".partial")
    with tmp.open("wb") as f:
        for chunk in r.iter_content(1 << 20):
            if chunk:
                f.write(chunk)
    tmp.replace(out_tif)
    print(f"[download] wrote {out_tif} ({out_tif.stat().st_size/1e6:.1f} MB)")
    return out_tif


def write_synthetic_dem(out_tif: Path, cfg: dict) -> Path:
    """Procedural loess-like DEM so the pipeline can run without an API key."""
    import rasterio
    from rasterio.transform import from_origin

    out_tif = Path(out_tif)
    out_tif.parent.mkdir(parents=True, exist_ok=True)
    syn = cfg["synthetic"]
    nx, ny, cell = int(syn["nx"]), int(syn["ny"]), float(syn["cell_m"])
    rng = np.random.default_rng(7)
    y, x = np.mgrid[0:ny, 0:nx]
    # Broad hills + dendritic channels + noise
    hills = 180 * np.exp(-((x - nx * 0.5) ** 2 + (y - ny * 0.45) ** 2) / (2 * (nx * 0.28) ** 2))
    hills += 90 * np.sin(x / 28.0) * np.cos(y / 41.0)
    channel = np.minimum(
        np.abs((y - 0.002 * (x - nx / 2) ** 2) - ny * 0.55),
        np.abs(x - nx * 0.62),
    )
    z = 1100 + hills - 140 * np.exp(-channel / 6.0)
    z += rng.normal(0, 1.8, z.shape)
    # gullies
    for cx, cy, w, d in [(0.3, 0.7, 8, 55), (0.75, 0.35, 6, 40), (0.5, 0.2, 10, 35)]:
        z -= d * np.exp(-((x - nx * cx) ** 2 + (y - ny * cy) ** 2) / (2 * w**2))

    west, north = 109.28, 36.78
    transform = from_origin(west, north, 0.00027, 0.00027)  # ~30 m
    profile = {
        "driver": "GTiff",
        "height": ny,
        "width": nx,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": -9999.0,
        "compress": "lzw",
    }
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(z.astype("float32"), 1)
    print(f"[download] synthetic DEM {out_tif}")
    return out_tif
