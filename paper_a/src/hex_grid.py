"""Equal-area hexagonal lattice on an already-projected DEM raster."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass
class HexGrid:
    q0: int
    r0: int
    size: float  # circumradius R (m)
    nrows: int
    ncols: int
    proj: str
    values: np.ndarray
    mask: np.ndarray

    @property
    def cell_area(self) -> float:
        return (3.0 * np.sqrt(3.0) / 2.0) * self.size**2

    @property
    def spacing(self) -> float:
        return np.sqrt(3.0) * self.size


def axial_to_offset(q: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    col = q + (r - (r & 1)) // 2
    row = r
    return row.astype(int), col.astype(int)


def offset_to_axial(row: np.ndarray, col: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q = col - (row - (row & 1)) // 2
    r = row
    return q, r


def _xy_to_qr(x, y, size_m: float):
    qf = (np.sqrt(3.0) / 3.0 * x - 1.0 / 3.0 * y) / size_m
    rf = (2.0 / 3.0 * y) / size_m
    return qf, rf


def axial_round(qf: np.ndarray, rf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Round fractional axial coordinates to the nearest hex (cube rounding)."""
    x = np.asarray(qf, dtype=np.float64)
    z = np.asarray(rf, dtype=np.float64)
    y = -x - z
    rx = np.round(x)
    ry = np.round(y)
    rz = np.round(z)
    dx = np.abs(rx - x)
    dy = np.abs(ry - y)
    dz = np.abs(rz - z)
    cond_x = (dx > dy) & (dx > dz)
    cond_y = (~cond_x) & (dy > dz)
    rx = np.where(cond_x, -ry - rz, rx)
    ry = np.where(cond_y, -rx - rz, ry)
    rz = np.where(~cond_x & ~cond_y, -rx - ry, rz)
    return rx.astype(int), rz.astype(int)


def hex_lattice_extent(z: np.ndarray, transform, size_m: float, pad: int = 2):
    h, w = z.shape
    xs = np.array([transform.c, transform.c + w * transform.a], dtype=np.float64)
    ys = np.array([transform.f, transform.f + h * transform.e], dtype=np.float64)
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())
    corners_x = np.array([xmin, xmax, xmin, xmax])
    corners_y = np.array([ymin, ymin, ymax, ymax])
    qf, rf = _xy_to_qr(corners_x, corners_y, size_m)
    q0 = int(np.floor(qf.min())) - pad
    r0 = int(np.floor(rf.min())) - pad
    q1 = int(np.ceil(qf.max())) + pad
    r1 = int(np.ceil(rf.max())) + pad
    qq, rr = np.meshgrid(np.arange(q0, q1 + 1), np.arange(r0, r1 + 1))
    row_b, col_b = axial_to_offset(qq - q0, rr - r0)
    nrows = int(row_b.max()) + 1
    ncols = int(col_b.max()) + 1
    return q0, r0, nrows, ncols


def hex_center_xy(q0, r0, nrows, ncols, size_m, dq=0.0, dr=0.0, angle_deg=0.0, origin_xy=None):
    rows, cols = np.mgrid[0:nrows, 0:ncols]
    q, r = offset_to_axial(rows, cols)
    q = q + q0 + float(dq)
    r = r + r0 + float(dr)
    x = size_m * (np.sqrt(3.0) * (q + r / 2.0))
    y = size_m * (1.5 * r)
    if abs(angle_deg) > 1e-12:
        ox, oy = origin_xy
        rad = np.deg2rad(angle_deg)
        c, s = np.cos(rad), np.sin(rad)
        dx, dy = x - ox, y - oy
        x = ox + c * dx - s * dy
        y = oy + s * dx + c * dy
    return x, y


def build_hex_grid(
    z: np.ndarray,
    transform,
    proj: str,
    size_m: float,
    nodata: float | None = None,
    method: str = "bilinear",
    order: int = 1,
    dq: float = 0.0,
    dr: float = 0.0,
    angle_deg: float = 0.0,
) -> HexGrid:
    """Dense pointy-top hex lattice covering the raster; bilinear sample at centers.

    Earlier versions binned each pixel center into the nearest hex. With hex spacing
    equal to the pixel size that map is 1-to-1, leaves ~half the offset array empty,
    and makes hex-vs-square volume differ by exactly √3/2. Sampling every hex center
    that falls on the DEM gives a hole-free tessellation of comparable coverage.
    """
    h, w = z.shape
    size_m = float(size_m)
    valid = np.isfinite(z)
    if nodata is not None:
        valid &= z != nodata
    if not np.any(valid):
        raise RuntimeError("DEM has no valid pixels after nodata mask")

    q0, r0, nrows, ncols = hex_lattice_extent(z, transform, size_m)
    ox = transform.c + 0.5 * w * transform.a
    oy = transform.f + 0.5 * h * transform.e
    x, y = hex_center_xy(q0, r0, nrows, ncols, size_m, dq=dq, dr=dr, angle_deg=angle_deg, origin_xy=(ox, oy))

    if method == "area":
        return _hex_from_area_weights(
            z,
            valid,
            transform,
            proj,
            size_m,
            q0,
            r0,
            nrows,
            ncols,
            dq=dq,
            dr=dr,
            angle_deg=angle_deg,
            origin_xy=(ox, oy),
        )

    col_f = (x - transform.c) / transform.a - 0.5
    row_f = (y - transform.f) / transform.e - 0.5
    inside = (col_f >= 0.0) & (col_f <= (w - 1)) & (row_f >= 0.0) & (row_f <= (h - 1))
    c0i = np.floor(col_f).astype(int)
    r0i = np.floor(row_f).astype(int)
    src_ok = inside.copy()
    for drow, dcol in ((0, 0), (0, 1), (1, 0), (1, 1)):
        rr_i = np.clip(r0i + drow, 0, h - 1)
        cc_i = np.clip(c0i + dcol, 0, w - 1)
        src_ok &= valid[rr_i, cc_i]

    fill = float(np.nanmean(z))
    z_fill = np.where(valid, z, fill).astype(np.float64)
    sampled = ndimage.map_coordinates(
        z_fill,
        np.vstack([row_f.ravel(), col_f.ravel()]),
        order=int(order),
        mode="constant",
        cval=fill,
    ).reshape(nrows, ncols)
    values = np.full((nrows, ncols), np.nan, dtype=np.float64)
    values[src_ok] = sampled[src_ok]
    return HexGrid(q0, r0, size_m, nrows, ncols, proj, values, src_ok)


def _hex_from_area_weights(
    z,
    valid,
    transform,
    proj,
    size_m,
    q0,
    r0,
    nrows,
    ncols,
    dq=0.0,
    dr=0.0,
    angle_deg=0.0,
    origin_xy=None,
):
    """Assign each square pixel to the containing hex (area-weighted mean).

    Lattice shifts (dq, dr) and in-plane rotation are applied in axial space so
    that area weighting tracks the same geometry as bilinear center sampling.
    """
    h, w = z.shape
    pr, pc = np.mgrid[0:h, 0:w]
    px = transform.c + (pc + 0.5) * transform.a
    py = transform.f + (pr + 0.5) * transform.e
    if origin_xy is not None and abs(angle_deg) > 1e-12:
        ox, oy = origin_xy
        rad = np.deg2rad(angle_deg)
        c, s = np.cos(rad), np.sin(rad)
        dx, dy = px - ox, py - oy
        px = ox + c * dx + s * dy
        py = oy - s * dx + c * dy
    qf, rf = _xy_to_qr(px, py, size_m)
    qq, rr = axial_round(qf - float(dq), rf - float(dr))
    row, col = axial_to_offset(qq - q0, rr - r0)
    ok = valid & (row >= 0) & (row < nrows) & (col >= 0) & (col < ncols)
    acc = np.zeros((nrows, ncols), dtype=np.float64)
    cnt = np.zeros((nrows, ncols), dtype=np.float64)
    np.add.at(acc, (row[ok], col[ok]), z[ok])
    np.add.at(cnt, (row[ok], col[ok]), 1.0)
    values = np.full((nrows, ncols), np.nan, dtype=np.float64)
    filled = cnt > 0
    values[filled] = acc[filled] / cnt[filled]
    return HexGrid(q0, r0, size_m, nrows, ncols, proj, values, filled)


def hex_to_square_array(grid: HexGrid) -> np.ndarray:
    """Nearest-neighbour paint onto a square array with the same cell area."""
    cell = np.sqrt(grid.cell_area)
    rows, cols = np.mgrid[0 : grid.nrows, 0 : grid.ncols]
    q, r = offset_to_axial(rows, cols)
    q = q + grid.q0
    r = r + grid.r0
    x = grid.size * (np.sqrt(3.0) * (q + r / 2.0))
    y = grid.size * (1.5 * r)
    x0, x1 = float(x[grid.mask].min()), float(x[grid.mask].max())
    y0, y1 = float(y[grid.mask].min()), float(y[grid.mask].max())
    nx = max(8, int(np.ceil((x1 - x0) / cell)) + 1)
    ny = max(8, int(np.ceil((y1 - y0) / cell)) + 1)
    sq = np.full((ny, nx), np.nan, dtype=np.float64)
    ix = np.clip(np.floor((x - x0) / cell).astype(int), 0, nx - 1)
    iy = np.clip(np.floor((y1 - y) / cell).astype(int), 0, ny - 1)
    sq[iy[grid.mask], ix[grid.mask]] = grid.values[grid.mask]
    return sq
