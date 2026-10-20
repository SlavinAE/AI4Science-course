"""Rasterisation of triangle meshes onto a regular XY grid (height-maps / dexels along +Z).

Cell (i, j) is centred at (x0 + j * res, y0 + i * res). `top` keeps the highest surface crossing a cell column,
`bottom` the lowest one; columns that no triangle crosses are filled with `empty`.
"""
from __future__ import annotations

import math

import numpy as np


class Grid:
    def __init__(self, x0: float, y0: float, x1: float, y1: float, res: float) -> None:
        self.res = float(res)
        self.x0, self.y0 = float(x0), float(y0)
        self.nx = int(math.floor((x1 - x0) / res)) + 1
        self.ny = int(math.floor((y1 - y0) / res)) + 1
        self.x1, self.y1 = self.x0 + (self.nx - 1) * res, self.y0 + (self.ny - 1) * res
        self._X = self._Y = None

    @property
    def shape(self) -> tuple[int, int]:
        return (self.ny, self.nx)

    @property
    def X(self) -> np.ndarray:
        if self._X is None:
            self._X, self._Y = np.meshgrid(self.x0 + np.arange(self.nx) * self.res, self.y0 + np.arange(self.ny) * self.res)
        return self._X

    @property
    def Y(self) -> np.ndarray:
        if self._Y is None:
            _ = self.X
        return self._Y

    def index(self, x, y):
        """Nearest cell (i, j) of points (clipped to the grid) and a mask of points inside the grid."""
        j = np.rint((np.asarray(x) - self.x0) / self.res).astype(np.int64)
        i = np.rint((np.asarray(y) - self.y0) / self.res).astype(np.int64)
        ok = (i >= 0) & (i < self.ny) & (j >= 0) & (j < self.nx)
        return np.clip(i, 0, self.ny - 1), np.clip(j, 0, self.nx - 1), ok


def rasterize(grid: Grid, verts, faces, mode: str = "top", empty: float = -np.inf) -> np.ndarray:
    """Highest (mode 'top') or lowest ('bottom') z of the mesh over every cell centre."""
    V = np.asarray(verts, float)
    F = np.asarray(faces, np.int64)
    out = np.full(grid.shape, -np.inf if mode == "top" else np.inf)
    if not len(F):
        out[:] = empty
        return out
    tri = V[F]
    xs, ys = tri[:, :, 0], tri[:, :, 1]
    j0 = np.maximum(np.ceil((xs.min(1) - grid.x0) / grid.res - 1e-9).astype(np.int64), 0)
    j1 = np.minimum(np.floor((xs.max(1) - grid.x0) / grid.res + 1e-9).astype(np.int64), grid.nx - 1)
    i0 = np.maximum(np.ceil((ys.min(1) - grid.y0) / grid.res - 1e-9).astype(np.int64), 0)
    i1 = np.minimum(np.floor((ys.max(1) - grid.y0) / grid.res + 1e-9).astype(np.int64), grid.ny - 1)
    (xa, ya, za), (xb, yb, zb), (xc, yc, zc) = tri[:, 0].T, tri[:, 1].T, tri[:, 2].T
    det = (yb - yc) * (xa - xc) + (xc - xb) * (ya - yc)
    keep = (np.abs(det) > 1e-12) & (j1 >= j0) & (i1 >= i0)
    ufunc = np.maximum if mode == "top" else np.minimum
    flat = out.reshape(-1)
    big = keep & ((j1 - j0 + 1) * (i1 - i0 + 1) > 20000)
    for k in np.flatnonzero(big):  # large triangles (flat faces of a box): scalar coefficients, broadcast over the bbox
        X = (grid.x0 + np.arange(j0[k], j1[k] + 1) * grid.res)[None, :]
        Y = (grid.y0 + np.arange(i0[k], i1[k] + 1) * grid.res)[:, None]
        l1 = ((yb[k] - yc[k]) * (X - xc[k]) + (xc[k] - xb[k]) * (Y - yc[k])) / det[k]
        l2 = ((yc[k] - ya[k]) * (X - xc[k]) + (xa[k] - xc[k]) * (Y - yc[k])) / det[k]
        l3 = 1.0 - l1 - l2
        m = (l1 >= -1e-9) & (l2 >= -1e-9) & (l3 >= -1e-9)
        z = l1 * za[k] + l2 * zb[k] + l3 * zc[k]
        sub = out[i0[k]:i1[k] + 1, j0[k]:j1[k] + 1]
        sub[m] = ufunc(sub[m], z[m])
    idx = np.flatnonzero(keep & ~big)
    nj, ni = (j1 - j0 + 1)[idx], (i1 - i0 + 1)[idx]
    counts = nj * ni
    # (triangle, cell) pairs in chunks of about 4M: barycentric test and z for all of them at once
    bounds = np.searchsorted(np.cumsum(counts), np.arange(4_000_000, int(counts.sum()) + 4_000_000, 4_000_000))
    lo = 0
    for hi in (list(bounds) + [len(idx)]) if len(idx) else []:
        hi = int(min(max(hi + 1, lo + 1), len(idx)))
        if lo >= len(idx):
            break
        sel, c = idx[lo:hi], counts[lo:hi]
        t = np.repeat(sel, c)
        local = np.arange(int(c.sum())) - np.repeat(np.cumsum(c) - c, c)
        w = np.repeat(nj[lo:hi], c)
        jj = j0[t] + local % w
        ii = i0[t] + local // w
        X = grid.x0 + jj * grid.res
        Y = grid.y0 + ii * grid.res
        l1 = ((yb[t] - yc[t]) * (X - xc[t]) + (xc[t] - xb[t]) * (Y - yc[t])) / det[t]
        l2 = ((yc[t] - ya[t]) * (X - xc[t]) + (xa[t] - xc[t]) * (Y - yc[t])) / det[t]
        l3 = 1.0 - l1 - l2
        m = (l1 >= -1e-9) & (l2 >= -1e-9) & (l3 >= -1e-9)
        z = l1 * za[t] + l2 * zb[t] + l3 * zc[t]
        ufunc.at(flat, (ii * grid.nx + jj)[m], z[m])
        lo = hi
    out[~np.isfinite(out)] = empty
    return out
