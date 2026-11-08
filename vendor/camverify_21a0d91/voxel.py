"""Volumetric model for multi-setup (indexed) machining.

Everything lives on one voxel grid in the ORIGINAL part frame. A setup is a machining side ("+Z", "-X", ...): its frame
is the part rotated so that side points up, with the bounding-box minimum at the origin (exactly what the FreeCAD
scripts do, see fc_scripts/_common.load_oriented). Because material removal is order independent,

    workpiece = stock  AND  for every setup k: { z_k <= h_k(x_k, y_k) }

where h_k is the height-map left by setup k's G-code (or, for a setup that is not built yet, the height-map of the
target seen from that side = the ideal result of that setup).
"""
from __future__ import annotations

import math

import numpy as np

# identical to fc_scripts/_common.ROT
ROT = {"+Z": None, "-Z": ((1, 0, 0), 180), "+X": ((0, 1, 0), -90), "-X": ((0, 1, 0), 90), "+Y": ((1, 0, 0), 90), "-Y": ((1, 0, 0), -90)}
OPPOSITE = {"+Z": "-Z", "-Z": "+Z", "+X": "-X", "-X": "+X", "+Y": "-Y", "-Y": "+Y"}


def rotation(up: str) -> np.ndarray:
    r = ROT[up]
    if r is None:
        return np.eye(3)
    (ax, ay, az), deg = r
    c, s = round(math.cos(math.radians(deg))), round(math.sin(math.radians(deg)))
    if ax:
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)
    if ay:
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


class Frame:
    """Original frame -> setup frame: p_k = R p + t."""

    def __init__(self, up: str, translation) -> None:
        self.up, self.R, self.t = up, rotation(up), np.asarray(translation, float)

    def to_setup(self, P: np.ndarray) -> np.ndarray:
        return P @ self.R.T + self.t

    def to_original(self, Pk: np.ndarray) -> np.ndarray:
        return (Pk - self.t) @ self.R


class VoxelGrid:
    def __init__(self, bbox, margin: float = 3.0, max_voxels: float = 9e6, min_res: float = 0.3) -> None:
        lo, hi = np.array(bbox[:3], float) - margin, np.array(bbox[3:], float) + margin
        lo = lo - np.array([0.0137, 0.0211, 0.0173])  # keep voxel centres off model planes (ray-parity degeneracy)
        size = hi - lo
        self.res = max(min_res, float((size.prod() / max_voxels) ** (1 / 3)))
        self.lo = lo
        self.n = np.maximum(1, np.ceil(size / self.res).astype(int))  # nx, ny, nz
        self.shape = (self.n[2], self.n[1], self.n[0])  # arrays are indexed [z, y, x]
        self._centres = None

    @property
    def cell_volume(self) -> float:
        return self.res ** 3

    def axes(self):
        return [self.lo[a] + (np.arange(self.n[a]) + 0.5) * self.res for a in range(3)]

    def centres(self) -> np.ndarray:
        """(N, 3) voxel centres in C order of an array indexed [z, y, x]."""
        if self._centres is None:
            x, y, z = self.axes()
            Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
            self._centres = np.c_[X.ravel(), Y.ravel(), Z.ravel()].astype(np.float32)
        return self._centres

    # -------------------------------------------------------------------------------------------- target
    def voxelize(self, verts, faces) -> np.ndarray:
        """Solid voxelisation by ray parity along z (mesh must be closed)."""
        V, F = np.asarray(verts, float), np.asarray(faces, int)
        x, y, z = self.axes()
        nx, ny, nz = self.n
        cols, zs = [], []
        for t in V[F]:
            (xa, ya, za), (xb, yb, zb), (xc, yc, zc) = t
            det = (yb - yc) * (xa - xc) + (xc - xb) * (ya - yc)
            if abs(det) < 1e-12:
                continue
            j0, j1 = np.searchsorted(x, t[:, 0].min()), np.searchsorted(x, t[:, 0].max())
            i0, i1 = np.searchsorted(y, t[:, 1].min()), np.searchsorted(y, t[:, 1].max())
            if j1 <= j0 or i1 <= i0:
                continue
            X, Y = np.meshgrid(x[j0:j1], y[i0:i1])
            l1 = ((yb - yc) * (X - xc) + (xc - xb) * (Y - yc)) / det
            l2 = ((yc - ya) * (X - xc) + (xa - xc) * (Y - yc)) / det
            l3 = 1 - l1 - l2
            # half-open edge rule so that a ray through a shared edge is counted once
            m = (l1 >= 0) & (l2 >= 0) & (l3 > 0) if det > 0 else (l1 > 0) & (l2 > 0) & (l3 >= 0)
            if not m.any():
                continue
            ii, jj = np.nonzero(m)
            cols.append((ii + i0) * nx + (jj + j0))
            zs.append((l1 * za + l2 * zb + l3 * zc)[m])
        solid = np.zeros(self.shape, bool)
        if not cols:
            return solid
        cols, zs = np.concatenate(cols), np.concatenate(zs)
        order = np.lexsort((zs, cols))
        cols, zs = cols[order], zs[order]
        # pair up consecutive hits within each column
        start = np.r_[True, cols[1:] != cols[:-1]]
        rank = np.arange(len(cols)) - np.maximum.accumulate(np.where(start, np.arange(len(cols)), 0))
        count = np.bincount(cols, minlength=nx * ny)[cols]
        ok = (rank % 2 == 0) & (rank + 1 < count)
        idx = np.nonzero(ok)[0]
        flat = solid.reshape(nz, ny * nx)
        for a, b, c in zip(zs[idx], zs[idx + 1], cols[idx]):
            k0, k1 = np.searchsorted(z, a), np.searchsorted(z, b)
            flat[k0:k1, c] = True
        return solid

    # -------------------------------------------------------------------------------------------- setups
    def lattice(self, frame: Frame):
        """Column lattice of this grid seen from a setup: returns (x0, y0, ncols_x, ncols_y, zmin_k, zmax_k)."""
        corners = np.array([[self.lo[0] + a * self.n[0] * self.res, self.lo[1] + b * self.n[1] * self.res, self.lo[2] + c * self.n[2] * self.res]
                            for a in (0, 1) for b in (0, 1) for c in (0, 1)])
        ck = frame.to_setup(corners)
        lo, hi = ck.min(0), ck.max(0)
        return lo, hi, int(round((hi[0] - lo[0]) / self.res)), int(round((hi[1] - lo[1]) / self.res))

    def columns(self, frame: Frame):
        """For every voxel: (flat column index in the setup lattice, z in the setup frame)."""
        lo, hi, ncx, ncy = self.lattice(frame)
        Pk = frame.to_setup(self.centres())
        j = np.clip(((Pk[:, 0] - lo[0]) / self.res).astype(np.int32), 0, ncx - 1)
        i = np.clip(((Pk[:, 1] - lo[1]) / self.res).astype(np.int32), 0, ncy - 1)
        return (i * ncx + j).astype(np.int32), Pk[:, 2].astype(np.float32), (ncy, ncx), lo, hi

    def under(self, frame: Frame, h: np.ndarray) -> np.ndarray:
        """Solid { z_k <= h_k } for a height-map given on this setup's lattice (shape (ncy, ncx))."""
        col, zk, shape, _, _ = self.columns(frame)
        assert h.shape == shape, (h.shape, shape)
        return (zk <= h.ravel()[col] + 1e-6).reshape(self.shape)

    def top_view(self, solid: np.ndarray, frame: Frame, empty: float) -> np.ndarray:
        """Height-map (max z_k + half a cell) of a voxel solid seen from a setup."""
        col, zk, shape, _, _ = self.columns(frame)
        h = np.full(shape[0] * shape[1], -np.inf)
        m = solid.ravel()
        np.maximum.at(h, col[m], zk[m] + self.res / 2)
        h[~np.isfinite(h)] = empty
        return h.reshape(shape)

    def column_thickness(self, solid: np.ndarray, frame: Frame) -> np.ndarray:
        """Thickness (mm of material, not a height difference) of a voxel solid per column of a setup's lattice.

        A height-map difference assumes the column between the two surfaces is solid, which is false for hollow parts
        (an enclosure seen from the side): use this when the VOLUME of a defect matters, not how deep it starts.
        """
        col, _, shape, _, _ = self.columns(frame)
        n = np.zeros(shape[0] * shape[1])
        np.add.at(n, col[solid.ravel()], 1.0)
        return (n * self.res).reshape(shape)

    def visible(self, target: np.ndarray, frame: Frame) -> np.ndarray:
        """Voxels that a tool coming from this side can reach without cutting the target (nothing of the target above)."""
        col, zk, shape, lo, _ = self.columns(frame)
        top = self.top_view(target, frame, empty=lo[2] - 1.0).ravel()
        return (zk >= top[col] - 1e-6).reshape(self.shape)


def dilate(a: np.ndarray, it: int = 1) -> np.ndarray:
    out = a.copy()
    for _ in range(it):
        p = np.pad(out, 1)
        out = p[1:-1, 1:-1, 1:-1] | p[:-2, 1:-1, 1:-1] | p[2:, 1:-1, 1:-1] | p[1:-1, :-2, 1:-1] | p[1:-1, 2:, 1:-1] | p[1:-1, 1:-1, :-2] | p[1:-1, 1:-1, 2:]
    return out
