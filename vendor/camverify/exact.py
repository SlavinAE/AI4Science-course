"""Exact swept-volume point query: the material a 3-axis program leaves, without a grid (ported from the sim line's
sim/src/exact.py; QUEUE P35).

A 3-axis tool (axis +Z, shank going up without end) removes in a move the set {P : exists t, d(t) <= R and
P.z >= z(t) + h(d(t))}, where c(t) is the tip on the segment, d(t) the XY distance from P to c(t), h(r) the height of
the cutting profile above the tip (flat 0, ball, bull nose, cone with a tip flat). A setup removes the union over its
moves, so P keeps material iff no move of the setup removes it. The surface metric (`verify.surface_deviation`) only
asks inside(P) at the probe points along the target normals, so only those points are evaluated:

  * moves are bucketed by their XY box (+R) on a uniform 2D grid; in a bucket they are sorted by the lowest tip
    height zmin, the scan stops at the first zmin above P.z (f(t) >= zmin) and at the first move that removes P;
  * per move: vertical - lowest tip + h(|w|); horizontal - h at the closest XY point of the segment; inclined flat -
    the lowest tip over the t-interval where d(t) <= R; inclined profiled tool - f(t) = z(t) + h(d(t)) is convex on
    that interval (d(t) convex, h convex non-decreasing), its minimum by golden section (error ~1e-9 of the interval).

Inputs as `verify.simulate_setup`: `gcode.parse` (arcs linearised at 0.01 mm), operations matched to tools by
`ops_meta_for`, profiles of `dexel.shape_for` (taps and thread mills cut nothing). Unlike the dexel, rapids cut as well
by default (a G0 into material is a crash, not air). A box blank is taken exactly (BoxWorkpiece: no grid phase on the
walls left on the blank's faces); other blanks - the stock dexel through `verify.Workpiece`.

Used by `MultiSetupVerifier.final()` and the single-setup CAM agent with CAM_METRIC_EXACT=1 (the final metric only:
the repair loop, its regions, rest and the shield stay on the dexel), and by `tools/exact_metric.py` on saved sessions
(also the whole part after machining as a closed mesh: mesh_nets / mesh_tets / mesh_of). Needs numba.
"""
from __future__ import annotations

import math
import time

import numpy as np

import numba as nb

from . import gcode
from .dexel import shape_for, stock_dexel
from .stock import transform
from .verify import Workpiece, ops_meta_for, sim_grid, surface_deviation
from .voxel import Frame

KIND = {"flat": 0, "ball": 1, "bull": 2, "cone": 3}
GOLD = 0.5 * (math.sqrt(5.0) - 1.0)


@nb.njit(cache=True)
def _h(kind, R, cr, tip, cot, d):
    if kind == 0:
        return 0.0
    if kind == 1:
        return R - math.sqrt(max(R * R - d * d, 0.0))
    if kind == 2:
        e = max(d - (R - cr), 0.0)
        return cr - math.sqrt(max(cr * cr - e * e, 0.0))
    return max(d - tip, 0.0) * cot


@nb.njit(cache=True)
def _removes(M, T, m, px, py, pz):
    """Move m (row of M: x0 y0 z0 x1 y1 z1 tool) removes P?  T rows: kind R cr tip cot."""
    x0, y0, z0, x1, y1, z1 = M[m, 0], M[m, 1], M[m, 2], M[m, 3], M[m, 4], M[m, 5]
    ti = int(M[m, 6])
    kind, R, cr, tip, cot = int(T[ti, 0]), T[ti, 1], T[ti, 2], T[ti, 3], T[ti, 4]
    wx, wy = px - x0, py - y0
    ux, uy, dz = x1 - x0, y1 - y0, z1 - z0
    a = ux * ux + uy * uy
    if a < 1e-18:  # vertical
        d = math.sqrt(wx * wx + wy * wy)
        return d <= R and pz >= min(z0, z1) + _h(kind, R, cr, tip, cot, d)
    # t-interval where d(t) <= R: a t^2 - 2 (w.u) t + |w|^2 - R^2 <= 0
    b = wx * ux + wy * uy
    c = wx * wx + wy * wy - R * R
    disc = b * b - a * c
    if disc < 0.0:
        return False
    s = math.sqrt(disc)
    t0, t1 = max((b - s) / a, 0.0), min((b + s) / a, 1.0)
    if t0 > t1:
        return False
    if pz < min(z0 + t0 * dz, z0 + t1 * dz):  # f >= z(t) >= its minimum on [t0, t1]
        return False
    if kind == 0:  # flat: f = z(t), linear
        return True
    tc = min(max(b / a, t0), t1)  # closest XY point on the interval
    if abs(dz) < 1e-12:  # horizontal: h(d) is smallest where d is
        ex, ey = wx - tc * ux, wy - tc * uy
        return pz >= z0 + _h(kind, R, cr, tip, cot, math.sqrt(ex * ex + ey * ey))
    # inclined profiled tool: golden section on the convex f(t) = z(t) + h(d(t)); early exit when f(t) <= pz
    lo, hi = t0, t1
    for t in (tc, t0, t1):
        ex, ey = wx - t * ux, wy - t * uy
        if z0 + t * dz + _h(kind, R, cr, tip, cot, math.sqrt(ex * ex + ey * ey)) <= pz:
            return True
    ta, tb = hi - GOLD * (hi - lo), lo + GOLD * (hi - lo)
    ex, ey = wx - ta * ux, wy - ta * uy
    fa = z0 + ta * dz + _h(kind, R, cr, tip, cot, math.sqrt(ex * ex + ey * ey))
    ex, ey = wx - tb * ux, wy - tb * uy
    fb = z0 + tb * dz + _h(kind, R, cr, tip, cot, math.sqrt(ex * ex + ey * ey))
    for _ in range(60):
        if fa <= pz or fb <= pz:
            return True
        if fa < fb:
            hi, tb, fb = tb, ta, fa
            ta = hi - GOLD * (hi - lo)
            ex, ey = wx - ta * ux, wy - ta * uy
            fa = z0 + ta * dz + _h(kind, R, cr, tip, cot, math.sqrt(ex * ex + ey * ey))
        else:
            lo, ta, fa = ta, tb, fb
            tb = lo + GOLD * (hi - lo)
            ex, ey = wx - tb * ux, wy - tb * uy
            fb = z0 + tb * dz + _h(kind, R, cr, tip, cot, math.sqrt(ex * ex + ey * ey))
        if hi - lo < 1e-10:
            break
    return min(fa, fb) <= pz


@nb.njit(cache=True)
def _build(M, T, x0, y0, cs, nx, ny):
    """CSR buckets: start[c]..start[c+1] in idx — moves whose XY box + R touches cell c, sorted by zmin."""
    n = M.shape[0]
    cnt = np.zeros(nx * ny + 1, np.int64)
    for m in range(n):
        R = T[int(M[m, 6]), 1]
        i0 = max(int((min(M[m, 0], M[m, 3]) - R - x0) / cs), 0)
        i1 = min(int((max(M[m, 0], M[m, 3]) + R - x0) / cs), nx - 1)
        j0 = max(int((min(M[m, 1], M[m, 4]) - R - y0) / cs), 0)
        j1 = min(int((max(M[m, 1], M[m, 4]) + R - y0) / cs), ny - 1)
        for j in range(j0, j1 + 1):
            for i in range(i0, i1 + 1):
                cnt[j * nx + i + 1] += 1
    start = np.cumsum(cnt)
    idx = np.empty(start[-1], np.int64)
    fill = start[:-1].copy()
    for m in range(n):
        R = T[int(M[m, 6]), 1]
        i0 = max(int((min(M[m, 0], M[m, 3]) - R - x0) / cs), 0)
        i1 = min(int((max(M[m, 0], M[m, 3]) + R - x0) / cs), nx - 1)
        j0 = max(int((min(M[m, 1], M[m, 4]) - R - y0) / cs), 0)
        j1 = min(int((max(M[m, 1], M[m, 4]) + R - y0) / cs), ny - 1)
        for j in range(j0, j1 + 1):
            for i in range(i0, i1 + 1):
                c = j * nx + i
                idx[fill[c]] = m
                fill[c] += 1
    zmin = np.minimum(M[:, 2], M[:, 5])
    for c in range(nx * ny):
        a, b = start[c], start[c + 1]
        if b - a > 1:
            seg = idx[a:b]
            idx[a:b] = seg[np.argsort(zmin[seg], kind="mergesort")]
    return start, idx, zmin


@nb.njit(parallel=True, cache=True)
def _inside(P, M, T, start, idx, zmin, x0, y0, cs, nx, ny):
    out = np.ones(P.shape[0], np.bool_)
    for k in nb.prange(P.shape[0]):
        px, py, pz = P[k, 0], P[k, 1], P[k, 2]
        i, j = int(math.floor((px - x0) / cs)), int(math.floor((py - y0) / cs))
        if i < 0 or j < 0 or i >= nx or j >= ny:
            continue
        c = j * nx + i
        for q in range(start[c], start[c + 1]):
            m = idx[q]
            if zmin[m] > pz:
                break
            if _removes(M, T, m, px, py, pz):
                out[k] = False
                break
    return out


@nb.njit(parallel=True, cache=True)
def _block_state(C, M, T, start, idx, zmin, x0, y0, cs, nx, ny):
    """Per block (its 8 corners C[b] in the setup frame): 1 — no move reaches the block's box (the setup removes
    nothing there), -1 — one move removes all 8 corners (a move's sweep is convex — the tool is the epigraph of a convex
    profile over a disk, swept along a segment — so it removes the whole block), 0 — neither is proven."""
    out = np.zeros(C.shape[0], np.int8)
    for b in nb.prange(C.shape[0]):
        xl, yl, zl = C[b, :, 0].min(), C[b, :, 1].min(), C[b, :, 2].min()
        xh, yh, zh = C[b, :, 0].max(), C[b, :, 1].max(), C[b, :, 2].max()
        i0, i1 = max(int(math.floor((xl - x0) / cs)), 0), min(int(math.floor((xh - x0) / cs)), nx - 1)
        j0, j1 = max(int(math.floor((yl - y0) / cs)), 0), min(int(math.floor((yh - y0) / cs)), ny - 1)
        touched = False
        for j in range(j0, j1 + 1):
            for i in range(i0, i1 + 1):
                c = j * nx + i
                for q in range(start[c], start[c + 1]):
                    m = idx[q]
                    if zmin[m] > zh:  # removal needs P.z >= the move's lowest tip
                        break
                    R = T[int(M[m, 6]), 1]
                    if (min(M[m, 0], M[m, 3]) - R <= xh and max(M[m, 0], M[m, 3]) + R >= xl
                            and min(M[m, 1], M[m, 4]) - R <= yh and max(M[m, 1], M[m, 4]) + R >= yl):
                        touched = True
                        break
                if touched:
                    break
            if touched:
                break
        if not touched:
            out[b] = 1
            continue
        k = np.argmin(C[b, :, 2])  # a move removing all corners removes the lowest one: scan its bucket
        i, j = int(math.floor((C[b, k, 0] - x0) / cs)), int(math.floor((C[b, k, 1] - y0) / cs))
        if i < 0 or j < 0 or i >= nx or j >= ny:
            continue
        c = j * nx + i
        for q in range(start[c], start[c + 1]):
            m = idx[q]
            if zmin[m] > zl:
                break
            ok = True
            for v in range(8):
                if not _removes(M, T, m, C[b, v, 0], C[b, v, 1], C[b, v, 2]):
                    ok = False
                    break
            if ok:
                out[b] = -1
                break
    return out


# Kuhn split of a cell into 6 tetrahedra: paths 0 -> e_a -> e_a + e_b -> (1,1,1); corner c = x | y << 1 | z << 2.
# Translation-invariant, so neighbouring cells split their common face the same way (no cracks).
KUHN = np.array([[0, 1, 3, 7], [0, 1, 5, 7], [0, 2, 3, 7], [0, 2, 6, 7], [0, 4, 5, 7], [0, 4, 6, 7]], np.int64)
CORNERS = np.array([[(c >> d) & 1 for d in range(3)] for c in range(8)], np.float64)


@nb.njit(cache=True)
def _ekey(a, b, gx, gy, gz, n1, n2):
    """Global key of the cell edge between corners a, b (nested: a Kuhn edge goes along +x/+y/+z only): the node at its
    low end * 8 + the direction bits."""
    lo = a & b
    return ((((gx + (lo & 1)) * n1 + gy + ((lo >> 1) & 1)) * n2 + gz + ((lo >> 2) & 1)) * 8) + ((a | b) ^ lo)


@nb.njit(cache=True)
def _put(out, w, fill, a0, b0, a1, b1, a2, b2, dx, dy, dz, gx, gy, gz, n1, n2):
    """Write the triangle on edges (a0 b0), (a1 b1), (a2 b2), turned so that its normal (on the edge midpoints, never
    degenerate) points along d = out of the material."""
    if fill:
        m0x, m0y, m0z = (a0 & 1) + (b0 & 1), ((a0 >> 1) & 1) + ((b0 >> 1) & 1), ((a0 >> 2) & 1) + ((b0 >> 2) & 1)
        m1x, m1y, m1z = (a1 & 1) + (b1 & 1), ((a1 >> 1) & 1) + ((b1 >> 1) & 1), ((a1 >> 2) & 1) + ((b1 >> 2) & 1)
        m2x, m2y, m2z = (a2 & 1) + (b2 & 1), ((a2 >> 1) & 1) + ((b2 >> 1) & 1), ((a2 >> 2) & 1) + ((b2 >> 2) & 1)
        ux, uy, uz = m1x - m0x, m1y - m0y, m1z - m0z
        vx, vy, vz = m2x - m0x, m2y - m0y, m2z - m0z
        s = (uy * vz - uz * vy) * dx + (uz * vx - ux * vz) * dy + (ux * vy - uy * vx) * dz
        k0, k1, k2 = _ekey(a0, b0, gx, gy, gz, n1, n2), _ekey(a1, b1, gx, gy, gz, n1, n2), _ekey(a2, b2, gx, gy, gz, n1, n2)
        out[w, 0] = k0
        if s > 0:
            out[w, 1], out[w, 2] = k1, k2
        else:
            out[w, 1], out[w, 2] = k2, k1
    return w + 1


@nb.njit(parallel=True, cache=True)
def _tets(S, org, n1, n2, K, off, out, fill):
    """Marching tetrahedra over leaf blocks: S[b] — node signs of block b ((L+1)^3, True — material), org[b] — its
    first node. fill=False counts triangles per block, fill=True writes their edge keys from off[b]."""
    nb_ = S.shape[0]
    L = S.shape[1] - 1
    cnt = np.zeros(nb_, np.int64)
    for bi in nb.prange(nb_):
        w = off[bi] if fill else 0
        ins = np.empty(4, np.int64)
        outs = np.empty(4, np.int64)
        for ci in range(L):
            for cj in range(L):
                for ck in range(L):
                    msk = 0
                    for c in range(8):
                        if S[bi, ci + (c & 1), cj + ((c >> 1) & 1), ck + ((c >> 2) & 1)]:
                            msk |= 1 << c
                    if msk == 0 or msk == 255:
                        continue
                    gx, gy, gz = org[bi, 0] + ci, org[bi, 1] + cj, org[bi, 2] + ck
                    for t in range(6):
                        ni = no = 0
                        for q in range(4):
                            v = K[t, q]
                            if (msk >> v) & 1:
                                ins[ni] = v
                                ni += 1
                            else:
                                outs[no] = v
                                no += 1
                        if ni == 0 or no == 0:
                            continue
                        dx = dy = dz = 0.0  # out-of-material direction: centroid of the air corners - of the material ones
                        for q in range(no):
                            dx += (outs[q] & 1) / no
                            dy += ((outs[q] >> 1) & 1) / no
                            dz += ((outs[q] >> 2) & 1) / no
                        for q in range(ni):
                            dx -= (ins[q] & 1) / ni
                            dy -= ((ins[q] >> 1) & 1) / ni
                            dz -= ((ins[q] >> 2) & 1) / ni
                        if ni == 1:
                            a = ins[0]
                            w = _put(out, w, fill, a, outs[0], a, outs[1], a, outs[2], dx, dy, dz, gx, gy, gz, n1, n2)
                        elif no == 1:
                            o = outs[0]
                            w = _put(out, w, fill, ins[0], o, ins[1], o, ins[2], o, dx, dy, dz, gx, gy, gz, n1, n2)
                        else:  # quad on edges ac, ad, bd, bc (a, b material; c, d air)
                            a, b, c, d = ins[0], ins[1], outs[0], outs[1]
                            w = _put(out, w, fill, a, c, a, d, b, d, dx, dy, dz, gx, gy, gz, n1, n2)
                            w = _put(out, w, fill, a, c, b, d, b, c, dx, dy, dz, gx, gy, gz, n1, n2)
        cnt[bi] = w - (off[bi] if fill else 0)
    return cnt


class SweepSetup:
    """Material left by one setup's program: inside(P) for points in the original frame."""

    def __init__(self, moves: np.ndarray, tools: np.ndarray, frame: Frame, cell: float = 2.0, max_cells: float = 4e6) -> None:
        self.frame, self.M, self.T = frame, moves, tools
        self.queries = 0
        if not len(moves):
            self.nx = self.ny = 0
            return
        R = tools[moves[:, 6].astype(int), 1]
        lo = np.minimum(moves[:, [0, 1]], moves[:, [3, 4]]).min(0) - R.max() - 1.0
        hi = np.maximum(moves[:, [0, 1]], moves[:, [3, 4]]).max(0) + R.max() + 1.0
        cs = max(cell, math.sqrt((hi[0] - lo[0]) * (hi[1] - lo[1]) / max_cells))
        self.x0, self.y0, self.cs = float(lo[0]), float(lo[1]), float(cs)
        self.nx, self.ny = int((hi[0] - lo[0]) / cs) + 1, int((hi[1] - lo[1]) / cs) + 1
        self.start, self.idx, self.zmin = _build(moves, tools, self.x0, self.y0, self.cs, self.nx, self.ny)

    def inside(self, P: np.ndarray) -> np.ndarray:
        self.queries += len(P)
        if not self.nx:
            return np.ones(len(P), bool)
        Pk = np.ascontiguousarray(self.frame.to_setup(P), np.float64)
        return _inside(Pk, self.M, self.T, self.start, self.idx, self.zmin, self.x0, self.y0, self.cs, self.nx, self.ny)


def box_of(mesh: dict, eps: float = 1e-6):
    """(lo, hi) when the mesh is an axis-aligned box (8 vertices on its bbox corners), else None."""
    V = np.asarray(mesh["vertices"], float)
    lo, hi = V.min(0), V.max(0)
    if len(V) != 8 or not (np.isclose(V, lo, atol=eps) | np.isclose(V, hi, atol=eps)).all():
        return None
    return lo, hi


class BoxWorkpiece:
    """A box blank exactly (points in the original frame) AND every setup's sweep: no stock dexel, so the stock adds
    no grid phase to walls the program leaves on the blank's faces (H54). Unlike the stock dexel the target is not
    merged into the blank (a box blank of the part holds it to rounding: eps)."""

    def __init__(self, lo, hi, setups: list[SweepSetup], eps: float = 1e-6) -> None:
        self.lo, self.hi, self.setups, self.eps = np.asarray(lo, float) - eps, np.asarray(hi, float) + eps, setups, eps

    def inside(self, P: np.ndarray) -> np.ndarray:
        m = ((P >= self.lo) & (P <= self.hi)).all(1)
        for s in self.setups:
            k = np.flatnonzero(m)
            if not len(k):
                break
            m[k] = s.inside(P[k])
        return m

    def classify(self, C: np.ndarray) -> np.ndarray:
        """Axis boxes by their 8 corners C (n, 8, 3; original frame, computed as the grid nodes are — a node on a
        sweep's boundary must get the same sign here and in inside()): 1 — all material, -1 — all air, 0 — not proven
        (for mesh_tets). Air: off the blank or inside one move's sweep of some setup; material: inside the blank, no
        setup reaches it. The sweep tests take the box grown by `eps`: a node on a sweep's boundary gets either sign in
        floating point (d <= R by rounding), convexity holds only away from it."""
        st = np.zeros(len(C), np.int8)
        blo, bhi = C.min(1), C.max(1)
        inb = ((blo >= self.lo) & (bhi <= self.hi)).all(1)
        off = ((bhi < self.lo) | (blo > self.hi)).any(1)
        C = blo[:, None, :] + (bhi - blo)[:, None, :] * CORNERS[None] + self.eps * (2 * CORNERS[None] - 1)
        mat = inb.copy()
        air = off.copy()
        for s in self.setups:
            if not s.nx:
                continue
            Ck = np.ascontiguousarray(s.frame.to_setup(C.reshape(-1, 3)).reshape(-1, 8, 3), np.float64)
            r = _block_state(Ck, s.M, s.T, s.start, s.idx, s.zmin, s.x0, s.y0, s.cs, s.nx, s.ny)
            mat &= r == 1
            air |= r == -1
        st[mat] = 1
        st[air] = -1
        return st


def mesh_of(inside, lo, hi, step: float, bisect: int = 10) -> tuple[np.ndarray, np.ndarray, dict]:
    """The whole material as a closed mesh (surface nets): inside() at the nodes of a grid of step `step` over [lo, hi]
    padded by a node of air, the crossing on every edge with a sign change by bisection (step / 2^bisect), a vertex per
    cell with a crossing — the mean of its crossings, a quad per crossed edge. Error ~step at sharp edges and walls
    thinner than a cell (two sheets in a cell share a vertex: not a manifold there), not in the sign of a node."""
    lo, hi = np.asarray(lo, float) - step, np.asarray(hi, float) + step
    n = (np.ceil((hi - lo) / step).astype(int) + 1)
    ax = [lo[d] + step * np.arange(n[d]) for d in range(3)]
    S = np.zeros(n, bool)
    X, Y = np.meshgrid(ax[0], ax[1], indexing="ij")
    for k in range(1, n[2] - 1):  # the outer layer stays air: the surface closes
        P = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, ax[2][k])])
        S[:, :, k] = inside(P).reshape(X.shape)
    S[0], S[-1], S[:, 0], S[:, -1] = False, False, False, False
    nodes = int(S.size)
    faces, cells, pts = [], [], []  # sparse: only cells next to a crossed edge
    for d in range(3):
        a = [slice(None)] * 3
        b = [slice(None)] * 3
        a[d], b[d] = slice(0, -1), slice(1, None)
        E = np.argwhere(S[tuple(a)] != S[tuple(b)])  # edge from node E to E + e_d
        if not len(E):
            continue
        A = np.column_stack([ax[q][E[:, q]] for q in range(3)])
        ina = S[E[:, 0], E[:, 1], E[:, 2]]
        t0, t1 = np.zeros(len(E)), np.ones(len(E))
        for _ in range(bisect):
            tm = 0.5 * (t0 + t1)
            P = A.copy()
            P[:, d] += tm * step
            same = inside(P) == ina
            t0, t1 = np.where(same, tm, t0), np.where(same, t1, tm)
        C = A.copy()
        C[:, d] += 0.5 * (t0 + t1) * step
        u, v = [q for q in range(3) if q != d]  # (u, v, d) is a right-handed cycle
        if (d, u, v) not in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            u, v = v, u
        ring = []
        for du, dv in ((-1, -1), (0, -1), (0, 0), (-1, 0)):  # the 4 cells around the edge, counter-clockwise about +d
            c = E.copy()
            c[:, u] += du
            c[:, v] += dv
            ring.append(np.ravel_multi_index(c.T, tuple(n - 1)))
        q = np.column_stack(ring)
        q[~ina] = q[~ina][:, ::-1]  # normal out of the material: along +d when the material is at the edge's start
        faces.append(q)
        cells.append(q.ravel())
        pts.append(np.repeat(C, 4, axis=0))
    if not faces:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64), {"step_mm": step, "nodes": nodes, "grid": n.tolist(),
                                                              "bisect": bisect, "crossed_edges": 0}
    Q = np.concatenate(faces)
    used, inv = np.unique(np.concatenate(cells), return_inverse=True)
    P = np.concatenate(pts)
    cnt = np.bincount(inv, minlength=len(used))
    V = np.column_stack([np.bincount(inv, P[:, q], len(used)) for q in range(3)]) / cnt[:, None]
    Q = np.searchsorted(used, Q)
    F = np.concatenate([Q[:, [0, 1, 2]], Q[:, [0, 2, 3]]])
    return V, F, {"step_mm": step, "nodes": nodes, "grid": n.tolist(), "bisect": bisect, "crossed_edges": int(len(Q))}


def _leaves(wp, lo, hi, step: float, block: int, leaf: int):
    """Grid of step `step` over [lo, hi] padded by a node of air, rounded up to whole top blocks of `block` cells.
    Blocks that wp.classify proves all material or all air (if wp has it) are dropped, the rest split by 2 down to
    `leaf` cells. Returns (lo, node counts n, leaf origins (cells), leaf size, per-level counts)."""
    lo, hi = np.asarray(lo, float) - step, np.asarray(hi, float) + step
    ncell = np.ceil((hi - lo) / step).astype(np.int64)
    ncell = -(-ncell // block) * block  # whole top blocks: the extra cells are off the blank (air)
    n = ncell + 1
    cand = np.stack(np.meshgrid(*[np.arange(0, ncell[d], block) for d in range(3)], indexing="ij"), -1).reshape(-1, 3)
    size, levels = block, []
    classify = getattr(wp, "classify", None)
    off8 = CORNERS.astype(np.int64)
    while True:
        st = (classify(lo + step * (cand[:, None, :] + size * off8[None])) if classify
              else np.zeros(len(cand), np.int8))
        levels.append({"size": int(size), "blocks": int(len(cand)), "material": int((st == 1).sum()), "air": int((st == -1).sum())})
        cand = cand[st == 0]
        if size <= leaf:
            break
        size //= 2
        cand = (cand[:, None, :] + size * off8[None]).reshape(-1, 3)
    return lo, n, cand, size, levels


def _crossings(wp, A: np.ndarray, D: np.ndarray, bisect: int) -> np.ndarray:
    """Crossing of the material boundary on edges A -> A + D (a sign change known): bisection, step / 2^bisect."""
    ina = wp.inside(A)
    t_lo, t_hi = np.zeros(len(A)), np.ones(len(A))
    for _ in range(bisect):
        tm = 0.5 * (t_lo + t_hi)
        same = wp.inside(A + tm[:, None] * D) == ina
        t_lo, t_hi = np.where(same, tm, t_lo), np.where(same, t_hi, tm)
    return A + (0.5 * (t_lo + t_hi))[:, None] * D, ina


def mesh_tets(wp, lo, hi, step: float, bisect: int = 10, block: int = 16, leaf: int = 4,
              chunk: int = 4000) -> tuple[np.ndarray, np.ndarray, dict]:
    """The whole material as a closed 2-manifold mesh (marching tetrahedra on the Kuhn split of a grid of step `step`
    over [lo, hi] padded by a node of air): the level set of a piecewise-linear sign field on a triangulation is a
    closed manifold whatever the signs, each triangle lies in its tetrahedron (no self-intersections), its vertex is
    the crossing of its edge by bisection (step / 2^bisect). Error ~step at sharp edges, not in the sign of a node.
    Sparse (_leaves): only the nodes of leaf blocks not proven uniform are evaluated — work ~ surface / step^2, not
    volume / step^3. ~4x the vertices of surface nets (crossings on 7 edge directions, not 3)."""
    t0 = time.time()
    lo, n, cand, size, levels = _leaves(wp, lo, hi, step, block, leaf)
    t_cls = time.time() - t0
    L = size + 1
    loc = np.stack(np.meshgrid(*[np.arange(L)] * 3, indexing="ij"), -1).reshape(-1, 3)
    keys = []
    for c0 in range(0, len(cand), chunk):
        org = cand[c0:c0 + chunk]
        P = lo + step * (org[:, None, :] + loc[None]).reshape(-1, 3)
        S = wp.inside(P).reshape(len(org), L, L, L)
        dummy = np.zeros((0, 3), np.int64)
        cnt = _tets(S, org, n[1], n[2], KUHN, np.zeros(len(org), np.int64), dummy, False)
        off = np.concatenate([[0], np.cumsum(cnt)[:-1]]).astype(np.int64)
        out = np.empty((int(cnt.sum()), 3), np.int64)
        _tets(S, org, n[1], n[2], KUHN, off, out, True)
        keys.append(out)
    t_sign = time.time() - t0 - t_cls
    info = {"method": "tets", "step_mm": step, "grid": n.tolist(), "bisect": bisect, "levels": levels,
            "leaf_blocks": int(len(cand)), "nodes_evaluated": int(len(cand) * L ** 3), "nodes_dense": int(n.prod())}
    K = np.concatenate(keys) if keys else np.zeros((0, 3), np.int64)
    if not len(K):
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64), info
    uk, inv = np.unique(K.ravel(), return_inverse=True)
    node, dirs = uk // 8, uk % 8
    ijk = np.column_stack(np.unravel_index(node, tuple(n)))
    V, _ = _crossings(wp, lo + step * ijk, step * CORNERS[dirs], bisect)
    F = inv.reshape(-1, 3)
    info.update({"crossed_edges": int(len(uk)), "seconds_classify": round(t_cls, 1), "seconds_signs": round(t_sign, 1),
                 "seconds_vertices": round(time.time() - t0 - t_cls - t_sign, 1)})
    return V, F, info


def _sheet_tables():
    """Manifold surface nets tables. Cell edges e = 0..11: (low corner, axis), slot of the cell among the 4 around the
    edge (ring (-1,-1), (0,-1), (0,0), (-1,0) in the right-handed (u, v) of the axis). SHEET[mask, e] — the sheet of a
    crossed edge in a cell of corner signs `mask` (-1: not crossed): crossed edges are joined along the cell's faces —
    the two crossings of a face, or on a face with 4 (material on a diagonal) each material corner's two edges, so
    material corners stay apart. The rule sees only the face's signs: both cells of a face pair its crossings alike."""
    uv = {0: (1, 2), 1: (2, 0), 2: (0, 1)}
    ring = [(-1, -1), (0, -1), (0, 0), (-1, 0)]
    edges = []  # (c0, c1, axis, slot)
    for a in range(3):
        u, v = uv[a]
        for bu in (0, 1):
            for bv in (0, 1):
                c0 = (bu << u) | (bv << v)
                edges.append((c0, c0 | (1 << a), a, ring.index((-bu, -bv))))
    E = np.array(edges, np.int64)
    SHEET = -np.ones((256, 12), np.int64)
    for m in range(256):
        sg = [(m >> c) & 1 for c in range(8)]
        cr = [sg[c0] != sg[c1] for c0, c1, _, _ in edges]
        par = list(range(12))

        def root(x):
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x
        for f in range(3):
            for s in (0, 1):
                fe = [e for e, (c0, c1, a, _) in enumerate(edges) if a != f and (c0 >> f) & 1 == s]
                ce = [e for e in fe if cr[e]]
                if len(ce) == 2:
                    par[root(ce[0])] = root(ce[1])
                elif len(ce) == 4:
                    for c in range(8):
                        if (c >> f) & 1 == s and sg[c]:
                            pe = [e for e in fe if c in edges[e][:2]]
                            par[root(pe[0])] = root(pe[1])
        lab = {}
        for e in range(12):
            if cr[e]:
                SHEET[m, e] = lab.setdefault(root(e), len(lab))
    return E, SHEET


EDGES12, SHEET = _sheet_tables()


@nb.njit(parallel=True, cache=True)
def _nets(S, org, n1, n2, E, SH, off, out, fill):
    """Manifold surface nets over leaf blocks: per crossed cell edge a record (edge key = low node * 3 + axis, slot of
    the cell around the edge, vertex key = cell's low node * 4 + sheet). fill=False counts per block."""
    nb_ = S.shape[0]
    L = S.shape[1] - 1
    cnt = np.zeros(nb_, np.int64)
    for bi in nb.prange(nb_):
        w = off[bi] if fill else 0
        for ci in range(L):
            for cj in range(L):
                for ck in range(L):
                    msk = 0
                    for c in range(8):
                        if S[bi, ci + (c & 1), cj + ((c >> 1) & 1), ck + ((c >> 2) & 1)]:
                            msk |= 1 << c
                    if msk == 0 or msk == 255:
                        continue
                    gx, gy, gz = org[bi, 0] + ci, org[bi, 1] + cj, org[bi, 2] + ck
                    cell = (gx * n1 + gy) * n2 + gz
                    for e in range(12):
                        sh = SH[msk, e]
                        if sh < 0:
                            continue
                        if fill:
                            c0 = E[e, 0]
                            node = ((gx + (c0 & 1)) * n1 + gy + ((c0 >> 1) & 1)) * n2 + gz + ((c0 >> 2) & 1)
                            out[w, 0] = node * 3 + E[e, 2]
                            out[w, 1] = E[e, 3]
                            out[w, 2] = cell * 4 + sh
                        w += 1
        cnt[bi] = w - (off[bi] if fill else 0)
    return cnt


def mesh_nets(wp, lo, hi, step: float, bisect: int = 10, block: int = 16, leaf: int = 4,
              chunk: int = 4000) -> tuple[np.ndarray, np.ndarray, dict]:
    """The whole material as a closed mesh, manifold surface nets: as surface nets (a quad per crossed grid edge over
    the vertices of its 4 cells, the vertex — the mean of its crossings), but a vertex per SHEET of a cell (SHEET), not
    per cell, so two sheets in one cell (a wall thinner than a cell) do not share a vertex. Sparse as mesh_tets
    (_leaves). Error ~step at sharp edges (the mean vertex cuts them), not in the sign of a node."""
    t0 = time.time()
    lo, n, cand, size, levels = _leaves(wp, lo, hi, step, block, leaf)
    t_cls = time.time() - t0
    L = size + 1
    loc = np.stack(np.meshgrid(*[np.arange(L)] * 3, indexing="ij"), -1).reshape(-1, 3)
    recs = []
    for c0 in range(0, len(cand), chunk):
        org = cand[c0:c0 + chunk]
        S = wp.inside(lo + step * (org[:, None, :] + loc[None]).reshape(-1, 3)).reshape(len(org), L, L, L)
        cnt = _nets(S, org, n[1], n[2], EDGES12, SHEET, np.zeros(len(org), np.int64), np.zeros((0, 3), np.int64), False)
        off = np.concatenate([[0], np.cumsum(cnt)[:-1]]).astype(np.int64)
        out = np.empty((int(cnt.sum()), 3), np.int64)
        _nets(S, org, n[1], n[2], EDGES12, SHEET, off, out, True)
        recs.append(out)
    t_sign = time.time() - t0 - t_cls
    info = {"method": "nets", "step_mm": step, "grid": n.tolist(), "bisect": bisect, "levels": levels,
            "leaf_blocks": int(len(cand)), "nodes_evaluated": int(len(cand) * L ** 3), "nodes_dense": int(n.prod())}
    R = np.concatenate(recs) if recs else np.zeros((0, 3), np.int64)
    if not len(R):
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64), info
    R = R[np.lexsort((R[:, 1], R[:, 0]))]
    if len(R) % 4 or (R[:, 1].reshape(-1, 4) != np.arange(4)).any() or (R[::4, 0] != R[3::4, 0]).any():
        raise RuntimeError("mesh_nets: a crossed edge without its 4 cells (a leaf block misses a neighbour)")
    ek = R[::4, 0]
    node, ax = ek // 3, ek % 3
    C, ina = _crossings(wp, lo + step * np.column_stack(np.unravel_index(node, tuple(n))), step * np.eye(3)[ax], bisect)
    uv_, vinv = np.unique(R[:, 2], return_inverse=True)
    cntv = np.bincount(vinv, minlength=len(uv_))
    Cr = np.repeat(C, 4, axis=0)
    V = np.column_stack([np.bincount(vinv, Cr[:, q], len(uv_)) for q in range(3)]) / cntv[:, None]
    Q = vinv.reshape(-1, 4)
    Q[~ina] = Q[~ina][:, ::-1]  # normal out of the material: along +axis when the material is at the edge's start
    F = np.concatenate([Q[:, [0, 1, 2]], Q[:, [0, 2, 3]]])
    info.update({"crossed_edges": int(len(ek)), "sheets": int(len(uv_)),
                 "cells_multi_sheet": int(len(np.unique(uv_[uv_ % 4 > 0] // 4))), "seconds_classify": round(t_cls, 1),
                 "seconds_signs": round(t_sign, 1), "seconds_vertices": round(time.time() - t0 - t_cls - t_sign, 1)})
    return V, F, info


def mesh_check(V: np.ndarray, F: np.ndarray) -> dict:
    """Validity of a triangle mesh: every edge in exactly 2 faces with opposite directions (closed, oriented), each
    vertex's faces one fan (manifold vertex), no degenerate faces, Euler characteristic, manifold3d's own check."""
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    nv = int(V.shape[0])
    und = np.sort(E, 1)
    ue, ec = np.unique(und[:, 0] * nv + und[:, 1], return_counts=True)
    de, dc = np.unique(E[:, 0] * nv + E[:, 1], return_counts=True)
    A = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
    # manifold vertex: the link edges (b -> c of each face corner at v) close into exactly one cycle around v
    cv = F.ravel()
    lk = np.column_stack([F[:, [1, 2, 0]].ravel(), F[:, [2, 0, 1]].ravel()])
    o = np.argsort(cv, kind="stable")
    fdeg = np.bincount(cv, minlength=nv)
    bad = _link_cycles(np.concatenate([[0], np.cumsum(fdeg)]), lk[o, 0].copy(), lk[o, 1].copy())
    out = {"vertices": nv, "faces": int(len(F)), "edges": int(len(ue)), "edges_not_2": int((ec != 2).sum()),
           "directed_edges_repeated": int((dc != 1).sum()), "vertices_unused": int((fdeg == 0).sum()),
           "vertices_nonmanifold": int(bad), "faces_zero_area": int((A <= 1e-12).sum()),
           "min_area_mm2": float(A.min()) if len(A) else None, "euler": int(nv - len(ue) + len(F))}
    try:
        import manifold3d as m3
        # float64 (Mesh64): the mesh is valid in double; the STL (float32) may round slivers flat — allowed
        man = m3.Manifold(m3.Mesh64(vert_properties=np.ascontiguousarray(V, np.float64), tri_verts=F.astype(np.uint64)))
        out["manifold3d"] = str(man.status()).split(".")[-1]
        out["manifold3d_genus"] = int(man.genus())
    except Exception as e:  # noqa: BLE001
        out["manifold3d"] = f"error: {e}"
    out["valid"] = bool(out["edges_not_2"] == 0 and out["directed_edges_repeated"] == 0 and out["vertices_nonmanifold"] == 0
                        and out["faces_zero_area"] == 0)
    return out


@nb.njit(cache=True)
def _link_cycles(start, src, dst):
    """Vertices whose link edges (src -> dst, grouped by vertex from start) are not one closed cycle."""
    bad = 0
    for v in range(len(start) - 1):
        a, b = start[v], start[v + 1]
        k = b - a
        if k == 0:
            continue
        seen = np.zeros(k, np.bool_)
        j, steps, ok = 0, 0, True
        while not seen[j]:
            seen[j] = True
            steps += 1
            nxt = -1
            for q in range(k):
                if src[a + q] == dst[a + j]:
                    nxt = q
                    break
            if nxt < 0:
                ok = False
                break
            j = nxt
        if not ok or steps != k or j != 0:
            bad += 1
    return bad


def setup_moves(gcode_file, operations, tools, rapids: bool, arc_tol: float, drop: set | None = None):
    """Moves of one setup as the agent reads them: rows x0 y0 z0 x1 y1 z1 tool_row, tool rows kind R cr tip cot.
    drop: operation labels not to cut (those a ready project comments out)."""
    parsed = gcode.parse(str(gcode_file), arc_tol=arc_tol)
    meta = ops_meta_for(parsed, operations, tools)
    rows, trows, tkey, skipped, dropped = [], [], {}, [], []
    for m, o in zip(meta, parsed["ops"]):
        if drop and o["name"] in drop:
            dropped.append(o["name"])
            continue
        s = shape_for(m["tool"], m["process"] or "") if m["tool"] else None
        if s is None:
            skipped.append(o["name"])
            continue
        key = (s.kind, s.R, s.cr, s.tip_r, s.cot)
        if key not in tkey:
            tkey[key] = len(trows)
            trows.append([KIND[s.kind], s.R, s.cr, s.tip_r, s.cot])
        ti = tkey[key]
        rows += [mv[:6] + (ti,) for mv in o["moves"] if mv[6] or rapids]
    M = np.asarray(rows, np.float64).reshape(-1, 7)
    if drop and set(dropped) != drop:
        raise ValueError(f"{gcode_file}: project labels not in the program: {sorted(drop - set(dropped))}")
    return M, np.asarray(trows, np.float64).reshape(-1, 5), {"moves": len(M), "ops": len(parsed["ops"]), "skipped_ops": skipped,
                                                               "dropped_ops": dropped, "warnings": parsed["warnings"]}


def workpiece(part_raw: dict, stock_raw: dict, frame0: Frame, programs: list, tol: float, max_cells: float, stock_dm=None,
              rapids: bool = True, arc_tol: float = 0.01, cell: float = 2.0, drops: list | None = None):
    """Material left by the programs [(setup frame, gcode_file, operations, tools)] in the frame of part_raw / stock_raw:
    a box blank exactly (BoxWorkpiece), another blank - its dexel (stock_dm, or built on the stock grid as
    MultiSetupVerifier's, frame0 = the primary setup's) AND every setup's sweep. Returns (workpiece, per-setup info)."""
    setups, info = [], []
    for i, (fr, g, ops, tools) in enumerate(programs):
        tb = time.time()
        M, T, inf = setup_moves(g, ops, tools, rapids, arc_tol, (drops or [None] * len(programs))[i])
        st = SweepSetup(M, T, fr, cell)
        setups.append(st)
        info.append({"up": fr.up, "gcode": str(g), **inf, "tools": len(T), "index_cells": st.nx * st.ny,
                     "index_entries": int(len(st.idx)) if st.nx else 0, "build_s": round(time.time() - tb, 1)})
    box = box_of(stock_raw)
    if box is not None:
        return BoxWorkpiece(*box, setups), info
    if stock_dm is None:
        stock_dm = stock_dexel(stock_grid(part_raw, stock_raw, frame0, tol, max_cells), transform(stock_raw, frame0), transform(part_raw, frame0))
    return Workpiece(stock_dm, frame0, setups), info


def stock_grid(part_raw: dict, stock_raw: dict, frame0: Frame, tol: float, max_cells: float):
    """The grid of MultiSetupVerifier's stock dexel (pad 1 mm in the primary setup's frame): its cell is the metric's step."""
    st0, pt0 = transform(stock_raw, frame0), transform(part_raw, frame0)
    bb = [min(st0["bbox"][0], pt0["bbox"][0]), min(st0["bbox"][1], pt0["bbox"][1]),
          max(st0["bbox"][3], pt0["bbox"][3]), max(st0["bbox"][4], pt0["bbox"][4])]
    return sim_grid(bb, 1.0, tol, max_cells)


def final_stats(part_raw: dict, stock_raw: dict, frame0: Frame, programs: list, tol: float, samples: int, max_cells: float,
                stock_dm=None, step: float | None = None, rapids: bool = True) -> dict:
    """The surface metric (`surface_deviation`) of the whole part on the exact query; step - the half-cell shifts of the
    metric (default: the stock grid's cell, as MultiSetupVerifier.final)."""
    t0 = time.time()
    if step is None:
        step = float(stock_dm.g.res) if stock_dm is not None else float(stock_grid(part_raw, stock_raw, frame0, tol, max_cells).res)
    wp, info = workpiece(part_raw, stock_raw, frame0, programs, tol, max_cells, stock_dm, rapids)
    sd = surface_deviation(part_raw, wp.inside, tol, samples, cell=step)
    for p, st in zip(info, wp.setups):
        p["queries"] = st.queries
    return {**sd["stats"], "metric": "exact", "rapids": "cut" if rapids else "skip",
            "stock_sim": "exact_box" if isinstance(wp, BoxWorkpiece) else "dexel", "step_mm": step,
            "seconds": round(time.time() - t0, 1), "threads": nb.get_num_threads(), "setups_exact": info}
