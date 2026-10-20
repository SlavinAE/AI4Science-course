"""Material-removal simulation on a dexel (height-map) grid with ANALYTIC swept volumes of the tool.

Every cut move (a straight segment after arc linearisation) removes the volume swept by the tool along it - the CAD
"sweep and subtract" evaluated on the grid columns, not a sequence of sampled tool stamps:
  * horizontal move, any tool of revolution:   cut(x, y) = z + h(dist((x, y), segment))
  * vertical move (plunge, drill cycle):       cut(x, y) = z_low + h(dist((x, y), axis))
  * inclined move, flat endmill (ramps):       exact: the lowest point of the move over the disk around (x, y)
  * inclined move, profiled tool:              split into pieces with dz <= cell / 4 (error <= cell / 8)
where h(r) is the height of the cutting surface above the tool tip at radius r (flat, ball, bull nose, drill point,
spot drill / chamfer mill / countersink cones). The only discretisation left is the XY cell size, which the harness
sets to half the verification tolerance (bounded by a cell budget), so walls are within a quarter of the tolerance.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

from .raster import Grid


class ToolShape:
    def __init__(self, kind: str, radius: float, half_angle_deg: float | None = None, corner_r: float = 0.0, tip_r: float = 0.0) -> None:
        self.kind, self.R = kind, float(radius)
        self.cr, self.tip_r = float(corner_r), float(tip_r)
        self.cot = 1.0 / math.tan(math.radians(half_angle_deg)) if half_angle_deg else 0.0

    def profile(self, d):
        if self.kind == "flat":
            return np.zeros_like(d)
        if self.kind == "ball":
            return self.R - np.sqrt(np.maximum(self.R * self.R - d * d, 0.0))
        if self.kind == "bull":
            flat = self.R - self.cr
            e = np.maximum(d - flat, 0.0)
            return self.cr - np.sqrt(np.maximum(self.cr * self.cr - e * e, 0.0))
        return np.maximum(d - self.tip_r, 0.0) * self.cot  # cone (drill point, spot drill, chamfer mill, countersink)


def shape_for(tool: dict, process: str = "") -> ToolShape | None:
    """Cutting shape of a library tool; None = removes nothing that is part of the B-Rep target (taps, thread mills)."""
    cat = tool.get("category", "endmill")
    sp = tool.get("shape_params") or {}
    d = float(tool.get("diameter_mm") or sp.get("Diameter") or 6.0)
    if cat in ("tap", "thread_mill") or process in ("tapping", "thread_milling"):
        return None
    if cat == "twist_drill":
        return ToolShape("cone", d / 2, float(sp.get("TipAngle") or 118.0) / 2)
    if cat == "spot_drill":
        return ToolShape("cone", d / 2, float(sp.get("TipAngle") or 90.0) / 2)
    if cat in ("chamfer_mill", "countersink"):
        return ToolShape("cone", d / 2, float(sp.get("CuttingEdgeAngle") or 90.0) / 2, tip_r=float(sp.get("TipDiameter") or 0.0) / 2)
    if cat == "ball_endmill":
        return ToolShape("ball", d / 2)
    if cat == "bull_endmill":
        return ToolShape("bull", d / 2, corner_r=float(tool.get("corner_radius_mm") or d / 2 - float(sp.get("FlatRadius") or d / 2)))
    return ToolShape("flat", d / 2)  # endmills, face mills, reamers, boring bars


class DexelMap:
    """Top / bottom of the material in every column of a setup grid (setup frame, tool axis +Z)."""

    def __init__(self, grid: Grid, top: np.ndarray, bottom: np.ndarray) -> None:
        self.g = grid
        self.h = np.asarray(top, np.float64).copy()  # float64: the numba kernel and the exact comparisons work on it
        self.bottom = np.asarray(bottom, np.float32).copy()
        self.owner = np.full(grid.shape, -1, np.int16)
        self._xs = grid.x0 + np.arange(grid.nx) * grid.res
        self._ys = grid.y0 + np.arange(grid.ny) * grid.res

    # -------------------------------------------------------------------------------------------- cutting
    def cut(self, s: ToolShape, x0, y0, z0, x1, y1, z1, k: int = 0) -> None:
        g, R = self.g, s.R
        j0 = max(int(math.floor((min(x0, x1) - R - g.x0) / g.res)), 0)
        j1 = min(int(math.ceil((max(x0, x1) + R - g.x0) / g.res)), g.nx - 1)
        i0 = max(int(math.floor((min(y0, y1) - R - g.y0) / g.res)), 0)
        i1 = min(int(math.ceil((max(y0, y1) + R - g.y0) / g.res)), g.ny - 1)
        if j1 < j0 or i1 < i0:
            return
        X = self._xs[None, j0:j1 + 1]
        Y = self._ys[i0:i1 + 1, None]
        dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
        L2 = dx * dx + dy * dy
        if L2 < 1e-12:  # vertical move: deepest point counts
            d = np.hypot(X - x0, Y - y0)
            m = d <= R
            cut = min(z0, z1) + s.profile(np.minimum(d, R))
        elif abs(dz) < 1e-9:
            t = np.clip(((X - x0) * dx + (Y - y0) * dy) / L2, 0.0, 1.0)
            d = np.hypot(X - (x0 + t * dx), Y - (y0 + t * dy))
            m = d <= R
            cut = z0 + s.profile(np.minimum(d, R))
        elif s.kind == "flat":
            L = math.sqrt(L2)
            t = ((X - x0) * dx + (Y - y0) * dy) / L2
            dp = np.abs((X - x0) * dy - (Y - y0) * dx) / L
            w = np.sqrt(np.maximum(R * R - dp * dp, 0.0)) / L
            ta, tb = np.clip(t - w, 0.0, 1.0), np.clip(t + w, 0.0, 1.0)
            m = (dp <= R) & (t + w >= 0.0) & (t - w <= 1.0)
            cut = np.minimum(z0 + ta * dz, z0 + tb * dz)
        else:
            n = min(400, max(1, int(math.ceil(abs(dz) / (g.res / 4)))))
            if n == 1:  # dz below a quarter cell: horizontal at the mean height (error <= cell / 8)
                zm = (z0 + z1) / 2
                self.cut(s, x0, y0, zm, x1, y1, zm, k)
                return
            for q in range(n):
                a, b = q / n, (q + 1) / n
                zm = z0 + (a + b) / 2 * dz
                self.cut(s, x0 + a * dx, y0 + a * dy, zm, x0 + b * dx, y0 + b * dy, zm, k)
            return
        sub = self.h[i0:i1 + 1, j0:j1 + 1]
        lower = m & (cut < sub - 1e-7)
        if lower.any():
            sub[lower] = cut[lower]
            self.owner[i0:i1 + 1, j0:j1 + 1][lower] = k

    def run_moves(self, moves, s: ToolShape | None, k: int, start: int = 0, stop: int | None = None) -> None:
        if s is None:
            return
        segs = [mv[:6] for mv in moves[start:stop] if mv[6]]
        if not segs:
            return
        if len(segs) >= GPU_MIN_SEGMENTS and _gpu_ready():  # CUDA (dexel_gpu.py): identical result, ~100x faster on big programs
            try:
                from .dexel_gpu import sweep_inplace
                sweep_inplace(self, np.asarray(segs, np.float64), s, k)
                return
            except Exception as e:  # noqa: BLE001  (never break the simulation: fall back to the CPU for this process)
                _GPU[0] = False
                print(f"[dexel] GPU sweep disabled ({type(e).__name__}: {str(e)[:200]}): CPU kernel", file=sys.stderr)
        if _jit is not None:  # same algorithm, compiled (optional dependency numba)
            S = np.asarray(segs, np.float64)
            kind = {"flat": 0, "ball": 1, "bull": 2}.get(s.kind, 3)
            _jit(self.h, self.owner, S, self.g.x0, self.g.y0, self.g.res, kind, s.R, s.cr, s.tip_r, s.cot, k)
            return
        for mv in segs:
            self.cut(s, *mv, k)

    # -------------------------------------------------------------------------------------------- queries
    def material(self) -> np.ndarray:
        return self.h > self.bottom + 1e-9

    def volume(self) -> float:
        return float(np.clip(self.h - self.bottom, 0, None).sum() * self.g.res ** 2)

    def inside(self, P: np.ndarray) -> np.ndarray:
        """Points (setup frame) inside the remaining material: the column holds material from bottom to top."""
        i, j, ok = self.g.index(P[:, 0], P[:, 1])
        return ok & (P[:, 2] <= self.h[i, j]) & (P[:, 2] >= self.bottom[i, j])

    def view(self, max_side: int = 900) -> "DexelView":
        """Down-sampled copy for rendering (the simulation itself keeps the fine grid)."""
        step = max(1, int(math.ceil(max(self.g.nx, self.g.ny) / max_side)))
        return DexelView(self, step)


class DexelView:
    """Light-weight render view of a DexelMap (attributes used by the video tools)."""

    def __init__(self, dm: DexelMap, step: int) -> None:
        g = dm.g
        self.step, self.res = step, g.res * step
        self.h = dm.h[::step, ::step]
        self.bottom = dm.bottom[::step, ::step]
        self.owner = dm.owner[::step, ::step]
        self.x0, self.y0 = g.x0, g.y0
        self.nx, self.ny = self.h.shape[1], self.h.shape[0]
        self.x1, self.y1 = g.x0 + (self.nx - 1) * self.res, g.y0 + (self.ny - 1) * self.res
        self.X, self.Y = np.meshgrid(g.x0 + np.arange(self.nx) * self.res, g.y0 + np.arange(self.ny) * self.res)
        finite = self.bottom[self.bottom > -1e8]
        self.zb = float(finite.min()) if finite.size else 0.0
        self.zt = float(self.h.max())

    def volume(self) -> float:
        return float(np.clip(self.h - self.bottom, 0, None).sum() * self.res ** 2)


def stock_dexel(grid: Grid, stock_mesh: dict, target_mesh: dict | None = None, return_target: bool = False):
    """Initial material = the stock (box or blank) in this setup frame; the target is never outside its own stock
    (a coarsely tessellated blank can dip under it by a fraction of a millimetre)."""
    from .raster import rasterize
    top = rasterize(grid, stock_mesh["vertices"], stock_mesh["faces"], "top", empty=-np.inf)
    bot = rasterize(grid, stock_mesh["vertices"], stock_mesh["faces"], "bottom", empty=np.inf)
    tt = None
    if target_mesh is not None:
        tt = rasterize(grid, target_mesh["vertices"], target_mesh["faces"], "top", empty=-np.inf)
        tb = rasterize(grid, target_mesh["vertices"], target_mesh["faces"], "bottom", empty=np.inf)
        top, bot = np.maximum(top, tt), np.minimum(bot, tb)
    none = ~np.isfinite(top) | ~np.isfinite(bot)
    floor = float(np.min(bot[~none])) - 1.0 if (~none).any() else 0.0
    top[none], bot[none] = floor, floor
    dm = DexelMap(grid, top, bot)
    return (dm, tt) if return_target else dm


# ------------------------------------------------------------------------------------------------ optional numba kernel
def _make_jit():
    try:
        import numba
    except ImportError:
        return None

    @numba.njit(cache=True)
    def prof(kind, R, cr, tip_r, cot, d):
        if kind == 0:
            return 0.0
        if kind == 1:
            return R - math.sqrt(max(R * R - d * d, 0.0))
        if kind == 2:
            e = max(d - (R - cr), 0.0)
            return cr - math.sqrt(max(cr * cr - e * e, 0.0))
        return max(d - tip_r, 0.0) * cot

    @numba.njit(cache=True)
    def cut1(h, owner, x0, y0, z0, x1, y1, z1, gx0, gy0, res, kind, R, cr, tip_r, cot, k):
        ny, nx = h.shape
        j0 = max(int(math.floor((min(x0, x1) - R - gx0) / res)), 0)
        j1 = min(int(math.ceil((max(x0, x1) + R - gx0) / res)), nx - 1)
        i0 = max(int(math.floor((min(y0, y1) - R - gy0) / res)), 0)
        i1 = min(int(math.ceil((max(y0, y1) + R - gy0) / res)), ny - 1)
        dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
        L2 = dx * dx + dy * dy
        L = math.sqrt(L2)
        for i in range(i0, i1 + 1):
            Y = gy0 + i * res
            for j in range(j0, j1 + 1):
                X = gx0 + j * res
                if L2 < 1e-12:
                    d = math.hypot(X - x0, Y - y0)
                    if d > R:
                        continue
                    c = min(z0, z1) + prof(kind, R, cr, tip_r, cot, d)
                elif abs(dz) < 1e-9 or kind != 0:
                    t = min(max(((X - x0) * dx + (Y - y0) * dy) / L2, 0.0), 1.0)
                    d = math.hypot(X - (x0 + t * dx), Y - (y0 + t * dy))
                    if d > R:
                        continue
                    c = z0 + prof(kind, R, cr, tip_r, cot, d)
                else:
                    t = ((X - x0) * dx + (Y - y0) * dy) / L2
                    dp = abs((X - x0) * dy - (Y - y0) * dx) / L
                    if dp > R:
                        continue
                    w = math.sqrt(max(R * R - dp * dp, 0.0)) / L
                    if t + w < 0.0 or t - w > 1.0:
                        continue
                    ta, tb = min(max(t - w, 0.0), 1.0), min(max(t + w, 0.0), 1.0)
                    c = min(z0 + ta * dz, z0 + tb * dz)
                if c < h[i, j] - 1e-7:
                    h[i, j] = c
                    owner[i, j] = k

    @numba.njit(cache=True)
    def run(h, owner, S, gx0, gy0, res, kind, R, cr, tip_r, cot, k):
        for q in range(S.shape[0]):
            x0, y0, z0, x1, y1, z1 = S[q, 0], S[q, 1], S[q, 2], S[q, 3], S[q, 4], S[q, 5]
            dz = z1 - z0
            if kind == 0 or abs(dz) < 1e-9 or (x0 == x1 and y0 == y1):
                cut1(h, owner, x0, y0, z0, x1, y1, z1, gx0, gy0, res, kind, R, cr, tip_r, cot, k)
                continue
            n = min(400, max(1, int(math.ceil(abs(dz) / (res / 4)))))
            for p in range(n):
                a, b = p / n, (p + 1) / n
                zm = z0 + (a + b) / 2 * dz
                cut1(h, owner, x0 + a * (x1 - x0), y0 + a * (y1 - y0), zm, x0 + b * (x1 - x0), y0 + b * (y1 - y0), zm, gx0, gy0, res, kind, R, cr, tip_r, cot, k)

    return run


_jit = _make_jit()

# ------------------------------------------------------------------------------------------------ optional CUDA path
# CAM_GPU: auto (default: use a visible CUDA device when CuPy is installed) | 0 (off). CAM_GPU_MIN_SEGMENTS: below this
# many cut moves in one call the upload / download costs more than the CPU kernel.
GPU_MIN_SEGMENTS = int(os.environ.get("CAM_GPU_MIN_SEGMENTS", "2000"))
_GPU = [None]


def _gpu_ready() -> bool:
    if _GPU[0] is None:
        ok = os.environ.get("CAM_GPU", "auto") != "0"
        if ok:
            try:
                import cupy as cp
                ok = cp.cuda.runtime.getDeviceCount() > 0
            except Exception:  # noqa: BLE001
                ok = False
        _GPU[0] = ok
    return bool(_GPU[0])
