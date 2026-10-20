"""Verification of simulated machining against the target B-Rep - ONE metric for the report, the repair loop, the
videos and the external `evaluate` command.

Surface metric: the target surface (tessellated finer than a quarter of the tolerance) is sampled with area-weighted
random points; at every point the simulated workpiece is probed along the outward normal:
    material left  e = how far material continues outside the target surface   (p + s n inside the workpiece)
    overcut        g = how far the workpiece is missing inside the target        (p - s n outside the workpiece)
A point is in tolerance when neither exceeds `tol`. The share of surface AREA in tolerance is reported; air around the
part does not count. Volumes are surface integrals (sum of sample area x deviation).

The workpiece is a set of dexel maps (one per setup, setup frame); a point (original frame) is inside when it is inside
the stock and below the machined surface of every setup:  W = stock AND for all setups k { z_k <= h_k(x_k, y_k) }.
Setups that are not machined yet use their ideal result (target top view from that side), so a setup can be checked
before the others exist. Column deviation maps (setup view) are still used to segment defects into regions and to
attribute them to features and operations for the repair loop.
"""
from __future__ import annotations

import math

import numpy as np

from . import gcode
from .dexel import DexelMap, shape_for, stock_dexel
from .raster import Grid
from .voxel import Frame


# ----------------------------------------------------------------------------------------------- sampling
def sample_surface(verts, faces, n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Area-weighted random points on a closed mesh with outward unit normals. Returns (P, N, area_per_point)."""
    V, F = np.asarray(verts, float), np.asarray(faces, np.int64)
    T = V[F]
    cr = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    area = 0.5 * np.linalg.norm(cr, axis=1)
    keep = area > 1e-12
    T, cr, area = T[keep], cr[keep], area[keep]
    if (np.einsum("ij,ij->i", T[:, 0], cr).sum()) < 0:  # inward-oriented mesh: flip normals
        cr = -cr
    N_tri = cr / (2 * area[:, None])
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(T), size=n, p=area / area.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s1 = np.sqrt(r1)
    P = (1 - s1)[:, None] * T[idx, 0] + (s1 * (1 - r2))[:, None] * T[idx, 1] + (s1 * r2)[:, None] * T[idx, 2]
    return P, N_tri[idx], np.full(n, area.sum() / n)


# ----------------------------------------------------------------------------------------------- workpiece
class SetupState:
    def __init__(self, frame: Frame, dexel: DexelMap | None = None, ideal: DexelMap | None = None) -> None:
        self.frame, self.dexel, self.ideal = frame, dexel, ideal

    def inside(self, P: np.ndarray) -> np.ndarray:
        dm = self.dexel or self.ideal
        if dm is None:
            return np.ones(len(P), bool)
        Pk = self.frame.to_setup(P)
        i, j, ok = dm.g.index(Pk[:, 0], Pk[:, 1])
        return ~ok | (Pk[:, 2] <= dm.h[i, j] + 1e-9)  # outside a setup's grid that setup removed nothing


class Workpiece:
    """stock (dexel of the blank in the primary frame) AND every setup's machined (or ideal) height field."""

    def __init__(self, stock: DexelMap, stock_frame: Frame, setups: list[SetupState]) -> None:
        self.stock, self.stock_frame, self.setups = stock, stock_frame, setups

    def inside(self, P: np.ndarray) -> np.ndarray:
        Pk = self.stock_frame.to_setup(P)
        i, j, ok = self.stock.g.index(Pk[:, 0], Pk[:, 1])
        m = ok & (Pk[:, 2] <= self.stock.h[i, j] + 1e-9) & (Pk[:, 2] >= self.stock.bottom[i, j] - 1e-9)
        for s in self.setups:
            if m.any():
                m &= s.inside(P)
        return m


def surface_deviation(target: dict, inside, tol: float, n: int = 600000, features: list[dict] | None = None,
                      frame: Frame | None = None, max_probe: float = 20.0, seed: int = 0, cell: float = 0.0) -> dict:
    """Area-weighted surface statistics. `inside(P)` takes points in the target's frame. `cell`: XY size of the
    simulation cells; every probe is also evaluated half a cell to each side in X and Y and the least defective
    reading is kept (a sample on a sharp edge must not be judged by the neighbouring column)."""
    P, N, w = sample_surface(target["vertices"], target["faces"], n, seed)
    ladder = sorted({round(x, 4) for x in [tol / 4, tol / 2, 3 * tol / 4, tol, 1.5 * tol, 2 * tol, 3 * tol, 5 * tol, 0.5, 1, 2, 3, 5, 10, max_probe] if 0 < x <= max_probe})
    L = np.asarray(ladder)
    h = cell / 2
    shifts = [np.zeros(3)] + ([np.array(v, float) for v in ((h, 0, 0), (-h, 0, 0), (0, h, 0), (0, -h, 0))] if h > 0 else [])

    def run_length(mask):  # largest ladder value s_k such that mask holds for all s <= s_k (contiguous from the start)
        stop = np.argmin(np.vstack([mask, np.zeros((1, mask.shape[1]), bool)]), axis=0)
        return np.where(stop > 0, L[np.maximum(stop - 1, 0)], 0.0)

    excess = gouge = None
    for d in shifts:
        e = run_length(np.stack([inside(P + s * N + d) for s in ladder]))      # material outside the target surface
        g = run_length(~np.stack([inside(P - s * N + d) for s in ladder]))     # material missing inside the target
        excess = e if excess is None else np.minimum(excess, e)
        gouge = g if gouge is None else np.minimum(gouge, g)
    within = (excess < tol - 1e-9) & (gouge < tol - 1e-9)
    dev = np.where(excess > 0, excess, -gouge)
    expected = expected_mask(P, features or [])
    area = w.sum()
    stats = {"surface_within_tol_pct": round(100 * float(w[within].sum() / area), 2),
             "surface_within_tol_pct_excl_expected": round(100 * float(w[within | expected].sum() / area), 2),
             "tol_mm": tol, "samples": int(n), "surface_mm2": round(float(area), 1),
             "excess_cm3": round(float((w * excess).sum()) / 1000, 3), "gouge_cm3": round(float((w * gouge).sum()) / 1000, 3),
             "max_excess_mm": round(float(excess.max()), 3), "max_gouge_mm": round(float(gouge.max()), 3),
             "excess_area_pct": round(100 * float(w[excess > tol].sum() / area), 2), "gouge_area_pct": round(100 * float(w[gouge > tol].sum() / area), 2),
             "probe_cap_mm": max_probe}
    return {"stats": stats, "points": P, "normals": N, "dev": dev, "within": within, "expected": expected}


def expected_mask(P: np.ndarray, features: list[dict]) -> np.ndarray:
    """Samples on the bottom of blind drilled holes (the drill point cone is expected, not a defect). Features are in
    the setup frame of the points."""
    m = np.zeros(len(P), bool)
    for f in features:
        if f.get("type") != "hole" or f.get("through") or not f.get("top") or not f.get("vertical", True):
            continue
        r = float(f["diameter"]) / 2
        bottom = float(f.get("top_z", f["top"][2])) - float(f.get("depth") or 0)
        tip = r / math.tan(math.radians(59.0)) + 0.1
        m |= (np.hypot(P[:, 0] - f["top"][0], P[:, 1] - f["top"][1]) <= r + 0.05) & (P[:, 2] <= bottom + 0.05) & (P[:, 2] >= bottom - tip)
    return m


# ----------------------------------------------------------------------------------------------- regions (repair loop)
def _label(mask):
    from scipy import ndimage
    return ndimage.label(mask)


def _feature_at(features, x, y):
    best, best_d = None, 1e9
    for f in features:
        t = f.get("type")
        if t == "hole" and f.get("top"):
            d = math.hypot(x - f["top"][0], y - f["top"][1]) - f["diameter"] / 2
        elif t in ("pocket", "slot", "step", "chamfer", "cavity") and f.get("bbox"):
            x0, y0, x1, y1 = f["bbox"][:4]
            dd = math.hypot(max(x0 - x, 0, x - x1), max(y0 - y, 0, y - y1))
            d = dd if dd > 0 else -0.01
        else:
            continue
        if d < best_d:
            best, best_d = f, d
    return best if best is not None and best_d <= 1.5 else None


def find_regions(dev, grid: Grid, owner, target_h, features, ops_meta, tol: float, tooling=None, part_bbox=None,
                 thickness_fn=None, min_volume_mm3: float = 2.0, surface_gouge=None) -> list[dict]:
    """Segment a column deviation map (setup view) into excess / gouge regions, attribute each to a feature and to the
    operation that cut there last, and classify it: error | inherent | expected | limitation | artefact.
    surface_gouge: cells under surface samples that the surface metric finds gouged; a column gouge without one is the
    column model's artefact (a flank grazing a wall column's centre counts as a cut through the whole column)."""
    from scipy import ndimage
    cell = grid.res ** 2
    by_index = {o["index"]: o for o in ops_meta}
    closing = top = None
    if tooling:
        # what the smallest endmill can reach = morphological closing of the target by its disk; computed on a grid
        # coarse enough for a footprint of about 8 cells across (it only classifies regions)
        step = max(1, int(tooling["min_d"] / 8 / grid.res))
        th = target_h[::step, ::step]
        n = max(1, int(round(tooling["min_d"] / 2 / (grid.res * step))))
        jj, ii = np.meshgrid(np.arange(-n, n + 1), np.arange(-n, n + 1))
        fp = (ii ** 2 + jj ** 2) <= n * n
        c = ndimage.grey_erosion(ndimage.grey_dilation(th, footprint=fp, mode="nearest"), footprint=fp, mode="nearest")
        closing = np.maximum(np.repeat(np.repeat(c, step, 0), step, 1)[:target_h.shape[0], :target_h.shape[1]], target_h)
        top = float(target_h.max())
    regions = []
    for kind, src in (("excess", dev), ("gouge", -dev)):
        labels, n = _label(src > tol)
        for r in range(1, n + 1):
            m = labels == r
            vals = src[m]
            if thickness_fn is not None and kind == "excess":
                vals = np.minimum(vals, thickness_fn(m))
            vol = float(vals.sum() * cell)
            if vol < min_volume_mm3:
                continue
            ii, jj = np.nonzero(m)
            kmax = int(np.argmax(src[m]))
            x, y = grid.x0 + jj.mean() * grid.res, grid.y0 + ii.mean() * grid.res
            xm, ym = grid.x0 + jj[kmax] * grid.res, grid.y0 + ii[kmax] * grid.res
            ci, cj = int(round(ii.mean())), int(round(jj.mean()))
            compact = 0 <= ci < m.shape[0] and 0 <= cj < m.shape[1] and m[ci, cj]  # the mean lies in the region (not a ring)
            feat = _feature_at(features, xm, ym) or (_feature_at(features, x, y) if compact else None)
            fid = feat["id"] if feat else "outside / top face"
            owners = owner[m]
            owners = owners[owners >= 0]
            last = by_index.get(int(np.bincount(owners).argmax())) if len(owners) else None
            related = [o for o in ops_meta if feat and feat["id"] in (o.get("feature_ids") or [])]
            reg = {"kind": kind, "max_mm": round(float(src[m].max()), 3), "volume_mm3": round(vol, 1), "area_mm2": round(float(m.sum() * cell), 1),
                   "at": [round(xm, 1), round(ym, 1)], "feature": fid, "feature_type": feat.get("type") if feat else None,
                   "last_cut_by_step": last["step"] if last else None, "last_cut_by": last["process"] if last else None,
                   "steps_on_feature": [o["step"] for o in related], "klass": "error", "note": ""}
            if kind == "gouge" and surface_gouge is not None and not (m & surface_gouge).any():
                reg.update(klass="artefact", note="no surface sample is gouged here: a flank grazing wall columns of the simulation grid")
            if kind == "gouge" and feat and feat.get("type") == "hole" and not feat.get("through") and reg["klass"] == "error":
                drills = [o for o in related if o.get("category") == "twist_drill"]
                tip = max((o["tool_diameter"] / 2 / math.tan(math.radians(o.get("tip_angle", 118.0) / 2)) for o in drills), default=0.0)
                if reg["max_mm"] <= tip + 0.25:
                    reg.update(klass="expected", note="drill-point cone below a flat-bottomed blind hole")
            if kind == "excess" and closing is not None:
                exc = src[m]
                unreachable = np.clip(closing[m] - target_h[m], 0.0, None)
                explained = float(np.minimum(exc, unreachable).sum()) / max(float(exc.sum()), 1e-9)
                deep = (top - target_h[m]) > tooling["max_reach"] + 0.5
                beyond = float(exc[deep].sum()) / max(float(exc.sum()), 1e-9)
                if explained >= 0.6 and float(unreachable.max()) > tol:
                    reg.update(klass="inherent", note=f"gap or inner corner narrower than the smallest endmill (D{tooling['min_d']:g}): smaller cutter, EDM or a design change")
                elif beyond >= 0.6:
                    reg.update(klass="limitation", note=f"deeper than the longest tool reach ({tooling['max_reach']:.0f} mm): opposite side or a longer tool")
            if kind == "excess" and feat and feat.get("type") in ("pocket", "slot", "step") and reg["klass"] == "error":
                mills = [o for o in related if o.get("category") in ("endmill", "ball_endmill", "bull_endmill")]
                r_tool = min((o["tool_diameter"] / 2 for o in mills), default=0.0)
                r_corner = float(feat.get("corner_radius") or 0.0)
                if mills and r_corner < r_tool - 1e-6 and reg["area_mm2"] <= 4 * r_tool ** 2 * (1 - math.pi / 4) * 1.6 + 2.0:
                    reg.update(klass="inherent", note=f"inner corner radius {r_corner:g} mm < tool radius {r_tool:g} mm (corner relief or a smaller finishing tool)")
            regions.append(reg)
    regions.sort(key=lambda r: -r["volume_mm3"])
    return regions


# ----------------------------------------------------------------------------------------------- setup simulation
def sim_grid(bbox_xy, pad: float, tol: float, max_cells: float) -> Grid:
    x0, y0, x1, y1 = bbox_xy
    x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
    res = max(tol / 2.0, math.sqrt((x1 - x0) * (y1 - y0) / max_cells))
    return Grid(x0, y0, x1, y1, round(res, 4))


def ops_meta_for(parsed: dict, operations: list[dict], tools: list[dict]) -> list[dict]:
    """Match parsed G-code operations to the request operations (by label, then by step prefix) and tools."""
    tool_by_no = {t["tool_number"]: t for t in tools}
    by_label = {o["label"]: o for o in operations}
    by_step = {f"{o['step']:02d}": o for o in operations}
    meta = []
    for k, o in enumerate(parsed["ops"]):
        rq = by_label.get(o["name"]) or by_step.get(o["name"][:2]) or {}
        tool = tool_by_no.get(rq.get("tool_number")) or tool_by_no.get(o.get("tool")) or {}
        sp = tool.get("shape_params") or {}
        meta.append({"index": k, "step": rq.get("step"), "process": rq.get("process"), "feature_ids": rq.get("feature_ids", []), "tool": tool,
                     "tool_diameter": float(tool.get("diameter_mm") or sp.get("Diameter") or 0), "category": tool.get("category"),
                     "tip_angle": float(sp.get("TipAngle") or 118.0), "name": o["name"]})
    return meta


def simulate_setup(gcode_file: str, part_k: dict, stock_k: dict, operations: list[dict], tools: list[dict], features: list[dict],
                   tol: float = 0.1, max_cells: float = 9e6, tooling: dict | None = None, samples: int = 600000,
                   others: list[SetupState] | None = None, frame: Frame | None = None, stock_state: tuple | None = None) -> dict:
    """Replay one setup's G-code on the dexel grid of its frame and compare with the target.
    part_k / stock_k: meshes in the setup frame. others: the other setups' states (multi-setup), frame: this setup's
    frame (original -> setup), stock_state: (stock dexel, stock frame) of the whole job."""
    parsed = gcode.parse(gcode_file)
    tool_max_r = max([float(t.get("diameter_mm") or 0) / 2 for t in tools] + [1.0])
    bb = [min(part_k["bbox"][0], stock_k["bbox"][0]), min(part_k["bbox"][1], stock_k["bbox"][1]),
          max(part_k["bbox"][3], stock_k["bbox"][3]), max(part_k["bbox"][4], stock_k["bbox"][4])]
    grid = sim_grid(bb, tool_max_r + 1.0, tol, max_cells)
    dm, target_h = stock_dexel(grid, stock_k, part_k, return_target=True)
    meta = ops_meta_for(parsed, operations, tools)
    for m, o in zip(meta, parsed["ops"]):
        dm.run_moves(o["moves"], shape_for(m["tool"], m["process"] or "") if m["tool"] else None, m["index"])
    target_h = np.where(np.isfinite(target_h), target_h, float(dm.bottom.min()))
    res = {"grid": grid, "dexel": dm, "parsed": parsed, "ops_meta": meta, "target_h": target_h}
    # surface metric in the setup frame (single setup) or in the original frame (multi-setup workpiece)
    if others is None:
        inside = dm.inside
        target = part_k
        feats = features
    else:
        stock_dm, stock_frame = stock_state
        wp = Workpiece(stock_dm, stock_frame, [SetupState(frame, dexel=dm)] + others)
        inside = wp.inside
        target = {"vertices": frame.to_original(np.asarray(part_k["vertices"], float)), "faces": part_k["faces"]}
        feats = []
    sd = surface_deviation(target, inside, tol, samples, feats, cell=grid.res)
    dev = dm.h - target_h
    dev = np.where(np.abs(dev) <= tol, 0.0, dev)
    dev[~(dm.material() | (target_h > dm.bottom))] = 0.0
    thickness_fn = None
    if others is not None:
        def _thickness(m, dm=dm, th=target_h, wp=wp):
            ii, jj = np.nonzero(m)
            z0, z1 = th[ii, jj], dm.h[ii, jj]
            ts = (np.arange(6) + 0.5) / 6
            Zs = z0[None, :] + (z1 - z0)[None, :] * ts[:, None]
            X = grid.x0 + jj * grid.res
            Y = grid.y0 + ii * grid.res
            P = np.stack([np.broadcast_to(X, Zs.shape), np.broadcast_to(Y, Zs.shape), Zs], -1).reshape(-1, 3)
            frac = wp.inside(frame.to_original(P)).reshape(Zs.shape).mean(0)
            return frac * np.maximum(z1 - z0, 0)
        thickness_fn = _thickness
    pb = part_k["bbox"]
    # where the surface metric itself sees a gouge (setup frame cells, a cell of slack around every sample)
    surf_g = np.zeros(grid.shape, bool)
    gs = sd["dev"] < -tol
    if gs.any():
        Pg = sd["points"][gs] if others is None else frame.to_setup(sd["points"][gs])
        gi, gj, gok = grid.index(Pg[:, 0], Pg[:, 1])
        surf_g[gi[gok], gj[gok]] = True
        from scipy import ndimage as _nd
        surf_g = _nd.binary_dilation(surf_g, iterations=2)
    regions = find_regions(dev, grid, dm.owner, target_h, features, meta, tol, tooling, pb, thickness_fn, surface_gouge=surf_g)
    stats = {**sd["stats"], "sim_cell_mm": grid.res, "sim_grid": list(grid.shape), "removed_cm3": None}
    res.update(stats=stats, regions=regions, errors=[r for r in regions if r["klass"] == "error"], surface=sd)
    return res
