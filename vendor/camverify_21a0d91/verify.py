"""Verification of simulated machining against the target B-Rep - ONE metric for the report, the repair loop (round
ranking), the multi-setup checks, the videos and the external `evaluate` command: `surface_metric`.

Surface metric (experiment 2, approach E): area-weighted random points ON THE B-REP faces of the target (projected onto
the true surfaces, exact outward normals: fc_scripts/sample_brep.py); at every point P the EXACT workpiece
(`exact.py`: the blank AND every setup's swept volumes, arcs at 0.0005 mm) is probed along the normal N at
s = tol/4, tol/2, 3 tol/4, tol, 1.05 tol, 1.5 tol, ... 20 mm, without any shift of the point:
    material left  e = the largest s with material at every P + s' N, s' <= s   (outside the target surface)
    overcut        g = the largest s with air at every P - s' N, s' <= s          (inside the target)
P is in tolerance when e < tol and g < tol (`surface_within_tol_pct`, the share of surface AREA). With the probe at
1.05 tol a point out of tolerance only between tol and 1.05 tol is in the band (`band_pct`); `surface_within_band_pct`
counts it in. Volumes are surface integrals (sum of point area x deviation). On three parts with known deviations this
metric met the analytic truth within the sampling error (0.01-0.06 pp).

The dexel simulation (one height map per setup) is not a metric any more: its column deviation map only segments the
defects into regions and attributes them to features and operations for the repair loop, and a region counts as an
error only where the metric's own points are out of tolerance.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from . import gcode
from .dexel import DexelMap, shape_for, stock_dexel
from .raster import Grid
from .voxel import Frame

BAND = 1.05  # second probe of the metric: out of tolerance only below BAND * tol = the band


# ----------------------------------------------------------------------------------------------- B-rep samples
def brep_samples(path, n: int, workdir, freecad=None, seed: int = 0, deflection: float = 0.025) -> dict:
    """The metric's points on the B-rep faces of the part (original model frame): {"P", "N", "w", "face", "info"}.
    Cached in workdir by the file's content, n, seed and deflection."""
    path, workdir = Path(path), Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha1(path.read_bytes()).hexdigest()[:12]
    # other normals with the two-sided vote: not the cached points of the old one (evolution A06)
    v2 = "v2_" if os.environ.get("CAM_NORMAL_VOTE2", "0") == "1" else ""
    prefix = workdir / f"brep_samples_{v2}{h}_{n}_{seed}_{deflection:g}"
    meta = prefix.with_suffix(".json")
    if not meta.exists():
        if freecad is None:
            from .freecad_runner import FreeCADRunner
            freecad = FreeCADRunner()
        info = freecad.run("sample_brep.py", {"path": str(path), "n": n, "seed": seed, "deflection": deflection, "raw": True,
                                              "out": str(prefix)}, workdir)
        meta.write_text(json.dumps({k: v for k, v in info.items() if not k.startswith("_")}, indent=1))
    info = json.loads(meta.read_text())
    out = {"P": np.fromfile(f"{prefix}.P.bin", np.float64).reshape(-1, 3), "N": np.fromfile(f"{prefix}.N.bin", np.float64).reshape(-1, 3),
           "w": np.fromfile(f"{prefix}.w.bin", np.float64), "face": np.fromfile(f"{prefix}.face.bin", np.int32), "info": info}
    if len(out["P"]) != len(out["w"]):
        raise ValueError(f"broken B-rep samples {prefix}: {len(out['P'])} points, {len(out['w'])} weights")
    return out


def samples_in(samples: dict, frame: Frame | None) -> dict:
    """The samples in a setup frame (points p_k = R p + t, normals R n); frame None = unchanged."""
    if frame is None:
        return samples
    return {**samples, "P": frame.to_setup(samples["P"]), "N": samples["N"] @ frame.R.T}


def mesh_samples(verts, faces, n: int, seed: int = 0) -> dict:
    """Fallback samples on a mesh (only for callers without the part's STEP file, e.g. old test scripts): the points
    lie on chords of curved faces, up to the tessellation's deflection inside or outside the true surface."""
    P, N, w = sample_surface(verts, faces, n, seed)
    return {"P": P, "N": N, "w": w, "face": np.zeros(n, np.int32), "info": {"samples": n, "on": "mesh"}}


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


def probe_ladder(tol: float, max_probe: float = 20.0) -> np.ndarray:
    return np.asarray(sorted({round(x, 6) for x in [tol / 4, tol / 2, 3 * tol / 4, tol, BAND * tol, 1.5 * tol, 2 * tol, 3 * tol, 5 * tol,
                                                     0.5, 1, 2, 3, 5, 10, max_probe] if 0 < x <= max_probe}))


def _run_length(inside, P, N, sign: float, L: np.ndarray, want: bool) -> np.ndarray:
    """Per point the largest ladder value s_k such that inside(P + sign s N) == want for every s <= s_k (0: already
    not at the first rung). Only the points still running are probed at the next rung."""
    run = np.zeros(len(P))
    alive = np.arange(len(P))
    for s in L:
        if not len(alive):
            break
        alive = alive[inside(P[alive] + (sign * s) * N[alive]) == want]
        run[alive] = s
    return run


def surface_metric(samples: dict, inside, tol: float, features: list[dict] | None = None, max_probe: float = 20.0) -> dict:
    """THE surface metric (module docstring). samples: {"P", "N", "w"} in the frame of `inside(P)` (exact workpiece);
    features: drill-point cones (expected_mask) in that frame."""
    P, N, w = samples["P"], samples["N"], samples["w"]
    L = probe_ladder(tol, max_probe)
    excess = _run_length(inside, P, N, 1.0, L, True)    # material outside the target surface
    gouge = _run_length(inside, P, N, -1.0, L, False)   # material missing inside the target
    within = (excess < tol - 1e-9) & (gouge < tol - 1e-9)
    within_band = (excess < BAND * tol - 1e-9) & (gouge < BAND * tol - 1e-9)
    dev = np.where(excess > 0, excess, -gouge)
    expected = expected_mask(P, features or [])
    area = w.sum()
    pct = lambda m: round(100 * float(w[m].sum() / area), 2)  # noqa: E731
    info = samples.get("info") or {}
    stats = {"surface_within_tol_pct": pct(within), "surface_within_tol_pct_excl_expected": pct(within | expected),
             "surface_within_band_pct": pct(within_band), "band_pct": pct(within_band & ~within), "band_factor": BAND,
             "tol_mm": tol, "samples": int(len(P)), "samples_on": info.get("on", "brep"), "surface_mm2": round(float(area), 1),
             "excess_cm3": round(float((w * excess).sum()) / 1000, 3), "gouge_cm3": round(float((w * gouge).sum()) / 1000, 3),
             "max_excess_mm": round(float(excess.max()), 3), "max_gouge_mm": round(float(gouge.max()), 3),
             "excess_area_pct": pct(excess >= tol - 1e-9), "gouge_area_pct": pct(gouge >= tol - 1e-9),
             "probe_cap_mm": max_probe, "metric": "exact"}
    return {"stats": stats, "points": P, "normals": N, "w": w, "dev": dev, "within": within, "within_band": within_band, "expected": expected}


def save_surface(path, sd: dict, frame: str) -> None:
    """The metric's points and verdicts for the videos (frame: "setup" or "original")."""
    np.savez_compressed(path, P=sd["points"].astype(np.float32), N=sd["normals"].astype(np.float16), dev=sd["dev"].astype(np.float32),
                        within=sd["within"], within_band=sd["within_band"], expected=sd["expected"], frame=frame,
                        tol=sd["stats"]["tol_mm"])


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
                 thickness_fn=None, min_volume_mm3: float = 2.0, surface_gouge=None, surface_excess=None) -> list[dict]:
    """Segment a column deviation map (setup view) into excess / gouge regions, attribute each to a feature and to the
    operation that cut there last, and classify it: error | inherent | expected | limitation | artefact.
    surface_gouge / surface_excess: cells under the metric's points that are out of tolerance (overcut / material
    left); a column region without one is the column model's artefact, not a defect (a flank grazing a wall column's
    centre counts as a cut through the whole column; a wall known to +-cell/2 leaves a column of excess)."""
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
    from .cpu_fast import enabled
    whole = (slice(None), slice(None))
    for kind, src in (("excess", dev), ("gouge", -dev)):
        labels, n = _label(src > tol)
        # CAM_CPU_FAST: every region inside its bounding box, not a full-grid mask per region (regions x ~9M cells);
        # the cells come in the same row-major order, so the same sums and argmax
        boxes = ndimage.find_objects(labels) if enabled() else [whole] * n
        for r in range(1, n + 1):
            sl = boxes[r - 1]
            m = labels[sl] == r
            i0, j0 = sl[0].start or 0, sl[1].start or 0
            sub = lambda a: a[sl][m]  # noqa: E731

            def cells():
                ii, jj = np.nonzero(m)
                return ii + i0, jj + j0
            vals = sub(src)
            if thickness_fn is not None and kind == "excess":
                vals = np.minimum(vals, thickness_fn(*cells()))
            vol = float(vals.sum() * cell)
            if vol < min_volume_mm3:
                continue
            ii, jj = cells()
            kmax = int(np.argmax(sub(src)))
            x, y = grid.x0 + jj.mean() * grid.res, grid.y0 + ii.mean() * grid.res
            xm, ym = grid.x0 + jj[kmax] * grid.res, grid.y0 + ii[kmax] * grid.res
            ci, cj = int(round(ii.mean())) - i0, int(round(jj.mean())) - j0
            compact = 0 <= ci < m.shape[0] and 0 <= cj < m.shape[1] and m[ci, cj]  # the mean lies in the region (not a ring)
            feat = _feature_at(features, xm, ym) or (_feature_at(features, x, y) if compact else None)
            fid = feat["id"] if feat else "outside / top face"
            owners = sub(owner)
            owners = owners[owners >= 0]
            last = by_index.get(int(np.bincount(owners).argmax())) if len(owners) else None
            related = [o for o in ops_meta if feat and feat["id"] in (o.get("feature_ids") or [])]
            reg = {"kind": kind, "max_mm": round(float(sub(src).max()), 3), "volume_mm3": round(vol, 1), "area_mm2": round(float(m.sum() * cell), 1),
                   "at": [round(xm, 1), round(ym, 1)], "feature": fid, "feature_type": feat.get("type") if feat else None,
                   "last_cut_by_step": last["step"] if last else None, "last_cut_by": last["process"] if last else None,
                   "steps_on_feature": [o["step"] for o in related], "klass": "error", "note": ""}
            if kind == "gouge" and surface_gouge is not None and not (m & surface_gouge[sl]).any():
                reg.update(klass="artefact", note="no surface sample is gouged here: a flank grazing wall columns of the simulation grid")
            if kind == "gouge" and feat and feat.get("type") == "hole" and not feat.get("through") and reg["klass"] == "error":
                drills = [o for o in related if o.get("category") == "twist_drill"]
                tip = max((o["tool_diameter"] / 2 / math.tan(math.radians(o.get("tip_angle", 118.0) / 2)) for o in drills), default=0.0)
                if reg["max_mm"] <= tip + 0.25:
                    reg.update(klass="expected", note="drill-point cone below a flat-bottomed blind hole")
            if kind == "excess" and closing is not None:
                exc = sub(src)
                unreachable = np.clip(sub(closing) - sub(target_h), 0.0, None)
                explained = float(np.minimum(exc, unreachable).sum()) / max(float(exc.sum()), 1e-9)
                deep = (top - sub(target_h)) > tooling["max_reach"] + 0.5
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
            if kind == "excess" and surface_excess is not None and not (m & surface_excess[sl]).any():
                reg.update(klass="artefact", note="no point of the surface metric has material left beyond the tolerance here: column model only")
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


_DEXEL_CACHE: dict = {}  # (setup key, chain key of the operations so far) -> state; insertion order = age


def _setup_dexel(grid: Grid, stock_k: dict, part_k: dict, meta: list[dict], ops: list[dict]):
    """The stock dexel of the setup with every operation swept. CAM_CPU_FAST: the state after each operation is kept
    (per process, CAM_DEXEL_CACHE_MB, oldest dropped first), keyed by the setup (grid, stock, part) and a chain hash of
    the operations so far (index, tool shape, cutting segments); a repair round, which mostly changes the last
    operations, sweeps only those from the longest cached prefix. Same moves on the same state: the same map."""
    from .cpu_fast import enabled
    shapes = [shape_for(m["tool"], m["process"] or "") if m["tool"] else None for m in meta]
    if not enabled():
        dm, target_h = stock_dexel(grid, stock_k, part_k, return_target=True)
        for m, o, s in zip(meta, ops, shapes):
            dm.run_moves(o["moves"], s, m["index"])
        return dm, target_h
    hs = hashlib.sha1(repr((grid.x0, grid.y0, grid.res, grid.nx, grid.ny)).encode())
    for mesh in (stock_k, part_k):
        hs.update(np.ascontiguousarray(mesh["vertices"], np.float64).tobytes())
        hs.update(np.ascontiguousarray(mesh["faces"], np.int64).tobytes())
    skey = hs.hexdigest()
    chain, c = [], hashlib.sha1(b"stock")
    for m, o, s in zip(meta, ops, shapes):
        segs = np.asarray([mv[:6] for mv in o["moves"] if mv[6]], np.float64)
        c = c.copy()
        c.update(repr((m["index"], None if s is None else (s.kind, s.R, s.cr, s.tip_r, s.cot))).encode())
        c.update(segs.tobytes())
        chain.append(c.hexdigest())
    base = _DEXEL_CACHE.get((skey, "stock"))
    if base is None:
        dm, target_h = stock_dexel(grid, stock_k, part_k, return_target=True)
        base = (dm.h.copy(), dm.bottom.copy(), None if target_h is None else target_h.copy())
    _dexel_keep((skey, "stock"), base)  # (again on a hit: the newest, the last to be dropped)
    start, state = 0, None
    for i in range(len(chain), 0, -1):
        state = _DEXEL_CACHE.get((skey, chain[i - 1]))
        if state is not None:
            start = i
            _dexel_keep((skey, chain[i - 1]), state)
            break
    dm = DexelMap(grid, base[0] if state is None else state[0], base[1])
    if state is not None:
        dm.owner = state[1].copy()
    for i in range(start, len(chain)):
        if shapes[i] is None:  # run_moves does nothing: no state of its own
            continue
        dm.run_moves(ops[i]["moves"], shapes[i], meta[i]["index"])
        _dexel_keep((skey, chain[i]), (dm.h.copy(), dm.owner.copy()))
    return dm, (None if base[2] is None else base[2].copy())


def _dexel_keep(key, state) -> None:
    budget = float(os.environ.get("CAM_DEXEL_CACHE_MB", "2048")) * 2 ** 20
    _DEXEL_CACHE.pop(key, None)
    _DEXEL_CACHE[key] = state
    size = lambda st: sum(a.nbytes for a in st if a is not None)  # noqa: E731
    total = sum(size(st) for st in _DEXEL_CACHE.values())
    while total > budget and len(_DEXEL_CACHE) > 1:
        k0 = next(iter(_DEXEL_CACHE))
        total -= size(_DEXEL_CACHE.pop(k0))


def simulate_setup(gcode_file: str, part_k: dict, stock_k: dict, operations: list[dict], tools: list[dict], features: list[dict],
                   tol: float = 0.1, max_cells: float = 9e6, tooling: dict | None = None, samples: dict | None = None,
                   others: list[SetupState] | None = None, frame: Frame | None = None, stock_state: tuple | None = None,
                   exact_others: list | None = None, stock_raw: dict | None = None) -> dict:
    """Measure one setup's G-code with the surface metric on the exact workpiece, and replay it on the dexel grid of
    its frame for the defect regions (repair loop), rest machining and the videos.
    Single setup (others None): part_k / stock_k / samples in the setup frame.
    Multi-setup: samples, stock_raw and exact_others (the other setups' exact material: the SweepSetup of a programmed
    setup, the IdealView of one still to come) in the original frame; frame: this setup's (original -> setup);
    others / stock_state (stock dexel, its frame): the dexel counterparts, for the thickness of excess regions."""
    from . import exact  # numba
    if samples is None:
        raise ValueError("simulate_setup: the surface metric needs its samples (verify.brep_samples)")
    if others is None:
        sweep, sinfo = exact.sweep_setup(Frame("+Z", [0.0, 0.0, 0.0]), gcode_file, operations, tools)
        wp_exact = exact.blank_solid(stock_k, [sweep])
    else:
        sweep, sinfo = exact.sweep_setup(frame, gcode_file, operations, tools)
        wp_exact = exact.blank_solid(stock_raw, [sweep] + list(exact_others or []))
    sd = surface_metric(samples, wp_exact.inside, tol, features if others is None else [])
    parsed = gcode.parse(gcode_file)
    tool_max_r = max([float(t.get("diameter_mm") or 0) / 2 for t in tools] + [1.0])
    bb = [min(part_k["bbox"][0], stock_k["bbox"][0]), min(part_k["bbox"][1], stock_k["bbox"][1]),
          max(part_k["bbox"][3], stock_k["bbox"][3]), max(part_k["bbox"][4], stock_k["bbox"][4])]
    grid = sim_grid(bb, tool_max_r + 1.0, tol, max_cells)
    meta = ops_meta_for(parsed, operations, tools)
    dm, target_h = _setup_dexel(grid, stock_k, part_k, meta, parsed["ops"])
    target_h = np.where(np.isfinite(target_h), target_h, float(dm.bottom.min()))
    res = {"grid": grid, "dexel": dm, "parsed": parsed, "ops_meta": meta, "target_h": target_h, "sweep": sweep}
    if others is not None:
        stock_dm, stock_frame = stock_state
        wp = Workpiece(stock_dm, stock_frame, [SetupState(frame, dexel=dm)] + others)
    dev = dm.h - target_h
    dev = np.where(np.abs(dev) <= tol, 0.0, dev)
    dev[~(dm.material() | (target_h > dm.bottom))] = 0.0
    thickness_fn = None
    if others is not None:
        def _thickness(ii, jj, dm=dm, th=target_h, wp=wp):
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
    # cells under the metric's points that are out of tolerance (setup frame, two cells of slack; points on faces
    # turned away from this side do not count)
    Pk = sd["points"] if others is None else frame.to_setup(sd["points"])
    Nz = sd["normals"][:, 2] if others is None else (sd["normals"] @ frame.R.T)[:, 2]

    def cells(mask):
        out = np.zeros(grid.shape, bool)
        mask = mask & (Nz >= -0.5)
        if mask.any():
            gi, gj, gok = grid.index(Pk[mask, 0], Pk[mask, 1])
            out[gi[gok], gj[gok]] = True
            from scipy import ndimage as _nd
            out = _nd.binary_dilation(out, iterations=2)
        return out
    regions = find_regions(dev, grid, dm.owner, target_h, features, meta, tol, tooling, pb, thickness_fn,
                           surface_gouge=cells(sd["dev"] <= -tol + 1e-9), surface_excess=cells(sd["dev"] >= tol - 1e-9))
    stats = {**sd["stats"], "arc_tol_mm": exact.ARC_TOL, "blank": "box" if isinstance(wp_exact, exact.BoxWorkpiece) else "mesh",
             "exact_moves": sinfo["moves"], "regions_cell_mm": grid.res}
    res.update(stats=stats, regions=regions, errors=[r for r in regions if r["klass"] == "error"], surface=sd)
    return res
