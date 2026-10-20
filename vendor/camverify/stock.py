"""Stock (blank) model shared by every stage: the default box (part bounding box + margins, defined in the primary
setup's frame) or a user blank (STEP solid in the part's coordinate system). Both are carried as a closed triangle mesh
in the part's ORIGINAL frame, so planning, CAM, simulation, verification and the videos use one representation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .voxel import Frame, VoxelGrid


def union_bbox(*boxes) -> list[float]:
    bs = np.array([list(map(float, b[:6])) for b in boxes if b is not None])
    return [*bs[:, :3].min(0).tolist(), *bs[:, 3:].max(0).tolist()]


def box_mesh(lo, hi) -> dict:
    x0, y0, z0 = map(float, lo)
    x1, y1, z1 = map(float, hi)
    V = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0], [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])
    F = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    return {"vertices": V.tolist(), "faces": F.tolist(), "bbox": [x0, y0, z0, x1, y1, z1]}


def transform(mesh: dict, frame: Frame) -> dict:
    """Mesh from the original frame into a setup frame (p_k = R p + t)."""
    V = frame.to_setup(np.asarray(mesh["vertices"], float))
    return {"vertices": V, "faces": np.asarray(mesh["faces"], np.int64), "bbox": [*V.min(0).tolist(), *V.max(0).tolist()]}


def to_original(mesh: dict, frame: Frame) -> dict:
    V = frame.to_original(np.asarray(mesh["vertices"], float))
    return {"vertices": V, "faces": np.asarray(mesh["faces"], np.int64), "bbox": [*V.min(0).tolist(), *V.max(0).tolist()]}


def default_stock(part_raw: dict, primary: Frame, margins: dict) -> dict:
    """Bounding box of the part in the primary setup frame + margins (xy each side, z_top, z_bottom), back in the
    original frame. Mesh dict with kind / margins metadata."""
    Vk = primary.to_setup(np.asarray(part_raw["vertices"], float))
    lo, hi = Vk.min(0), Vk.max(0)
    m = float(margins.get("xy_margin", 2.0))
    box = box_mesh([lo[0] - m, lo[1] - m, lo[2] - float(margins.get("z_bottom", 0.0))], [hi[0] + m, hi[1] + m, hi[2] + float(margins.get("z_top", 1.0))])
    out = to_original(box, primary)
    out.update(kind="box", margins=dict(margins), frame_up=primary.up)
    return out


def allowances(stock_k: dict, part_k_bbox) -> dict:
    """Material allowance around the part in a setup frame (mm): top, bottom and the smallest side allowance."""
    s, p = stock_k["bbox"], part_k_bbox
    return {"top": round(s[5] - p[5], 3), "bottom": round(p[2] - s[2], 3),
            "side": round(min(p[0] - s[0], p[1] - s[1], s[3] - p[3], s[4] - p[4]), 3)}


def save_mesh(path: str | Path, mesh: dict) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, vertices=np.asarray(mesh["vertices"], float), faces=np.asarray(mesh["faces"], np.int64),
                        bbox=np.asarray(mesh["bbox"], float))
    return str(path)


def load_mesh(path: str | Path) -> dict:
    d = np.load(path)
    return {"vertices": d["vertices"], "faces": d["faces"], "bbox": d["bbox"].tolist()}


def fit_check(part_mesh: dict, stock_mesh: dict, max_voxels: float = 2e6) -> dict:
    """Share of the part that lies outside the stock (a blank in the wrong coordinate system shows up here)."""
    grid = VoxelGrid(union_bbox(part_mesh["bbox"], stock_mesh["bbox"]), margin=1.0, max_voxels=max_voxels, min_res=0.1)
    part = grid.voxelize(part_mesh["vertices"], part_mesh["faces"])
    stock = grid.voxelize(stock_mesh["vertices"], stock_mesh["faces"])
    cv = grid.cell_volume / 1000
    n_part = max(int(part.sum()), 1)
    return {"part_outside_stock_pct": round(100.0 * float((part & ~stock).sum()) / n_part, 2),
            "stock_cm3": round(float(stock.sum()) * cv, 3), "part_cm3": round(float(part.sum()) * cv, 3),
            "removal_cm3": round(float((stock & ~part).sum()) * cv, 3), "voxel_mm": round(grid.res, 3)}
