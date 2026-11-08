"""Разбор падения P6 на снимке 13d20b5: нарушения — точки ближе cell/2 к ребру сдвинутой грани.

  .venv/bin/python src/hw2_p6_edges.py

Пишет results/hw2_p6_edges.txt и .json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hw2_parts as H  # noqa: E402

N, SEED, CELL2 = 20_000, 0, H.CELL / 2
CASES = [("x+", 0.11), ("z+", 0.11), ("x+", 0.09)]


def edge_dist(P: np.ndarray, face: str) -> np.ndarray:
    ax = H.FACES[face][0]
    other = [k for k in range(3) if k != ax]
    return np.min(np.stack([np.minimum(P[:, k] - H.LO[k], H.HI[k] - P[:, k]) for k in other]), axis=0)


def main() -> None:
    t0 = time.time()
    rows, all_ok = [], True
    for face, d in CASES:
        ro = H.metric("13d20b5", H.Box(face, +d).inside, N, SEED, target=H.box_mesh())
        ri = H.metric("13d20b5", H.Box(face, -d).inside, N, SEED, target=H.box_mesh())
        on = H.face_mask(ro["P"], face)
        bad = on & ((ro["within"] != ri["within"]) | (np.abs(ro["dev"] + ri["dev"]) > 1e-9))
        d_bad = edge_dist(ro["P"], face)[bad]
        ok = bool(bad.any()) and float(d_bad.max()) < CELL2
        all_ok &= ok
        rows.append({"face": face, "delta_mm": d, "face_points": int(on.sum()), "violations": int(bad.sum()),
                     "max_edge_dist_mm": None if not bad.any() else round(float(d_bad.max()), 4),
                     "min_edge_dist_mm": None if not bad.any() else round(float(d_bad.min()), 4),
                     "all_closer_than_cell2": ok})
    prov = H.provenance(("13d20b5",))
    out = {"n": N, "seed": SEED, "cell2_mm": CELL2, "cases": rows, "passed": all_ok,
           "seconds": round(time.time() - t0, 2), "provenance": prov}
    lines = [f"ДЗ 2, разбор P6 на 13d20b5: {time.strftime('%Y-%m-%d %H:%M:%S %z')}",
             f"коммит {prov['commit'][:7]}, n = {N}, seed = {SEED}, порог cell/2 = {CELL2} мм",
             "нарушения P6 — точки сдвинутой грани, у которых вердикт или |dev| не симметричны при ±δ", ""]
    for r in rows:
        lines.append(f"{r['face']} δ = {r['delta_mm']}: нарушений {r['violations']} из {r['face_points']} точек грани; "
                     f"расстояние до ребра {r['min_edge_dist_mm']}–{r['max_edge_dist_mm']} мм; "
                     f"{'все < cell/2' if r['all_closer_than_cell2'] else 'ЕСТЬ ТОЧКИ ≥ cell/2'}")
    lines += ["", "ИТОГ: " + ("все нарушения ближе cell/2 к ребру" if all_ok else "есть нарушения дальше cell/2")]
    (H.ROOT / "results").mkdir(exist_ok=True)
    (H.ROOT / "results" / "hw2_p6_edges.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str) + "\n")
    (H.ROOT / "results" / "hw2_p6_edges.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
