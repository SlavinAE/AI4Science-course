#!/usr/bin/env python3
"""Week 1, Experiment 2: does the wall-with-undercut metric count excess > tol?"""

import sys
sys.path.insert(0, "vendor")

from camverify.verify import surface_deviation
import numpy as np

# --- target mesh: 100 x 60 x 20 block ---
V = np.array([
    [0, 0, 0], [100, 0, 0], [100, 60, 0], [0, 60, 0],
    [0, 0, 20], [100, 0, 20], [100, 60, 20], [0, 60, 20],
], dtype=float)

F = np.array([
    [0, 2, 1], [0, 3, 2],
    [4, 5, 6], [4, 6, 7],
    [0, 1, 5], [0, 5, 4],
    [2, 3, 7], [2, 7, 6],
    [1, 2, 6], [1, 6, 5],
    [0, 4, 7], [0, 7, 3],
], dtype=np.int64)

target = {"vertices": V, "faces": F}

# --- inside for the "machined" part: x up to 100.11 (undercut on x=100 face) ---
def inside(P: np.ndarray) -> np.ndarray:
    return ((P[:, 0] >= 0) & (P[:, 0] <= 100.11) &
            (P[:, 1] >= 0) & (P[:, 1] <= 60) &
            (P[:, 2] >= 0) & (P[:, 2] <= 20))

# --- run both cell sizes ---
tol = 0.1
results = []
for cell in [0.0, 0.05]:
    r = surface_deviation(target, inside, tol, cell=cell)
    stats = r["stats"]
    results.append((cell, stats["surface_within_tol_pct"], stats["excess_area_pct"]))
    print(f"cell={cell}: surface_within_tol_pct={stats['surface_within_tol_pct']}%, "
          f"excess_area_pct={stats['excess_area_pct']}%, "
          f"max_excess={stats['max_excess_mm']}mm")

# --- write results file ---
import os
os.makedirs("results", exist_ok=True)
with open("results/w1_metric_wall.txt", "w") as f:
    for cell, within, excess in results:
        f.write(f"cell={cell}: surface_within_tol_pct={within}%, excess_area_pct={excess}%\n")

print("Results written to results/w1_metric_wall.txt")