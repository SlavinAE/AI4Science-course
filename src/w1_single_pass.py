import sys
sys.path.insert(0, "vendor")

from camverify.raster import Grid
from camverify.dexel import DexelMap, ToolShape
import numpy as np

# --- Setup ---
grid = Grid(0, 0, 100, 60, 0.05)
block_height = 20.0
dm = DexelMap(grid, np.full(grid.shape, block_height), np.zeros(grid.shape))

# --- Volume before cut ---
vol_before = dm.volume()

# --- Tool & pass ---
tool = ToolShape("flat", 5.0)  # radius 5 mm (diameter 10 mm)
dm.cut(tool, 25, 30, 18, 75, 30, 18)

# --- Volume after cut ---
vol_after = dm.volume()

# --- Removed volume ---
removed = vol_before - vol_after

print(f"Grid resolution : {grid.res} mm")
print(f"Grid size       : {grid.nx} x {grid.ny} cells")
print(f"Volume before   : {vol_before:.4f} mm³")
print(f"Volume after    : {vol_after:.4f} mm³")
print(f"Removed volume  : {removed:.4f} mm³")