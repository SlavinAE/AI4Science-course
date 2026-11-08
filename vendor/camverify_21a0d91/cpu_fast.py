"""CAM_CPU_FAST (default 1): CPU-side speed-ups that give bit-identical results; CAM_CPU_FAST=0: the old code paths.

  * zlevel: the finishing gouge guard `allowed` once per operation, not once per Z level; flat dilations by a disk
    as running 1-D maxima along the rows (`dilate`), not scipy's footprint scan;
  * shield._morph (drop / closing): offsets of the tool disk sorted by the profile height, stop once no later offset can
    win (bound: max / min over the square around the cell); a flat core (bull: radius R - cr, h = 0) first as a plain
    dilation / erosion, the sorted ring from that value on;
  * `dilate` (all its uses): bands of rows in threads (the row filters release the GIL), every cell the same maxima in
    the same order;
  * dexel: the numba sweep over bands of grid rows in parallel (every cell sees the moves in the same order); a move
    piece visits only boxes whose 16 x 16 block maxima of h are above its low end, per row the chord of its capsule, and
    cells above its low end (no other cell can be lowered: the cut profile is >= 0);
  * raster.rasterize: a numba kernel over bands of grid rows (the numpy (triangle, cell) pairs with ufunc.at: ~x20);
  * verify.find_regions: every region inside its bounding box (ndimage.find_objects), not a full-grid mask per region;
  * verify._setup_dexel: the setup's dexel state after each operation kept per process (CAM_DEXEL_CACHE_MB, default
    2048): a repair round sweeps only the operations after the longest unchanged prefix;
  * shield.Shield.lifts (numba): a segment whose box holds no target above its lowest end (16 x 16 block maxima of T)
    gets lift 0 without the scan; the scan starts at lift 0, visits the block of the largest T first and skips the
    blocks / cells whose bound T + delta - min z cannot raise the lift, per row only the chord of the tool capsule;
  * rest.raster: the cells to visit (fixed cells dilated by the tool disk) by `dilate`, not scipy's footprint scan;
  * rest.plan_tools (ball / bull on the CPU): a candidate's closing only on the residual cells (its `ok` is read only
    there), its drop only within a tool disk of them; the rest of the drop only for the tools chosen.
"""
from __future__ import annotations

import os

import numpy as np
from scipy.ndimage import maximum_filter1d


def enabled() -> bool:
    return os.environ.get("CAM_CPU_FAST", "1") == "1"


def _row_widths(fp: np.ndarray):
    """Half-width of every footprint row if each row is one run centred on the middle column (a disk), else None."""
    fp = np.asarray(fp, bool)
    ny, nx = fp.shape
    if ny % 2 == 0 or nx % 2 == 0:
        return None
    c = nx // 2
    out = []
    for row in fp:
        idx = np.flatnonzero(row)
        if not len(idx):
            out.append(-1)
            continue
        w = c - idx[0]
        if idx[-1] - c != w or len(idx) != 2 * w + 1:
            return None
        out.append(int(w))
    return out


def dilate(A: np.ndarray, fp: np.ndarray, binary: bool = False) -> np.ndarray:
    """ndimage.grey_dilation(A, footprint=fp, mode="nearest") or, binary=True, ndimage.binary_dilation(A, structure=fp)
    for a footprint made of centred row runs (a disk): the maximum over its rows of a 1-D running maximum."""
    from scipy import ndimage
    widths = _row_widths(fp)
    if widths is None:  # not a disk: the general scan
        return ndimage.binary_dilation(A, structure=fp) if binary else ndimage.grey_dilation(A, footprint=fp, mode="nearest")
    X = np.asarray(A, np.uint8) if binary else np.asarray(A)
    ny = X.shape[0]
    if binary:
        out = np.zeros(X.shape, np.uint8)
    elif X.dtype.kind == "f":
        out = np.full(X.shape, -np.inf, X.dtype)
    else:
        out = np.full(X.shape, np.iinfo(X.dtype).min, X.dtype)
    nt = _threads()
    if nt > 1 and ny >= 8 * nt and X.size >= 1 << 20:  # bands of rows in threads (the filters release the GIL)
        from concurrent.futures import ThreadPoolExecutor
        step = -(-ny // nt)
        with ThreadPoolExecutor(nt) as ex:
            list(ex.map(lambda c0: _band(X, widths, binary, out, c0, min(c0 + step, ny)), range(0, ny, step)))
    else:
        _band(X, widths, binary, out, 0, ny)
    return out.astype(bool) if binary else out


def _threads() -> int:
    """Threads for `dilate`: the CPUs of this process, at most OMP_NUM_THREADS when it is set."""
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    omp = os.environ.get("OMP_NUM_THREADS", "")
    return max(1, min(n, int(omp))) if omp.isdigit() else n


def _band(X, widths, binary, out, c0, c1):
    """Rows c0 <= i < c1 of `dilate`: the row maxima of X over rows c0 - n .. c1 + n (all a band reads), every cell
    the same maxima in the same order as over the whole array."""
    ny = X.shape[0]
    n = len(widths) // 2
    s0, s1 = max(c0 - n, 0), min(c1 + n, ny)
    for w in sorted(set(widths) - {-1}):
        M = maximum_filter1d(X[s0:s1], 2 * w + 1, axis=1, mode="constant" if binary else "nearest", cval=0)
        for a, wa in enumerate(widths):
            if wa != w:
                continue
            da = a - n
            lo, hi = max(c0, -da), min(c1, ny - da)
            if lo < hi:
                np.maximum(out[lo:hi], M[lo + da - s0:hi + da - s0], out=out[lo:hi])
            if not binary:  # outside the array repeats the edge row (in place, no row copies)
                e = min(lo, c1)
                if c0 < e:  # i + da < 0: here s0 = 0
                    np.maximum(out[c0:e], M[0], out=out[c0:e])
                b = max(hi, c0)
                if b < c1:  # i + da > ny - 1: here s1 = ny
                    np.maximum(out[b:c1], M[ny - 1 - s0], out=out[b:c1])
