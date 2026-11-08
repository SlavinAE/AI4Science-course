"""ДЗ 2: синтетические детали с аналитической истиной и единый вызов двух снимков метрики (METHOD.md, разделы 2–3).

Снимки: METRIC_IMPL=21a0d91 (vendor/camverify_21a0d91, verify.surface_metric) или 13d20b5 (vendor/camverify,
verify.surface_deviation со сдвигами проб на ±cell/2). Единицы — мм, допуск TOL = 0,1.
"""
from __future__ import annotations

import hashlib
import importlib
import math
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".numba_cache"))  # снимки только читаются: кэш numba — рядом
sys.path.insert(0, str(ROOT / "vendor"))

TOL = 0.1
CELL = 0.05  # ячейка dexel харнесса = tol/2: сдвиги проб снимка 13d20b5
EPS = 1e-6  # полоса исключения: вердикт меняется между tol - EPS и tol + EPS
EDGE = 1e-9  # граница обработанной детали — материал (замкнутое тело)
IMPLS = {"21a0d91": "camverify_21a0d91", "13d20b5": "camverify"}
SUMS = {"21a0d91": "SHA256SUMS_21a0d91", "13d20b5": "SHA256SUMS"}
LO, HI = np.zeros(3), np.array([100.0, 60.0, 20.0])
FACES = {"x-": (0, 0), "x+": (0, 1), "y-": (1, 0), "y+": (1, 1), "z-": (2, 0), "z+": (2, 1)}
FACE_AREA = {"x-": 1200.0, "x+": 1200.0, "y-": 2000.0, "y+": 2000.0, "z-": 6000.0, "z+": 6000.0}
BOX_AREA = 18400.0
DELTAS = (0.02, 0.05, 0.09, 0.0999, 0.1, 0.1001, 0.11, 0.12, 0.15, 0.3)
FULL_DELTAS = (0.05, 0.09, 0.11, 0.12, 0.15)
FULL_SEEDS = tuple(range(10))
FULL_N = 600_000


def impl_default() -> str:
    impl = os.environ.get("METRIC_IMPL", "21a0d91")
    if impl not in IMPLS:
        raise ValueError(f"METRIC_IMPL={impl}: ожидается одно из {sorted(IMPLS)}")
    return impl


def load(impl: str, name: str):
    return importlib.import_module(f"{IMPLS[impl]}.{name}")


# ------------------------------------------------------------------------------------------- положение детали
def rot(axis: int, deg: int = 90) -> np.ndarray:
    c, s = round(math.cos(math.radians(deg))), round(math.sin(math.radians(deg)))
    R = np.eye(3)
    i, j = [k for k in range(3) if k != axis]
    R[i, i], R[i, j], R[j, i], R[j, j] = c, -s, s, c
    return R


class Pose:
    """p' = R p + t: деталь и её обработка переносятся вместе."""

    def __init__(self, R=None, t=None) -> None:
        self.R = np.eye(3) if R is None else np.asarray(R, float)
        self.t = np.zeros(3) if t is None else np.asarray(t, float)

    def fwd(self, P: np.ndarray) -> np.ndarray:
        return P @ self.R.T + self.t

    def inv(self, P: np.ndarray) -> np.ndarray:
        return (P - self.t) @ self.R


SHIFT = (0.0123, 0.0377, 0.0191)
POSES = {"id": Pose(), "shift": Pose(t=SHIFT), "rx": Pose(rot(0), SHIFT), "ry": Pose(rot(1), SHIFT), "rz": Pose(rot(2), SHIFT)}


# ------------------------------------------------------------------------------------------- D1, D2: блок
def box_mesh(lo=LO, hi=HI, pose: Pose = POSES["id"]) -> dict:
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    V = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], float)
    F = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                  [2, 3, 7], [2, 7, 6], [1, 2, 6], [1, 6, 5], [0, 4, 7], [0, 7, 3]], np.int64)
    return {"vertices": pose.fwd(V), "faces": F}


class Box:
    """Обработанный блок: грань `face` сдвинута на d (d > 0 — наружу, недорез; d < 0 — внутрь, зарез);
    grow — блок, раздутый во все стороны (мутант)."""

    def __init__(self, face: str | None = None, d: float = 0.0, grow: float = 0.0) -> None:
        self.lo, self.hi = LO - grow, HI + grow
        if face:
            ax, side = FACES[face]
            if side:
                self.hi = self.hi.copy()
                self.hi[ax] += d
            else:
                self.lo = self.lo.copy()
                self.lo[ax] -= d

    def inside(self, P: np.ndarray) -> np.ndarray:
        return ((P >= self.lo - EDGE) & (P <= self.hi + EDGE)).all(1)


def posed(inside, pose: Pose):
    return lambda P: inside(pose.inv(P))


def face_mask(P0: np.ndarray, face: str, tol: float = 1e-9) -> np.ndarray:
    """Точки цели (в исходной системе блока) на грани face."""
    ax, side = FACES[face]
    return np.abs(P0[:, ax] - (HI if side else LO)[ax]) < tol


# ------------------------------------------------------------------------------------------- D3: паз
SLOT_A, SLOT_B, SLOT_Z, SLOT_R = (25.0, 30.0), (75.0, 30.0), 18.0, 5.0


def seg_dist(P: np.ndarray, a, b) -> np.ndarray:
    """XY-расстояние от точек до отрезка ab."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    u = b - a
    w = P[:, :2] - a
    L2 = float(u @ u)
    t = np.clip(w @ u / L2, 0.0, 1.0) if L2 > 0 else np.zeros(len(P))
    e = w - t[:, None] * u
    return np.hypot(e[:, 0], e[:, 1])


def in_block(P: np.ndarray) -> np.ndarray:
    return ((P >= LO - EDGE) & (P <= HI + EDGE)).all(1)


class Slot:
    """Аналитическая обработанная деталь D3: блок минус заметённый объём одного горизонтального хода плоской фрезы
    радиуса R с кончиком на высоте z (хвостовик вверх без конца)."""

    def __init__(self, z: float = SLOT_Z, R: float = SLOT_R) -> None:
        self.z, self.R = z, R

    def removed(self, P: np.ndarray) -> np.ndarray:
        return (P[:, 2] >= self.z) & (seg_dist(P, SLOT_A, SLOT_B) <= self.R)

    def inside(self, P: np.ndarray) -> np.ndarray:
        return in_block(P) & ~self.removed(P)

    def band(self, P: np.ndarray, N: np.ndarray) -> np.ndarray:
        """Точки цели, чья проба идёт вдоль боковой границы заметённого тела: нормаль вертикальна, XY-расстояние до оси
        хода в пределах EPS от R (там аналитика и exact могут разойтись в последнем знаке). У стенок паза проба уходит
        от границы, их неоднозначность ловит полоса по лучу."""
        return (np.abs(N[:, 2]) > 0.5) & (np.abs(seg_dist(P, SLOT_A, SLOT_B) - self.R) < EPS)


def true_within_d1(face: str, d: float) -> float:
    """Аналитическая доля поверхности блока в допуске, %: сдвинутая грань вне допуска при |δ| ≥ tol."""
    return 100.0 if abs(d) < TOL else 100.0 * (1.0 - FACE_AREA[face] / BOX_AREA)


def slot_total_area() -> float:
    return float(sum(a for _, a in slot_areas()))


def true_within_d3_depth(d: float) -> float:
    """Доля поверхности цели D3 в допуске, %: дно паза с зарезом δ (кончик на 18 − δ)."""
    ast = 2 * SLOT_R * (SLOT_B[0] - SLOT_A[0]) + math.pi * SLOT_R * SLOT_R
    return 100.0 if abs(d) < TOL else 100.0 * (1.0 - ast / slot_total_area())


def n_eff(w: np.ndarray) -> float:
    w = np.asarray(w, float)
    return float((w.sum() ** 2) / np.square(w).sum())


def slot_areas() -> list[tuple[str, float]]:
    L, H, R = SLOT_B[0] - SLOT_A[0], HI[2] - SLOT_Z, SLOT_R
    ast = 2 * R * L + math.pi * R * R
    return [("z-", 6000.0), ("x-", 1200.0), ("x+", 1200.0), ("y-", 2000.0), ("y+", 2000.0), ("top", 6000.0 - ast),
            ("floor", ast), ("wall-", L * H), ("wall+", L * H), ("arc-a", math.pi * R * H), ("arc-b", math.pi * R * H)]


def slot_samples(n: int, seed: int) -> dict:
    """Точки цели D3 по площади граней на настоящей поверхности (плоскости и полуцилиндры), наружные нормали;
    вес точки — площадь цели / n, как в sample_surface харнесса."""
    rng = np.random.default_rng(seed)
    faces = slot_areas()
    areas = np.array([a for _, a in faces])
    counts = rng.multinomial(n, areas / areas.sum())
    (ax, ay), R, z0 = SLOT_A, SLOT_R, SLOT_Z
    Ps, Ns, lab = [], [], []
    for k, ((name, _), m) in enumerate(zip(faces, counts)):
        if m == 0:
            continue
        U = rng.random((m, 2))
        if name in ("z-", "x-", "x+", "y-", "y+"):
            axis, side = FACES[name]
            i, j = [q for q in range(3) if q != axis]
            P = np.zeros((m, 3))
            P[:, i], P[:, j] = U[:, 0] * HI[i], U[:, 1] * HI[j]
            P[:, axis] = HI[axis] if side else 0.0
            N = np.zeros((m, 3))
            N[:, axis] = 1.0 if side else -1.0
        elif name in ("top", "floor"):
            keep, got = (lambda Q: seg_dist(Q, SLOT_A, SLOT_B) > R) if name == "top" else (lambda Q: seg_dist(Q, SLOT_A, SLOT_B) <= R), []
            lo, hi = (np.array([0.0, 0.0]), HI[:2]) if name == "top" else (np.array([ax - R, ay - R]), np.array([SLOT_B[0] + R, ay + R]))
            while sum(len(g) for g in got) < m:
                Q = np.zeros((2 * m + 64, 3))
                Q[:, :2] = lo + rng.random((len(Q), 2)) * (hi - lo)
                got.append(Q[keep(Q)])
            P = np.concatenate(got)[:m]
            P[:, 2] = HI[2] if name == "top" else z0
            N = np.tile([0.0, 0.0, 1.0], (m, 1))
        elif name in ("wall-", "wall+"):
            P = np.column_stack([ax + U[:, 0] * (SLOT_B[0] - ax), np.full(m, ay - R if name == "wall-" else ay + R), z0 + U[:, 1] * (HI[2] - z0)])
            N = np.tile([0.0, 1.0 if name == "wall-" else -1.0, 0.0], (m, 1))
        else:
            c = SLOT_A if name == "arc-a" else SLOT_B
            th = (math.pi / 2 if name == "arc-a" else -math.pi / 2) + U[:, 0] * math.pi
            P = np.column_stack([c[0] + R * np.cos(th), c[1] + R * np.sin(th), z0 + U[:, 1] * (HI[2] - z0)])
            N = np.column_stack([-np.cos(th), -np.sin(th), np.zeros(m)])
        Ps.append(P)
        Ns.append(N)
        lab.append(np.full(m, k, np.int32))
    return {"P": np.concatenate(Ps), "N": np.concatenate(Ns), "w": np.full(n, areas.sum() / n),
            "face": np.concatenate(lab), "faces": [f for f, _ in faces], "info": {"samples": n, "on": "analytic"}}


def slot_exact(impl: str, z: float = SLOT_Z, R: float = SLOT_R, dR: float = 0.0):
    """Обработанная деталь D3 на модели exact снимка: заготовка-коробка и заметённый объём одного хода."""
    exact, Frame = load(impl, "exact"), load(impl, "voxel").Frame
    M = np.array([[SLOT_A[0], SLOT_A[1], z, SLOT_B[0], SLOT_B[1], z, 0]], float)
    T = np.array([[0, R + dR, 0, 0, 0]], float)
    return exact.BoxWorkpiece(LO, HI, [exact.SweepSetup(M, T, Frame("+Z", (0, 0, 0)))])


# ------------------------------------------------------------------------------------------- D4: один ход
MOVES = {
    "flat-horiz": ("flat", (25.0, 30.0, 18.0), (75.0, 30.0, 18.0)),
    "ball-horiz": ("ball", (25.0, 30.0, 15.0), (75.0, 30.0, 15.0)),
    "flat-vert": ("flat", (50.0, 30.0, 20.0), (50.0, 30.0, 12.0)),
    "ball-vert": ("ball", (50.0, 30.0, 20.0), (50.0, 30.0, 12.0)),
    "flat-incl": ("flat", (20.0, 25.0, 19.0), (80.0, 40.0, 12.0)),
    "ball-incl": ("ball", (20.0, 25.0, 19.0), (80.0, 40.0, 12.0)),
}
MOVE_R = 5.0
INCL_POINTS, INCL_T, INCL_BAND = 4000, 100_000, 1e-4


def profile(kind: str, R: float, d: np.ndarray) -> np.ndarray:
    return np.zeros_like(d) if kind == "flat" else R - np.sqrt(np.maximum(R * R - d * d, 0.0))


def move_points(case: str, n: int, seed: int) -> np.ndarray:
    _, p0, p1 = MOVES[case]
    p0, p1 = np.array(p0), np.array(p1)
    lo = np.minimum(p0, p1) - [MOVE_R + 1, MOVE_R + 1, 1]
    hi = np.maximum(p0, p1) + [MOVE_R + 1, MOVE_R + 1, MOVE_R + 1]
    return lo + np.random.default_rng(seed).random((n, 3)) * (hi - lo)


def move_truth(case: str, P: np.ndarray, R: float = MOVE_R) -> tuple[np.ndarray, np.ndarray]:
    """(снят, полоса) по определению: P снят ⇔ ∃t: d(t) ≤ R и z ≥ z(t) + h(d(t)). Горизонтальный и вертикальный ход —
    формулой, наклонный — перебором INCL_T значений t (полоса INCL_BAND покрывает ошибку перебора)."""
    kind, p0, p1 = MOVES[case]
    p0, p1 = np.array(p0), np.array(p1)
    if np.allclose(p0[:2], p1[:2]) or abs(p1[2] - p0[2]) < 1e-12:
        d = np.hypot(P[:, 0] - p0[0], P[:, 1] - p0[1]) if np.allclose(p0[:2], p1[:2]) else seg_dist(P, p0[:2], p1[:2])
        floor = min(p0[2], p1[2]) + profile(kind, R, np.minimum(d, R))
        return (d <= R) & (P[:, 2] >= floor), (np.abs(d - R) < EPS) | ((d <= R + EPS) & (np.abs(P[:, 2] - floor) < EPS))
    t = np.linspace(0.0, 1.0, INCL_T)
    C = p0 + t[:, None] * (p1 - p0)
    rem, band = np.zeros(len(P), bool), np.zeros(len(P), bool)
    for k in range(0, len(P), 32):
        Q = P[k:k + 32]
        d = np.hypot(Q[:, None, 0] - C[None, :, 0], Q[:, None, 1] - C[None, :, 1])
        f = np.where(d <= R, C[None, :, 2] + profile(kind, R, np.minimum(d, R)), np.inf)
        fmin, dmin = f.min(1), d.min(1)
        rem[k:k + 32] = fmin <= Q[:, 2]
        band[k:k + 32] = (np.abs(Q[:, 2] - fmin) < INCL_BAND) | (np.abs(dmin - R) < INCL_BAND)
    return rem, band


def move_sweep(impl: str, case: str, dR: float = 0.0):
    exact, Frame = load(impl, "exact"), load(impl, "voxel").Frame
    kind, p0, p1 = MOVES[case]
    M = np.array([[*p0, *p1, 0]], float)
    T = np.array([[0 if kind == "flat" else 1, MOVE_R + dR, 0, 0, 0]], float)
    return exact.SweepSetup(M, T, Frame("+Z", (0, 0, 0)))


# ------------------------------------------------------------------------------------------- истина по лучу
def ray_truth(inside, P: np.ndarray, N: np.ndarray, tol: float = TOL, steps: int = 200) -> dict:
    """Истина вердикта на аналитической модели: недорез ≥ tol ⇔ материал во всех P + sN, s ∈ (0, tol]; зарез ≥ tol ⇔
    воздух во всех P − sN. Луч проверяется с шагом tol/steps. band — ответ разный при tol − EPS и tol + EPS."""
    S = np.linspace(tol / steps, tol - EPS, steps)

    def run(sign: float, want: bool):
        ok = np.ones(len(P), bool)
        for s in S:
            ok &= inside(P + (sign * s) * N) == want
        return ok, ok & (inside(P + (sign * (tol + EPS)) * N) == want)

    e_lo, e_hi = run(1.0, True)
    g_lo, g_hi = run(-1.0, False)
    band = (e_lo != e_hi) | (g_lo != g_hi)
    return {"out": e_lo | g_lo, "excess": e_lo, "gouge": g_lo, "band": band}


# ------------------------------------------------------------------------------------------- вызов метрики
def metric(impl: str, inside, n: int, seed: int, target: dict | None = None, samples: dict | None = None,
           tol: float = TOL, cell: float = CELL) -> dict:
    """Доля поверхности в допуске одним из снимков. target — меш цели (оба снимка: точки выбираются одной и той же
    sample_surface), samples — готовые точки (только 21a0d91). Классы точек — по определению снимка:
    21a0d91 — недорез/зарез ≥ tol, 13d20b5 — > tol (как его excess_area_pct / gouge_area_pct)."""
    verify = load(impl, "verify")
    if impl == "21a0d91":
        s = samples if samples is not None else verify.mesh_samples(target["vertices"], target["faces"], n, seed)
        r = verify.surface_metric(s, inside, tol)
        P, N, w = s["P"], s["N"], s["w"]
        exc, gou = r["dev"] >= tol - 1e-9, r["dev"] <= -(tol - 1e-9)
    else:
        if samples is not None or target is None:
            raise ValueError("13d20b5 выбирает точки сам по мешу цели: нужен target, samples не поддерживаются")
        r = verify.surface_deviation(target, inside, tol, n, seed=seed, cell=cell)
        P, N = r["points"], r["normals"]
        w = np.full(len(P), r["stats"]["surface_mm2"] / len(P))
        exc, gou = r["dev"] > tol, r["dev"] < -tol
    return {"P": P, "N": N, "w": w, "dev": np.asarray(r["dev"], float), "within": np.asarray(r["within"], bool),
            "excess": exc, "gouge": gou, "stats": r["stats"]}


# ------------------------------------------------------------------------------------------- происхождение
def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot_check(impl: str) -> dict:
    """SHA256 файлов снимка против его SHA256SUMS."""
    folder = ROOT / "vendor" / IMPLS[impl]
    want = {}
    for line in (ROOT / "vendor" / SUMS[impl]).read_text().split("\n"):
        if line.strip():
            h, name = line.split(maxsplit=1)
            want[name.lstrip("*")] = h
    got = {name: sha256(folder / name) for name in want}
    return {"folder": f"vendor/{IMPLS[impl]}", "sums": f"vendor/{SUMS[impl]}", "ok": got == want, "files": got}


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def provenance(impls=tuple(IMPLS)) -> dict:
    import numba
    import scipy
    dirty = git("status", "--porcelain", "--", "src", "checks", "vendor", "METHOD.md", "AGENT.md")
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(dirty), "dirty_files": dirty.split("\n") if dirty else [],
            "snapshots": {i: snapshot_check(i) for i in impls},
            "env": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                    "numba": numba.__version__, "platform": platform.platform()}}
