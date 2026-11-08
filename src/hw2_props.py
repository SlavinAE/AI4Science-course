"""ДЗ 2: свойства P1–P8 метрики (METHOD.md, раздел 4 и Поправка 1). Каждая функция возвращает словарь с числом
нарушений `violations` (0 — свойство выполнено) и данными для журнала. Мутанты обёртки (`mut`) портят вход или
выход метрики снаружи снимка — так проверяется, что свойство вообще способно упасть:

  phase      inside сдвинут на cell/2 по X (фаза колонки dexel)
  grow       обработанный блок раздут на 0,05 мм во все стороны
  tol        метрика считает с допуском 0,125 вместо 0,1
  normals    нормали точек развёрнуты внутрь (только 21a0d91: в 13d20b5 точки выбираются внутри)
  unordered  порядок вердиктов не сохраняется между вызовами (перестановка, своя для каждого вызова)
  strict     классы недореза и зареза строго больше tol (определение снимка 13d20b5)
  radius     радиус фрезы в модели материала на 0,01 мм больше
"""
from __future__ import annotations

import zlib

import numpy as np

import hw2_parts as H

GROW, PHASE, TOL_MUT, DR = 0.05, H.CELL / 2, 0.125, 0.01
P6_DELTAS = tuple(d for d in H.DELTAS if d not in (0.05, 0.1, 0.15, 0.3))  # Поправка 1: не ступени лестницы и не tol


def _d1(impl: str, face: str | None, d: float, n: int, seed: int, pose: str = "id", mut: str | None = None) -> tuple[dict, object]:
    """Метрика на блоке D1/D2. Возвращает (результат, истинный inside) — истина не видит мутанта."""
    ps = H.POSES[pose]
    true_inside = H.posed(H.Box(face, d).inside, ps)
    m_inside = H.posed(H.Box(face, d, GROW if mut == "grow" else 0.0).inside, ps)
    if mut == "phase":
        m_inside = (lambda f: (lambda P: f(P - [PHASE, 0.0, 0.0])))(m_inside)
    target = H.box_mesh(pose=ps)
    tol = TOL_MUT if mut == "tol" else H.TOL
    if mut == "normals":
        if impl != "21a0d91":
            raise ValueError("мутант normals — только для 21a0d91")
        s = H.load(impl, "verify").mesh_samples(target["vertices"], target["faces"], n, seed)
        r = H.metric(impl, m_inside, n, seed, samples={**s, "N": -s["N"]}, tol=tol)
        r["N"] = s["N"]
    else:
        r = H.metric(impl, m_inside, n, seed, target=target, tol=tol)
    return _post(r, mut, seed, face, d, pose), true_inside


def _post(r: dict, mut: str | None, seed: int, *key) -> dict:
    if mut == "strict":
        r["excess"], r["gouge"] = r["dev"] > H.TOL, r["dev"] < -H.TOL
    if mut == "unordered":
        perm = np.random.default_rng(zlib.crc32(repr((seed, *key)).encode())).permutation(len(r["dev"]))
        for k in ("dev", "within", "excess", "gouge"):
            r[k] = r[k][perm]
    return r


def _verdict_vs_truth(r: dict, inside, extra_band=None) -> dict:
    t = H.ray_truth(inside, r["P"], r["N"])
    band = t["band"] if extra_band is None else t["band"] | extra_band
    bad = (r["within"] == t["out"]) & ~band
    return {"violations": int(bad.sum()), "band": int(band.sum()), "points": int(len(bad)),
            "truth_out": int((t["out"] & ~band).sum()), "metric_out": int((~r["within"]).sum()),
            "within_pct": r["stats"]["surface_within_tol_pct"]}


# ------------------------------------------------------------------------------------------- P1–P6: блок
def p1_identity(impl, n, seed, pose="id", mut=None) -> dict:
    """Материал = цель ⇒ все точки в допуске, отклонение 0, классов недореза и зареза нет."""
    r, _ = _d1(impl, None, 0.0, n, seed, pose, mut)
    v = int((~r["within"]).sum() + (r["dev"] != 0).sum() + (r["excess"] | r["gouge"]).sum())
    return {"violations": v, "within_pct": r["stats"]["surface_within_tol_pct"], "max_abs_dev": float(np.abs(r["dev"]).max())}


def p2_threshold(impl, n, seed, face, d, mut=None) -> dict:
    """Вердикт каждой точки совпадает с истиной по лучу на аналитическом блоке (вне полосы EPS)."""
    r, inside = _d1(impl, face, d, n, seed, mut=mut)
    return _verdict_vs_truth(r, inside)


def p3_monotone(impl, n, seed, face, sign, mut=None) -> dict:
    """δ₁ < δ₂ ⇒ точки в допуске при δ₂ — подмножество точек в допуске при δ₁; |dev| не убывает."""
    prev, v = None, 0
    for d in H.DELTAS:
        r, _ = _d1(impl, face, sign * d, n, seed, mut=mut)
        if prev is not None:
            v += int((r["within"] & ~prev["within"]).sum() + (np.abs(r["dev"]) < np.abs(prev["dev"]) - 1e-12).sum())
        prev = r
    return {"violations": v}


def p4_partition(impl, n, seed, face, d, mut=None) -> dict:
    """Каждая точка ровно в одном классе: в допуске / недорез / зарез; доли снимка в сумме 100 % (±0,02)."""
    r, _ = _d1(impl, face, d, n, seed, mut=mut)
    cls = r["within"].astype(int) + r["excess"].astype(int) + r["gouge"].astype(int)
    st = r["stats"]
    total = st["surface_within_tol_pct"] + st["excess_area_pct"] + st["gouge_area_pct"]
    return {"violations": int((cls != 1).sum()) + int(abs(total - 100.0) > 0.02), "no_class": int((cls == 0).sum()),
            "pct_sum": round(total, 2), "within_pct": st["surface_within_tol_pct"]}


def p5_invariance(impl, n, seed, face, d, pose, mut=None) -> dict:
    """Поворот и сдвиг детали вместе с обработкой не меняют вердикт и отклонение ни одной точки."""
    r0, _ = _d1(impl, face, d, n, seed, "id", mut)
    r1, _ = _d1(impl, face, d, n, seed, pose, mut)
    err = float(np.abs(H.POSES[pose].fwd(r0["P"]) - r1["P"]).max())
    if err > 1e-9:
        raise RuntimeError(f"P5: точки двух положений не соответствуют друг другу (ошибка {err:.2e} мм)")
    v = int((r0["within"] != r1["within"]).sum() + (np.abs(r0["dev"] - r1["dev"]) > 1e-9).sum())
    return {"violations": v, "within_pct": (r0["stats"]["surface_within_tol_pct"], r1["stats"]["surface_within_tol_pct"])}


def p6_symmetry(impl, n, seed, face, d, mut=None) -> dict:
    """На точках сдвинутой грани: сдвиг на δ наружу и внутрь даёт тот же вердикт и отклонение с другим знаком."""
    ro, _ = _d1(impl, face, +d, n, seed, mut=mut)
    ri, _ = _d1(impl, face, -d, n, seed, mut=mut)
    on = H.face_mask(ro["P"], face)
    v = int((ro["within"][on] != ri["within"][on]).sum() + (np.abs(ro["dev"][on] + ri["dev"][on]) > 1e-9).sum())
    return {"violations": v, "face_points": int(on.sum())}


# ------------------------------------------------------------------------------------------- P7: модель материала
def p7_sweep(impl, case, n, seed, mut=None) -> dict:
    """SweepSetup.inside снимка совпадает с определением заметаемого объёма (вне полосы)."""
    n = min(n, H.INCL_POINTS) if case.endswith("incl") else n
    P = H.move_points(case, n, seed)
    removed, band = H.move_truth(case, P)
    st = H.move_sweep(impl, case, DR if mut == "radius" else 0.0)
    bad = (st.inside(P) == removed) & ~band
    return {"violations": int(bad.sum()), "band": int(band.sum()), "points": n, "removed": int(removed.sum())}


# ------------------------------------------------------------------------------------------- P8: сквозное, паз
def p8_slot(impl, n, seed, variant, d, mut=None) -> dict:
    """Паз D3: точки цели на настоящей поверхности, материал — exact снимка; вердикт = истина по лучу на Slot."""
    if impl != "21a0d91":
        raise ValueError("P8 — только 21a0d91: 13d20b5 выбирает точки по мешу цели, цилиндрические стенки на нём — хорды")
    z, R = (H.SLOT_Z - d, H.SLOT_R) if variant == "depth" else (H.SLOT_Z, H.SLOT_R + d)
    truth = H.Slot(z, R)
    wp = H.slot_exact(impl, z, R, DR if mut == "radius" else 0.0)
    s = H.slot_samples(n, seed)
    r = H.metric(impl, wp.inside, n, seed, samples=s, tol=TOL_MUT if mut == "tol" else H.TOL)
    r = _post(r, mut, seed, variant, d)
    return _verdict_vs_truth(r, truth.inside, truth.band(r["P"]))
