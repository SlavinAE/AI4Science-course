"""Свойства P1–P8 метрики «доля поверхности в допуске» (METHOD.md, раздел 4; Поправка 1).

Запуск: .venv/bin/python -m pytest checks -q
Снимок — METRIC_IMPL (21a0d91 по умолчанию или 13d20b5), число точек — HW2_N (20 000), seed — HW2_SEED (0).
"""
import os

import pytest

import hw2_parts as H
import hw2_props as Pr

IMPL = H.impl_default()
N = int(os.environ.get("HW2_N", "20000"))
SEED = int(os.environ.get("HW2_SEED", "0"))
P7_N = 1_000_000
SIGNED = [s * d for s in (1, -1) for d in H.DELTAS]


def test_snapshot_integrity():
    r = H.snapshot_check(IMPL)
    assert r["ok"], r


@pytest.mark.parametrize("pose", list(H.POSES))
def test_p1_identity(pose):
    r = Pr.p1_identity(IMPL, N, SEED, pose)
    assert r["violations"] == 0, r


@pytest.mark.parametrize("d", SIGNED)
@pytest.mark.parametrize("face", list(H.FACES))
def test_p2_threshold(face, d):
    r = Pr.p2_threshold(IMPL, N, SEED, face, d)
    assert r["violations"] == 0, r


@pytest.mark.parametrize("sign", (1, -1))
@pytest.mark.parametrize("face", list(H.FACES))
def test_p3_monotone(face, sign):
    r = Pr.p3_monotone(IMPL, N, SEED, face, sign)
    assert r["violations"] == 0, r


@pytest.mark.parametrize("d", SIGNED)
@pytest.mark.parametrize("face", list(H.FACES))
def test_p4_partition(face, d):
    r = Pr.p4_partition(IMPL, N, SEED, face, d)
    assert r["violations"] == 0, r


@pytest.mark.parametrize("pose", ("shift", "rx", "ry", "rz"))
@pytest.mark.parametrize("d", (0.05, 0.11, -0.11))
@pytest.mark.parametrize("face", list(H.FACES))
def test_p5_invariance(face, d, pose):
    r = Pr.p5_invariance(IMPL, N, SEED, face, d, pose)
    assert r["violations"] == 0, r


@pytest.mark.parametrize("d", Pr.P6_DELTAS)
@pytest.mark.parametrize("face", list(H.FACES))
def test_p6_symmetry(face, d):
    r = Pr.p6_symmetry(IMPL, N, SEED, face, d)
    assert r["violations"] == 0, r


@pytest.mark.parametrize("case", list(H.MOVES))
def test_p7_sweep(case):
    r = Pr.p7_sweep(IMPL, case, P7_N, SEED)
    assert r["violations"] == 0, r


@pytest.mark.skipif(IMPL != "21a0d91", reason="P8 — только 21a0d91 (точки на настоящей поверхности паза)")
@pytest.mark.parametrize("variant,d", [("depth", 0.0)] + [(v, d) for v in ("depth", "radius") for d in H.DELTAS])
def test_p8_slot(variant, d):
    r = Pr.p8_slot(IMPL, N, SEED, variant, d)
    assert r["violations"] == 0, r
