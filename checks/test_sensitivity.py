"""Чувствительность верификатора (METHOD.md, раздел 4): каждое свойство падает хотя бы на одном мутанте обёртки.
Мутанты портят вход или выход метрики снаружи снимка (hw2_props.py); снимки не меняются.
Проверяется на проверяемом снимке 21a0d91."""
import pytest

import hw2_parts as H
import hw2_props as Pr

N, SEED = 5000, 0
CASES = [
    ("P1", "phase", lambda m: Pr.p1_identity("21a0d91", N, SEED, mut=m)),
    ("P1", "grow", lambda m: Pr.p1_identity("21a0d91", N, SEED, mut=m)),
    ("P1", "normals", lambda m: Pr.p1_identity("21a0d91", N, SEED, mut=m)),
    ("P2", "tol", lambda m: Pr.p2_threshold("21a0d91", N, SEED, "x+", 0.11, mut=m)),
    ("P2", "grow", lambda m: Pr.p2_threshold("21a0d91", N, SEED, "x+", 0.0999, mut=m)),
    ("P2", "unordered", lambda m: Pr.p2_threshold("21a0d91", N, SEED, "x+", 0.11, mut=m)),
    ("P3", "unordered", lambda m: Pr.p3_monotone("21a0d91", N, SEED, "x+", 1, mut=m)),
    ("P4", "strict", lambda m: Pr.p4_partition("21a0d91", N, SEED, "x+", 0.1001, mut=m)),  # недорез ровно tol: δ ∈ [tol; 1,05 tol)
    ("P5", "phase", lambda m: Pr.p5_invariance("21a0d91", N, SEED, "x+", 0.05, "rz", mut=m)),
    ("P5", "unordered", lambda m: Pr.p5_invariance("21a0d91", N, SEED, "x+", 0.11, "ry", mut=m)),
    ("P6", "grow", lambda m: Pr.p6_symmetry("21a0d91", N, SEED, "x+", 0.0999, mut=m)),
    ("P7", "radius", lambda m: Pr.p7_sweep("21a0d91", "flat-horiz", 200_000, SEED, mut=m)),
    ("P8", "tol", lambda m: Pr.p8_slot("21a0d91", N, SEED, "depth", 0.11, mut=m)),
    ("P8", "radius", lambda m: Pr.p8_slot("21a0d91", N, SEED, "radius", 0.0999, mut=m)),
]


@pytest.mark.skipif(H.impl_default() != "21a0d91", reason="мутанты проверяются на проверяемом снимке 21a0d91")
@pytest.mark.parametrize("prop,mut,run", CASES, ids=[f"{p}-{m}" for p, m, _ in CASES])
def test_mutant_is_caught(prop, mut, run):
    r = run(mut)
    assert r["violations"] > 0, f"{prop} не падает на мутанте {mut}: {r}"


def test_every_property_has_a_mutant():
    assert {p for p, _, _ in CASES} == {f"P{k}" for k in range(1, 9)}
