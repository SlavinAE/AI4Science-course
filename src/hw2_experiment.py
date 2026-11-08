"""ДЗ 2: шаги глубины (METHOD.md, раздел 7). Каждый шаг пишет results/hw2_<шаг>.json и .txt с коммитом проекта,
SHA256 снимков, seed и n и печатает «ПРОЙДЕН» или завершается кодом 1.

  .venv/bin/python src/hw2_experiment.py dry     пробный прогон
  .venv/bin/python src/hw2_experiment.py smoke   короткая проверка: pytest checks на обоих снимках
  .venv/bin/python src/hw2_experiment.py sweep   перебор δ × повороты × сдвиги на 21a0d91
  .venv/bin/python src/hw2_experiment.py full    полный расчёт: 10 seed, три варианта, правило п. 4–6

Шаг не запускается, если в src/, checks/, vendor/, METHOD.md или AGENT.md есть незакоммиченные изменения
(--allow-dirty — только для отладки, такой результат в отчёт не идёт).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hw2_parts as H  # noqa: E402
import hw2_props as Pr  # noqa: E402

RESULTS = H.ROOT / "results"
NEW, OLD = "21a0d91", "13d20b5"
EXPECTED_OLD_FAIL = {"p2", "p4", "p5", "p6"}  # METHOD.md, правило принятия, п. 3; P6 — Поправка 4


def write(step: str, data: dict, lines: list[str]) -> None:
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"hw2_{step}.json").write_text(json.dumps(data, indent=1, ensure_ascii=False, default=str) + "\n")
    (RESULTS / f"hw2_{step}.txt").write_text("\n".join(lines) + "\n")
    env = data["provenance"]["env"]
    (RESULTS / "env.txt").write_text("".join(f"{k}: {v}\n" for k, v in env.items()))


def header(step: str, prov: dict, params: dict) -> list[str]:
    snaps = ", ".join(f"{v['folder']} SHA256 {'OK' if v['ok'] else 'НЕ СОВПАДАЕТ'}" for v in prov["snapshots"].values())
    return [f"ДЗ 2, шаг «{step}»: {time.strftime('%Y-%m-%d %H:%M:%S %z')}",
            f"коммит проекта {prov['commit']}{' (ЕСТЬ НЕЗАКОММИЧЕННЫЕ ИЗМЕНЕНИЯ)' if prov['dirty'] else ''}",
            f"снимки: {snaps}", "параметры: " + ", ".join(f"{k}={v}" for k, v in params.items()), ""]


# ------------------------------------------------------------------------------------------- пробный прогон
def dry(prov: dict) -> tuple[dict, list[str], bool]:
    """Оба снимка импортируются, numba компилируется; D1 с недорезом 0,11 мм на грани x+, n = 10⁴, seed 0; совпадение
    точек двух снимков при одном seed; P1 на проверяемом снимке."""
    n, seed, face, d = 10_000, 0, "x+", 0.11
    out, t = {"params": {"n": n, "seed": seed, "face": face, "delta_mm": d, "tol_mm": H.TOL, "cell_mm": H.CELL}}, {}
    lines = header("пробный прогон", prov, out["params"])

    t0 = time.time()
    for impl in (NEW, OLD):
        for mod in ("verify", "exact"):
            H.load(impl, mod)
    t["import_s"] = time.time() - t0

    target, inside = H.box_mesh(), H.Box(face, d).inside
    runs = {}
    for name, impl, cell in ((f"{NEW}", NEW, H.CELL), (f"{OLD} cell=0.05", OLD, H.CELL), (f"{OLD} cell=0", OLD, 0.0)):
        t0 = time.time()
        runs[name] = H.metric(impl, inside, n, seed, target=target, cell=cell)
        t[f"metric {name}_s"] = time.time() - t0
    ref = runs[NEW]
    same = {k: bool(np.array_equal(ref["P"], r["P"]) and np.array_equal(ref["N"], r["N"])) for k, r in runs.items() if k != NEW}
    out["same_points_as_new"] = same

    truth = H.ray_truth(inside, ref["P"], ref["N"])
    face_share = 100.0 * 1200.0 / 18400.0
    out["analytic_within_pct"] = round(100.0 - face_share, 3)
    out["truth_sample"] = {"out": int(truth["out"].sum()), "band": int(truth["band"].sum()),
                           "within_pct_on_sample": round(100.0 * float(ref["w"][~truth["out"]].sum() / ref["w"].sum()), 2)}
    out["variants"] = {}
    for k, r in runs.items():
        st = r["stats"]
        out["variants"][k] = {"within_pct": st["surface_within_tol_pct"], "excess_area_pct": st["excess_area_pct"],
                              "gouge_area_pct": st["gouge_area_pct"], "max_excess_mm": st["max_excess_mm"],
                              "verdict_vs_truth_mismatches": int(((r["within"] == truth["out"]) & ~truth["band"]).sum()),
                              "no_class_points": int((~r["within"] & ~r["excess"] & ~r["gouge"]).sum())}

    t0 = time.time()
    out["p1_new"] = Pr.p1_identity(NEW, n, seed)
    t["p1_s"] = time.time() - t0
    t0 = time.time()
    out["exact_compile"] = {"p8_depth0": Pr.p8_slot(NEW, n, seed, "depth", 0.0), "p7_flat_horiz": Pr.p7_sweep(NEW, "flat-horiz", 100_000, seed)}
    t["exact_first_calls_s"] = time.time() - t0
    out["seconds"] = {k: round(v, 2) for k, v in t.items()}

    lines += [f"Точки двух снимков при seed {seed}: " + ", ".join(f"{k}: {'совпадают' if v else 'НЕ совпадают'}" for k, v in same.items()),
              f"Истина D1 (недорез {d} мм на {face}): аналитическая доля в допуске {out['analytic_within_pct']} %; "
              f"на этой выборке {out['truth_sample']['within_pct_on_sample']} %, вне допуска {out['truth_sample']['out']} точек, "
              f"в полосе EPS {out['truth_sample']['band']}", "",
              f"{'вариант':<18} {'в допуске %':>11} {'недорез %':>10} {'зарез %':>8} {'max недорез':>12} {'расхожд. с истиной':>19} {'без класса':>11}"]
    for k, v in out["variants"].items():
        lines.append(f"{k:<18} {v['within_pct']:>11} {v['excess_area_pct']:>10} {v['gouge_area_pct']:>8} {v['max_excess_mm']:>12} "
                     f"{v['verdict_vs_truth_mismatches']:>19} {v['no_class_points']:>11}")
    e = out["exact_compile"]
    lines += ["", f"P1 на {NEW}: нарушений {out['p1_new']['violations']} (в допуске {out['p1_new']['within_pct']} %, max |dev| {out['p1_new']['max_abs_dev']})",
              f"exact, первые вызовы: паз D3 номинальный — нарушений {e['p8_depth0']['violations']}, в допуске {e['p8_depth0']['within_pct']} %, "
              f"полоса {e['p8_depth0']['band']}; один ход flat-horiz на 10⁵ точках — нарушений {e['p7_flat_horiz']['violations']}, полоса {e['p7_flat_horiz']['band']}",
              "Время, с: " + ", ".join(f"{k} {v}" for k, v in out["seconds"].items())]
    ok = out["p1_new"]["violations"] == 0
    lines += ["", "Условие перехода: нет исключений, P1 проходит, совпадение точек записано — " + ("ПРОЙДЕН" if ok else "НЕ ПРОЙДЕН")]
    return out, lines, ok


# ------------------------------------------------------------------------------------------- короткая проверка
def failed_props(xml: Path) -> tuple[dict, set]:
    root = ET.parse(xml).getroot()
    counts, failed = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}, set()
    for case in root.iter("testcase"):
        counts["tests"] += 1
        kinds = {c.tag for c in case}
        if kinds & {"failure", "error"}:
            counts["failures" if "failure" in kinds else "errors"] += 1
            name = case.get("name", "")
            failed.add(name.split("_")[1] if name.startswith("test_p") else name.split("[")[0])
        elif "skipped" in kinds:
            counts["skipped"] += 1
    return counts, failed


def smoke(prov: dict) -> tuple[dict, list[str], bool]:
    """pytest checks с n = 2·10⁴ на обоих снимках: на 21a0d91 всё зелёное, на 13d20b5 падают ровно ожидаемые свойства."""
    n = 20_000
    out = {"params": {"n": n, "seed": 0}, "runs": {}}
    lines = header("короткая проверка", prov, out["params"])
    for impl in (NEW, OLD):
        xml, log = RESULTS / f"hw2_smoke_{impl}.xml", RESULTS / f"hw2_smoke_{impl}.log"
        t0 = time.time()
        p = subprocess.run([sys.executable, "-m", "pytest", "checks", "-q", "-p", "no:cacheprovider", f"--junitxml={xml}"],
                           cwd=H.ROOT, capture_output=True, text=True, env={**os.environ, "METRIC_IMPL": impl, "HW2_N": str(n), "HW2_SEED": "0"})
        log.write_text(p.stdout + p.stderr)
        counts, failed = failed_props(xml)
        out["runs"][impl] = {"exit": p.returncode, **counts, "failed_props": sorted(failed), "seconds": round(time.time() - t0, 1)}
        lines.append(f"{impl}: код {p.returncode}, тестов {counts['tests']}, упало {counts['failures']}, ошибок {counts['errors']}, "
                     f"пропущено {counts['skipped']}, упавшие свойства {sorted(failed) or '—'} ({out['runs'][impl]['seconds']} с)")
    new, old = out["runs"][NEW], out["runs"][OLD]
    ok = new["exit"] == 0 and set(old["failed_props"]) == EXPECTED_OLD_FAIL
    lines += ["", f"Ожидание: {NEW} — код 0; {OLD} — падают ровно {sorted(EXPECTED_OLD_FAIL)}",
              "Условие перехода — " + ("ПРОЙДЕН" if ok else "НЕ ПРОЙДЕН")]
    return out, lines, ok


# ------------------------------------------------------------------------------------------- перебор
def sweep(prov: dict) -> tuple[dict, list[str], bool]:
    """Все δ × грани × положения и P7–P8 на 21a0d91, n = 10⁵, seed 0: 0 поточечных расхождений."""
    n, seed, impl = 100_000, 0, NEW
    signed = [s * d for s in (1, -1) for d in H.DELTAS]
    cases, t0 = [], time.time()
    for face in H.FACES:
        for d in signed:
            r = Pr.p2_threshold(impl, n, seed, face, d)
            cases.append({"prop": "P2", "face": face, "delta": d, **r})
        for sign in (1, -1):
            r = Pr.p3_monotone(impl, n, seed, face, sign)
            cases.append({"prop": "P3", "face": face, "sign": sign, **r})
        for d in Pr.P6_DELTAS:
            r = Pr.p6_symmetry(impl, n, seed, face, d)
            cases.append({"prop": "P6", "face": face, "delta": d, **r})
        for d in (0.05, 0.11, -0.11):
            for pose in ("shift", "rx", "ry", "rz"):
                r = Pr.p5_invariance(impl, n, seed, face, d, pose)
                cases.append({"prop": "P5", "face": face, "delta": d, "pose": pose, **r})
    for pose in H.POSES:
        r = Pr.p1_identity(impl, n, seed, pose)
        cases.append({"prop": "P1", "pose": pose, **r})
    for case in H.MOVES:
        r = Pr.p7_sweep(impl, case, 1_000_000, seed)
        cases.append({"prop": "P7", "case": case, **r})
    for variant, d in [("depth", 0.0)] + [(v, d) for v in ("depth", "radius") for d in H.DELTAS]:
        r = Pr.p8_slot(impl, n, seed, variant, d)
        cases.append({"prop": "P8", "variant": variant, "delta": d, **r})
    bad = [c for c in cases if c["violations"] > 0]
    out = {"params": {"n": n, "seed": seed, "impl": impl}, "n_cases": len(cases), "n_failed": len(bad),
           "failed": bad, "seconds": round(time.time() - t0, 1)}
    lines = header("перебор", prov, out["params"])
    by = {}
    for c in cases:
        by.setdefault(c["prop"], [0, 0])
        by[c["prop"]][0] += 1
        by[c["prop"]][1] += int(c["violations"] > 0)
    lines += [f"{p}: случаев {n0}, с нарушениями {n1}" for p, (n0, n1) in by.items()]
    if bad:
        lines += ["", "Первые нарушения:"] + [json.dumps(c, ensure_ascii=False) for c in bad[:8]]
    ok = not bad
    lines += ["", f"Время {out['seconds']} с. Условие перехода: 0 поточечных расхождений на {impl} — "
              + ("ПРОЙДЕН" if ok else "НЕ ПРОЙДЕН")]
    return out, lines, ok


# ------------------------------------------------------------------------------------------- полный расчёт
VARIANTS = ((f"{NEW}", NEW, H.CELL), (f"{OLD} cell=0.05", OLD, H.CELL), (f"{OLD} cell=0", OLD, 0.0))
RATIO_LO, RATIO_HI, THRESH = 0.74, 1.26, 0.1  # χ²_29, 95 %; Поправка 6


def _summ(xs: list[float]) -> dict:
    x = np.asarray(xs, float)
    m, s = float(x.mean()), float(x.std(ddof=1) if len(x) > 1 else 0.0)
    half = float(stats.t.ppf(0.975, len(x) - 1) * s / math.sqrt(len(x))) if len(x) > 1 else 0.0
    return {"mean": round(m, 6), "sd": round(s, 6), "ci95": [round(m - half, 6), round(m + half, 6)],
            "values": [round(v, 6) for v in xs]}


def _one_d1(impl: str, face: str, d: float, seed: int, cell: float) -> dict:
    target, inside = H.box_mesh(), H.Box(face, d).inside
    r = H.metric(impl, inside, H.FULL_N, seed, target=target, cell=cell)
    true = H.true_within_d1(face, d)
    ne = H.n_eff(r["w"])
    unrounded = 100.0 * float(r["w"][r["within"]].sum() / r["w"].sum())
    return {"within_pct": r["stats"]["surface_within_tol_pct"], "within_unrounded": unrounded,
            "delta_pp": r["stats"]["surface_within_tol_pct"] - true,
            "delta_unrounded": unrounded - true, "true_pct": true, "n_eff": ne,
            "sigma_binom": 100.0 * math.sqrt(max(true / 100.0 * (1.0 - true / 100.0), 0.0) / ne)}


def _one_d3(seed: int, d: float) -> dict:
    z, samples = H.SLOT_Z - d, H.slot_samples(H.FULL_N, seed)
    r = H.metric(NEW, H.slot_exact(NEW, z, H.SLOT_R).inside, H.FULL_N, seed, samples=samples)
    true = H.true_within_d3_depth(d)
    ne = H.n_eff(r["w"])
    unrounded = 100.0 * float(r["w"][r["within"]].sum() / r["w"].sum())
    return {"within_pct": r["stats"]["surface_within_tol_pct"], "within_unrounded": unrounded,
            "delta_pp": r["stats"]["surface_within_tol_pct"] - true,
            "delta_unrounded": unrounded - true, "true_pct": true, "n_eff": ne,
            "sigma_binom": 100.0 * math.sqrt(max(true / 100.0 * (1.0 - true / 100.0), 0.0) / ne)}


def _case_ok(row: dict) -> tuple[bool, str]:
    """П. 4 и 5 правила (Поправка 5: при p ∈ {0,1} — SD = 0 вместо отношения к σ_бином)."""
    p = row["true_pct"] / 100.0
    ci = row["new"]["ci95"]
    unbias = ci[0] >= -THRESH and ci[1] <= THRESH
    if 0.0 < p < 1.0:
        ratio = row["new"]["sd"] / row["sigma_binom"] if row["sigma_binom"] else math.inf
        spread = RATIO_LO <= ratio <= RATIO_HI
        why = f"CI {ci} внутри ±{THRESH}; SD/σ = {ratio:.3f}"
    else:
        spread = row["new"]["sd"] == 0.0
        why = f"p ∈ {{0,1}}: SD = {row['new']['sd']} (должно 0), CI {ci}"
    return unbias and spread, why


def holm(pvals: list[float]) -> list[float]:
    m, order = len(pvals), np.argsort(pvals)
    adj, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[int(i)])
        adj[int(i)] = min(1.0, running)
    return adj


def full(prov: dict) -> tuple[dict, list[str], bool]:
    """Случаи раздела 5, n = 600 000, seed 0–9; правило принятия п. 4–6."""
    t0 = time.time()
    d1, d3 = [], []
    for face in ("x+", "z+"):
        for d in H.FULL_DELTAS:
            per = {name: [] for name, _, _ in VARIANTS}
            meta = None
            for seed in H.FULL_SEEDS:
                for name, impl, cell in VARIANTS:
                    one = _one_d1(impl, face, d, seed, cell)
                    per[name].append(one)
                    if meta is None:
                        meta = {"true_pct": one["true_pct"], "n_eff": one["n_eff"], "sigma_binom": one["sigma_binom"]}
            row = {"part": "D1", "face": face, "delta_mm": d, **meta,
                   "new": _summ([x["delta_unrounded"] for x in per[f"{NEW}"]]),
                   "new_rounded": _summ([x["delta_pp"] for x in per[f"{NEW}"]]),
                   "old_cell05": _summ([x["delta_unrounded"] for x in per[f"{OLD} cell=0.05"]]),
                   "old_cell0": _summ([x["delta_unrounded"] for x in per[f"{OLD} cell=0"]]),
                   "per_seed": {name: [x["delta_unrounded"] for x in xs] for name, xs in per.items()}}
            ok, why = _case_ok(row)
            row["rule4_5_ok"], row["rule4_5"] = ok, why
            d1.append(row)
    for d in H.FULL_DELTAS:
        xs = [_one_d3(seed, d) for seed in H.FULL_SEEDS]
        row = {"part": "D3", "face": "floor", "delta_mm": d, "true_pct": xs[0]["true_pct"],
               "n_eff": xs[0]["n_eff"], "sigma_binom": xs[0]["sigma_binom"],
               "new": _summ([x["delta_unrounded"] for x in xs]),
               "new_rounded": _summ([x["delta_pp"] for x in xs]),
               "per_seed": {f"{NEW}": [x["delta_unrounded"] for x in xs]}}
        ok, why = _case_ok(row)
        row["rule4_5_ok"], row["rule4_5"] = ok, why
        d3.append(row)

    pairs, pvals = [], []
    for row in d1:
        for old_key, old_name in (("old_cell05", f"{OLD} cell=0.05"), ("old_cell0", f"{OLD} cell=0")):
            a = np.asarray(row["per_seed"][f"{NEW}"], float)
            b = np.asarray(row["per_seed"][old_name], float)
            diff = a - b
            t = stats.ttest_rel(a, b)
            pairs.append({"face": row["face"], "delta_mm": row["delta_mm"], "versus": old_name,
                          "mean_diff": round(float(diff.mean()), 6), "sd_diff": round(float(diff.std(ddof=1)), 6),
                          "p_raw": float(t.pvalue)})
            pvals.append(float(t.pvalue))
    adj = holm(pvals)
    for p, a in zip(pairs, adj):
        p["p_holm"] = round(a, 6)
        p["practical"] = abs(p["mean_diff"]) >= THRESH and a < 0.05

    rule45 = all(r["rule4_5_ok"] for r in d1 + d3)
    out = {"params": {"n": H.FULL_N, "seeds": list(H.FULL_SEEDS), "deltas": list(H.FULL_DELTAS),
                      "threshold_pp": THRESH, "ratio": [RATIO_LO, RATIO_HI]},
           "d1": d1, "d3": d3, "paired": pairs, "seconds": round(time.time() - t0, 1),
           "rule4_5_ok": rule45}
    lines = header("полный расчёт", prov, out["params"])
    lines += [f"{'случай':<28} {'истина %':>9} {'среднее Δ':>10} {'SD':>8} {'95% CI':>24} {'SD/σ':>8} {'п.4–5':>6}"]
    for r in d1 + d3:
        name = f"{r['part']} {r.get('face')} δ={r['delta_mm']}"
        ci = r["new"]["ci95"]
        ratio = r["new"]["sd"] / r["sigma_binom"] if r["sigma_binom"] else float("nan")
        lines.append(f"{name:<28} {r['true_pct']:>9.3f} {r['new']['mean']:>10.4f} {r['new']['sd']:>8.4f} "
                     f"[{ci[0]:+.4f}; {ci[1]:+.4f}] {ratio:>8.3f} {'да' if r['rule4_5_ok'] else 'НЕТ':>6}")
    lines += ["", "Парное сравнение 21a0d91 − 13d20b5 (Холм, α = 0,05; существенно при |Δ| ≥ 0,1 п.п. и p_holm < 0,05):",
              f"{'случай':<22} {'против':<18} {'средняя разность':>16} {'p_holm':>10} {'существенно':>12}"]
    for p in pairs:
        lines.append(f"D1 {p['face']} δ={p['delta_mm']:<6} {p['versus']:<18} {p['mean_diff']:>16.4f} {p['p_holm']:>10.4g} "
                     f"{'да' if p['practical'] else 'нет':>12}")
    lines += ["", f"Время {out['seconds']} с. П. 4–5 на всех случаях 21a0d91 — "
              + ("ПРОЙДЕН" if rule45 else "НЕ ПРОЙДЕН")]
    return out, lines, rule45


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("dry", "smoke", "sweep", "full"))
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args()
    prov = H.provenance()
    if prov["dirty"] and not a.allow_dirty:
        raise SystemExit("Незакоммиченные изменения: " + "; ".join(prov["dirty_files"]) + ". Закоммитьте код до прогона.")
    if not all(s["ok"] for s in prov["snapshots"].values()):
        raise SystemExit("SHA256 снимка не совпадает с SHA256SUMS: " + json.dumps(prov["snapshots"], ensure_ascii=False))
    out, lines, ok = {"dry": dry, "smoke": smoke, "sweep": sweep, "full": full}[a.step](prov)
    write(a.step, {"step": a.step, "provenance": prov, **out, "passed": ok}, lines)
    print("\n".join(lines))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
