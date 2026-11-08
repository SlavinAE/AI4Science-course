"""ДЗ 2: шаги глубины (METHOD.md, раздел 7). Каждый шаг пишет results/hw2_<шаг>.json и .txt с коммитом проекта,
SHA256 снимков, seed и n и печатает «ПРОЙДЕН» или завершается кодом 1.

  .venv/bin/python src/hw2_experiment.py dry     пробный прогон
  .venv/bin/python src/hw2_experiment.py smoke   короткая проверка: pytest checks на обоих снимках

Шаг не запускается, если в src/, checks/, vendor/, METHOD.md или AGENT.md есть незакоммиченные изменения
(--allow-dirty — только для отладки, такой результат в отчёт не идёт).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("dry", "smoke"))
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args()
    prov = H.provenance()
    if prov["dirty"] and not a.allow_dirty:
        raise SystemExit("Незакоммиченные изменения: " + "; ".join(prov["dirty_files"]) + ". Закоммитьте код до прогона.")
    if not all(s["ok"] for s in prov["snapshots"].values()):
        raise SystemExit("SHA256 снимка не совпадает с SHA256SUMS: " + json.dumps(prov["snapshots"], ensure_ascii=False))
    out, lines, ok = {"dry": dry, "smoke": smoke}[a.step](prov)
    write(a.step, {"step": a.step, "provenance": prov, **out, "passed": ok}, lines)
    print("\n".join(lines))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
