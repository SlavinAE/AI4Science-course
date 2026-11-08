"""Независимая сверка пробного прогона ДЗ 2 (results/hw2_dry.json): без hw2_parts / hw2_props, только снимки и формулы.

  .venv/bin/python src/hw2_dry_check.py

1. Сколько точек выборки (seed 0, n = 10⁴) лежит на грани x+ и согласуется ли это с площадью грани (биномиальный z).
2. Вердикты снимков, вызванных напрямую, на блоке с гранью x+, выдвинутой на 0,11 мм: какие точки вне допуска,
   какие без класса — ровно точки грани x+ или нет.
3. Совпадают ли точки двух снимков при одном seed.
4. Числа в results/hw2_dry.json совпадают с пересчётом.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
from camverify import verify as old  # noqa: E402
from camverify_21a0d91 import verify as new  # noqa: E402

N, SEED, TOL, D = 10_000, 0, 0.1, 0.11
V = np.array([[0, 0, 0], [100, 0, 0], [100, 60, 0], [0, 60, 0], [0, 0, 20], [100, 0, 20], [100, 60, 20], [0, 60, 20]], float)
F = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6], [1, 2, 6], [1, 6, 5],
              [0, 4, 7], [0, 7, 3]], np.int64)
target = {"vertices": V, "faces": F}


def inside(P):  # блок, грань x = 100 выдвинута на D
    return (P[:, 0] >= 0) & (P[:, 0] <= 100 + D) & (P[:, 1] >= 0) & (P[:, 1] <= 60) & (P[:, 2] >= 0) & (P[:, 2] <= 20)


lines, ok = [], True


def check(name, cond, detail):
    global ok
    ok &= bool(cond)
    lines.append(f"[{'OK' if cond else 'НЕТ'}] {name}: {detail}")


# 1. точки на грани x+
s = new.mesh_samples(V, F, N, SEED)
P = s["P"]
on = np.abs(P[:, 0] - 100) < 1e-9
k, p = int(on.sum()), 1200 / 18400
mu, sd = N * p, math.sqrt(N * p * (1 - p))
z = (k - mu) / sd
check("точки на грани x+", abs(z) < 3, f"{k} из {N}; ожидание {mu:.0f} ± {sd:.1f} (площадь 1200/18 400), z = {z:+.2f}")

# 2. вердикты снимков напрямую
rn = new.surface_metric(s, inside, TOL)
check("21a0d91: вне допуска ровно точки грани x+", np.array_equal(~rn["within"], on),
      f"вне допуска {int((~rn['within']).sum())}, из них на грани {int((~rn['within'] & on).sum())}; в допуске {rn['stats']['surface_within_tol_pct']} %")
ro5 = old.surface_deviation(target, inside, TOL, N, seed=SEED, cell=0.05)
check("13d20b5 cell=0,05: все точки засчитаны в допуск (стенка не поймана)", bool(ro5["within"].all()),
      f"в допуске {ro5['stats']['surface_within_tol_pct']} %, max недорез {ro5['stats']['max_excess_mm']} мм")
ro0 = old.surface_deviation(target, inside, TOL, N, seed=SEED, cell=0.0)
noclass = ~ro0["within"] & ~(ro0["dev"] > TOL) & ~(ro0["dev"] < -TOL)
check("13d20b5 cell=0: точки без класса — ровно точки грани x+", np.array_equal(noclass, on),
      f"без класса {int(noclass.sum())}; в допуске {ro0['stats']['surface_within_tol_pct']} %, недорез {ro0['stats']['excess_area_pct']} %, "
      f"сумма классов {ro0['stats']['surface_within_tol_pct'] + ro0['stats']['excess_area_pct'] + ro0['stats']['gouge_area_pct']:.2f} %")

# 3. одинаковые точки
same = all(np.array_equal(P, r["points"]) and np.array_equal(s["N"], r["normals"]) for r in (ro5, ro0))
check("точки двух снимков при seed 0 совпадают", same, "P и N побитово равны" if same else "различаются")

# 4. сверка с results/hw2_dry.json
j = json.loads((ROOT / "results" / "hw2_dry.json").read_text())
v = j["variants"]
rec = {"21a0d91": (rn["stats"]["surface_within_tol_pct"], 0, 0), "13d20b5 cell=0.05": (ro5["stats"]["surface_within_tol_pct"], k, 0),
       "13d20b5 cell=0": (ro0["stats"]["surface_within_tol_pct"], 0, k)}
got = {name: (v[name]["within_pct"], v[name]["verdict_vs_truth_mismatches"], v[name]["no_class_points"]) for name in rec}
check("числа results/hw2_dry.json совпадают с пересчётом", got == rec, f"коммит прогона {j['provenance']['commit'][:7]}; {got}")

print("\n".join(lines))
print("ИТОГ: всё сходится" if ok else "ИТОГ: есть расхождения")
sys.exit(0 if ok else 1)
