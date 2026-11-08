# vendor/camverify_21a0d91 — второй снимок измерительного прибора

| источник | значение |
|---|---|
| репозиторий | закрытый репозиторий автора (харнесс CAM-агента) |
| коммит | `21a0d91` (2026-11-04), HEAD на момент снимка; рабочее дерево `cam_agent/services/` чистое |
| путь в репозитории | `cam_agent/services/` |
| файлы | `exact.py`, `verify.py`, `gcode.py`, `dexel.py`, `raster.py`, `voxel.py`, `cpu_fast.py` — `git show 21a0d91:cam_agent/services/<файл>` |
| свой файл | `camverify_21a0d91/__init__.py` (вместо пакетного: тот подключает FreeCAD) |
| контрольные суммы | `SHA256SUMS_21a0d91`; проверка: `cd vendor/camverify_21a0d91 && sha256sum -c ../SHA256SUMS_21a0d91` |
| зависимости | numpy, scipy, numba |

Подключение: `sys.path.insert(0, "vendor")`, затем `from camverify_21a0d91 import verify, exact`.
Отличия от `camverify/` (`13d20b5`), важные для ДЗ 2: метрика `verify.surface_metric` берёт готовые точки и не
сдвигает пробы на ±cell/2; классы недореза и зареза — `>= tol`; модель материала — `exact` (заготовка и заметаемые
объёмы ходов без сетки). Снимок только читается; выводы проекта относятся к этому коммиту.
