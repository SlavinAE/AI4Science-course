"""ISO / Fanuc / LinuxCNC G-code reader for simulation, verification and cycle-time estimates.

Handles: block numbers (N), comments ( ) and ;, modal motion (G0/G1/G2/G3 without repeating the G word), absolute and
incremental distances (G90/G91), mm and inch (G21/G20), arcs with I/J/K or R (XY plane; other planes are linearised
with a warning), helical arcs, canned cycles G73/G74/G76/G81-G89 with G98/G99 retract and modal repeats (X/Y only
lines), LinuxCNC rigid tapping G33.1, tool changes (T.. M6) and dwell. Operations are delimited by the
"(begin operation: NAME)" comments of FreeCAD posts; without them every tool change starts a new operation.
"""
from __future__ import annotations

import math
import re

_WORD = re.compile(r"([A-Z])\s*([-+]?(?:\d+\.?\d*|\.\d+))")
_OP = re.compile(r"\(\s*begin operation\s*:\s*(.*)\)\s*$", re.I)  # labels may contain parentheses
CYCLES = {"G73", "G74", "G76", "G81", "G82", "G83", "G84", "G85", "G86", "G87", "G88", "G89"}


def _code(v: float) -> str:
    return f"G{int(v)}" if abs(v - int(v)) < 1e-9 else f"G{v:g}"


def parse(path_or_text: str, arc_tol: float = 0.01, initial_z: float | None = None) -> dict:
    """-> {"ops": [{"name", "tool", "moves": [(x0,y0,z0,x1,y1,z1,cut,feed)], "lines", "cycles": [...]}], "warnings", "tool_changes",
          "units"}. feed in mm/min (None for rapids)."""
    text = path_or_text
    if "\n" not in path_or_text and len(path_or_text) < 4096:
        with open(path_or_text, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    ops: list[dict] = []
    cur = None
    warnings: list[str] = []
    x = y = 0.0
    z = initial_z
    motion, absolute, scale, plane, retract98 = "G0", True, 1.0, "G17", True
    feed = None
    tool = None
    tool_changes = 0
    cycle_init_z = None
    r_plane = None
    pending_tool = None

    def op():
        nonlocal cur
        if cur is None:
            cur = {"name": f"T{tool}" if tool is not None else "program", "tool": tool, "moves": [], "lines": 0, "cycles": []}
            ops.append(cur)
        return cur

    def add(x0, y0, z0, x1, y1, z1, cut):
        if z0 is None:
            z0 = z1
        o = op()
        o["moves"].append((x0, y0, z0, x1, y1, z1, cut, feed if cut else None))

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _OP.search(line)
        if m:
            cur = {"name": m.group(1).strip(), "tool": tool, "moves": [], "lines": 0, "cycles": [], "explicit": True}
            ops.append(cur)
            continue
        code = re.sub(r"\(.*?\)", " ", line.split(";")[0]).upper()
        if not code.strip() or code.strip().startswith("%"):
            continue
        words = [(k, float(v)) for k, v in _WORD.findall(code)]
        if not words:
            continue
        gs = [_code(v) for k, v in words if k == "G"]
        ms = [int(v) for k, v in words if k == "M"]
        p = {}
        for k, v in words:
            if k not in ("G", "M", "N"):
                p[k] = v
        for g in gs:
            if g in ("G20",):
                scale = 25.4
            elif g == "G21":
                scale = 1.0
            elif g == "G90":
                absolute = True
            elif g == "G91":
                absolute = False
            elif g in ("G17", "G18", "G19"):
                plane = g
            elif g == "G98":
                retract98 = True
            elif g == "G99":
                retract98 = False
            elif g == "G80":
                motion = "G0" if motion in CYCLES else motion
            elif g in ("G0", "G00", "G1", "G01", "G2", "G02", "G3", "G03", "G33.1") or g in CYCLES:
                motion = {"G00": "G0", "G01": "G1", "G02": "G2", "G03": "G3"}.get(g, g)
                if g in CYCLES:
                    cycle_init_z = z
        if "T" in p:
            pending_tool = int(p["T"])
        if 6 in ms and pending_tool is not None:
            tool = pending_tool
            tool_changes += 1
            if cur is not None and not cur.get("explicit") and cur["moves"]:
                cur = None  # no operation comments in this program: a tool change starts the next operation
            elif cur is not None:
                cur["tool"] = tool
        if "F" in p:
            feed = p["F"] * scale
        has_xyz = any(k in p for k in ("X", "Y", "Z"))
        if not has_xyz and motion not in CYCLES:
            continue
        o = op()
        o["lines"] += 1

        def target(axis, current):
            if axis not in p:
                return current
            val = p[axis] * scale
            if current is None:
                return val
            return val if absolute else current + val

        if motion in ("G0", "G1"):
            nx, ny, nz = target("X", x), target("Y", y), target("Z", z)
            if z is not None and nz is not None:  # before the first Z word the height is unknown: position only
                add(x, y, z, nx, ny, nz, motion == "G1")
            x, y, z = nx, ny, nz
        elif motion in ("G2", "G3"):
            nx, ny, nz = target("X", x), target("Y", y), target("Z", z)
            if z is None and nz is None:
                warnings.append("arc before the first Z word ignored")
                x, y = nx, ny
                continue
            z0 = z if z is not None else nz
            if plane != "G17":
                warnings.append(f"arc in {plane} linearised")
                add(x, y, z0, nx, ny, nz, True)
            else:
                if "R" in p:
                    r = p["R"] * scale
                    dx, dy = nx - x, ny - y
                    q = math.hypot(dx, dy)
                    if q < 1e-9 or abs(r) < q / 2 - 1e-6:
                        warnings.append("invalid R arc linearised")
                        add(x, y, z0, nx, ny, nz, True)
                        x, y, z = nx, ny, nz
                        continue
                    h = math.sqrt(max(r * r - q * q / 4, 0.0))
                    sgn = (1 if motion == "G3" else -1) * (1 if r > 0 else -1)
                    cx, cy = x + dx / 2 - sgn * h * dy / q, y + dy / 2 + sgn * h * dx / q
                else:
                    cx, cy = x + p.get("I", 0.0) * scale, y + p.get("J", 0.0) * scale
                rad = math.hypot(x - cx, y - cy)
                a0, a1 = math.atan2(y - cy, x - cx), math.atan2(ny - cy, nx - cx)
                if motion == "G2":
                    if a1 >= a0 - 1e-9:
                        a1 -= 2 * math.pi
                else:
                    if a1 <= a0 + 1e-9:
                        a1 += 2 * math.pi
                sweep = abs(a1 - a0)
                n = max(2, int(math.ceil(sweep / max(2 * math.acos(max(-1.0, 1 - arc_tol / max(rad, 1e-6))), 1e-3))))
                px, py, pz = x, y, z0
                for k in range(1, n + 1):
                    a = a0 + (a1 - a0) * k / n
                    qx, qy, qz = (cx + rad * math.cos(a), cy + rad * math.sin(a), z0 + (nz - z0) * k / n) if k < n else (nx, ny, nz)
                    add(px, py, pz, qx, qy, qz, True)
                    px, py, pz = qx, qy, qz
            x, y, z = nx, ny, nz
        elif motion == "G33.1":  # LinuxCNC rigid tapping: feed to Z at pitch K, reverse, feed back
            nz = target("Z", z)
            add(x, y, z, x, y, nz, True)
            add(x, y, nz, x, y, z, True)
            o["cycles"].append({"cycle": "G33.1", "x": x, "y": y, "z": nz, "pitch": p.get("K", 0.0) * scale})
        elif motion in CYCLES:
            nx, ny = target("X", x), target("Y", y)
            if "Q" in p:
                o["cycle_q"] = p["Q"] * scale
            if "R" in p:
                r_plane = p["R"] * scale if absolute else (z or 0.0) + p["R"] * scale
            if "Z" in p:
                o["cycle_z"] = p["Z"] * scale if absolute else (r_plane or 0.0) + p["Z"] * scale
            zb = o.get("cycle_z")
            if zb is None or r_plane is None:
                warnings.append(f"{motion} without R/Z ignored")
                x, y = nx, ny
                continue
            start = z if z is not None else r_plane
            add(x, y, start, nx, ny, start, False)
            if start > r_plane:
                add(nx, ny, start, nx, ny, r_plane, False)
            add(nx, ny, r_plane, nx, ny, zb, True)
            back_feed = motion in ("G74", "G84", "G85", "G89")  # tapping and reaming feed out
            ret = (cycle_init_z if retract98 and cycle_init_z is not None else r_plane)
            add(nx, ny, zb, nx, ny, max(r_plane, ret if ret is not None else r_plane), back_feed)
            o["cycles"].append({"cycle": motion, "x": nx, "y": ny, "z": zb, "r": r_plane, "q": o.get("cycle_q"), "f": feed})
            x, y = nx, ny
            z = max(r_plane, ret if ret is not None else r_plane)
    ops = [o for o in ops if o["moves"]]
    return {"ops": ops, "warnings": sorted(set(warnings)), "tool_changes": tool_changes, "units": "mm"}


def cycle_time_min(parsed: dict, rapid_mm_min: float = 24000.0, tool_change_s: float = 5.0) -> dict:
    feed_min = rapid_min = 0.0
    feed_len = rapid_len = 0.0
    for o in parsed["ops"]:
        for x0, y0, z0, x1, y1, z1, cut, f in o["moves"]:
            L = math.dist((x0, y0, z0), (x1, y1, z1))
            if cut and f:
                feed_len += L
                feed_min += L / max(f, 1e-6)
            else:
                rapid_len += L
                rapid_min += L / rapid_mm_min
    tc = parsed.get("tool_changes", 0) * tool_change_s / 60.0
    return {"cutting_min": round(feed_min, 2), "rapid_min": round(rapid_min, 2), "tool_change_min": round(tc, 2),
            "total_min": round(feed_min + rapid_min + tc, 2), "feed_length_mm": round(feed_len, 1), "rapid_length_mm": round(rapid_len, 1)}


def rapids_in_material(parsed: dict, top_of_stock: float, tol: float = 0.05) -> list[dict]:
    """G0 moves with an XY component below the top of the stock (possible crash unless that area is already cleared)."""
    out = []
    for o in parsed["ops"]:
        n = sum(1 for x0, y0, z0, x1, y1, z1, cut, _ in o["moves"] if not cut and min(z0, z1) < top_of_stock - tol and math.hypot(x1 - x0, y1 - y0) > 1e-6)
        if n:
            out.append({"op": o["name"], "count": n})
    return out
