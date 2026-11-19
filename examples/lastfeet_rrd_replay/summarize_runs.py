#!/usr/bin/env python3
"""Tabulate the .json files written by replay_rrd_compare.py: robot build ("baseline") vs the build under test ("main").

A run is a baseline run if its label contains "baseline", a main run if it contains "main" or "sba0" (the default
patch settings: gravity freeze on, SBA init off). Runs with other labels (e.g. "sba1") and overlay files are left out.
Repeats of the same build on the same recording are averaged.

Labels containing "-seg" were run with seg_load.py holding the GPU next to the replay ("-segbatch": a lighter,
batched load). They are kept apart from the runs made alone, and a second section compares the conditions per build,
together with the loader's own throughput (<label>.seg.json) and the GPU / CPU load tegrastats logged
(<label>.tegra.log).

    python summarize_runs.py                 # results/ next to this script
    python summarize_runs.py path/to/results
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def build_of(label):
    if "baseline" in label:
        return "baseline"
    if "main" in label or "sba0" in label:
        return "main"
    return None


def condition_of(label):
    """alone | seg (seg_load.py back to back) | segbatch (one 4-image pass every 200 ms) | segbatch20 (same, replay
    paced at 20 fps; "frames > frame period" then counts frames over 50 ms) | svc<N>[@20] (the real segformer service
    fed from the recording by feed_rgb_from_rrd.py, with --batch-timeout-ms N; "-20fps" = replay paced at 20 fps)."""
    m = re.search(r"-segsvc(\d+)(-20fps)?", label)
    if m:
        return f"svc{m.group(1)}" + ("@20" if m.group(2) else "")
    if "-segbatch20" in label:
        return "segbatch20"
    if "-segbatch" in label:
        return "segbatch"
    return "seg" if "-seg" in label else "alone"


def load(results, condition="alone"):
    runs = defaultdict(lambda: defaultdict(list))  # recording -> build -> [metrics]
    for f in sorted(results.glob("*.json")):
        if "__overlay" in f.name or f.name.endswith((".seg.json", ".feed.json")):
            continue
        d = json.loads(f.read_text())
        if len(d.get("runs", [])) != 1:
            continue
        m = d["runs"][0]
        b = build_of(m["label"])
        if b is None or "track_ms" not in m or condition_of(m["label"]) != condition:
            continue
        m["_file"] = f.name
        runs[d["recording"]][b].append(m)
    return runs


def tegra_summary(results, label):
    """Mean / p95 GPU utilisation (GR3D_FREQ) and mean CPU utilisation from <label>.tegra.log, or None."""
    f = results / f"{label}.tegra.log"
    if not f.exists():
        return None
    gpu, cpu = [], []
    for line in f.read_text(errors="replace").splitlines():
        g = re.search(r"GR3D_FREQ (\d+)%", line)
        c = re.search(r"CPU \[([^\]]+)\]", line)
        if g:
            gpu.append(int(g.group(1)))
        if c:
            cores = [int(x.split("%")[0]) for x in c.group(1).split(",") if "%" in x]
            if cores:
                cpu.append(np.mean(cores))
    if not gpu:
        return None
    return {"gpu_mean": float(np.mean(gpu)), "gpu_p95": float(np.percentile(gpu, 95)),
            "cpu_mean": float(np.mean(cpu)) if cpu else float("nan")}


def seg_summary(results, label):
    f = results / f"{label}.seg.json"
    return json.loads(f.read_text()) if f.exists() else None


def avg(ms, getter):
    v = [getter(m) for m in ms]
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else float("nan")


ROWS = [
    ("track() mean ms", lambda m: m["track_ms"]["mean"]),
    ("track() p95 ms", lambda m: m["track_ms"]["p95"]),
    ("track() p99 ms", lambda m: m["track_ms"]["p99"]),
    ("track() max ms", lambda m: m["track_ms"]["max"]),
    ("frames > frame period", lambda m: m["track_over_frame_period"]),
    ("trend last/first third", lambda m: m["track_ramp_last_over_first_third"]),
    ("lag max ms", lambda m: m.get("lag_ms", {}).get("max")),
    ("gravity tick", lambda m: m["gravity_first_frame"]),
    ("gravity tilt deg", lambda m: m.get("gravity_tilt_deg", {}).get("median")),
    ("tracked / ticks", lambda m: m["tracked_ticks"] / m["ticks"]),
    ("end->start slam m", lambda m: m["end_to_start_m"]["slam"]),
    ("loop closures", lambda m: m["loop_closures"]),
    ("path length m", lambda m: m["path_len_m"]["slam"]),
]


def main():
    results = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "results"
    runs = load(results)
    both = {r: b for r, b in runs.items() if "baseline" in b and "main" in b}
    if not both:
        sys.exit(f"no recording in {results} has both baseline and main runs")

    print(f"{'recording':34s} {'build':8s} {'n':>2s} " + " ".join(f"{n[:14]:>14s}" for n, _ in ROWS[:6]))
    for rec in sorted(both):
        for b in ("baseline", "main"):
            ms = both[rec][b]
            print(f"{rec[:34]:34s} {b:8s} {len(ms):>2d} " + " ".join(f"{avg(ms, g):14.2f}" for _, g in ROWS[:6]))
    print()
    print("pooled over recordings (each recording weighted equally; repeats averaged first)")
    print(f"{'':26s} {'baseline':>10s} {'main':>10s} {'main vs baseline':>18s}")
    wins = {}
    for name, g in ROWS:
        per = {b: [avg(both[r][b], g) for r in sorted(both)] for b in ("baseline", "main")}
        a, m = float(np.nanmean(per["baseline"])), float(np.nanmean(per["main"]))
        rel = (m / a - 1) * 100 if a else float("nan")
        print(f"{name:26s} {a:10.2f} {m:10.2f} {rel:+17.0f}%")
        wins[name] = sum(1 for x, y in zip(per["baseline"], per["main"]) if y < x)
    n = len(both)
    print()
    print(f"recordings where main is lower than baseline (out of {n}):")
    for name in ("track() mean ms", "track() p95 ms", "track() p99 ms", "track() max ms", "frames > frame period"):
        print(f"  {name:24s} {wins[name]}/{n}")

    contention(results)


CONTENTION_ROWS = [r for r in ROWS if r[0] in (
    "track() mean ms", "track() p95 ms", "track() p99 ms", "track() max ms", "frames > frame period",
    "lag max ms", "gravity tick", "end->start slam m", "loop closures")]


BASE_CONDS = ("alone", "seg", "segbatch", "segbatch20")


def cond_title(c):
    fixed = {"alone": "alone", "seg": "seg continuous", "segbatch": "seg batched", "segbatch20": "batched @20fps"}
    if c in fixed:
        return fixed[c]
    m = re.fullmatch(r"svc(\d+)(@20)?", c)
    if m:
        return f"svc t={m.group(1)}" + (" @20" if m.group(2) else "")
    return c


def cond_sort_key(c):
    if c in BASE_CONDS:
        return (0, BASE_CONDS.index(c), 0, 0)
    m = re.fullmatch(r"svc(\d+)(@20)?", c)
    return (1, 1 if m and m.group(2) else 0, int(m.group(1)) if m else 0, 0)


def feed_summary(results, label):
    f = results / f"{label}.feed.json"
    return json.loads(f.read_text()) if f.exists() else None


def contention(results):
    present_conds = set()
    for f in results.glob("*.json"):
        if "__overlay" in f.name or f.name.endswith((".seg.json", ".feed.json")):
            continue
        try:
            d = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        for m in d.get("runs", []) if len(d.get("runs", [])) == 1 else []:
            if build_of(m["label"]):
                present_conds.add(condition_of(m["label"]))
    order = sorted(present_conds | {"alone"}, key=cond_sort_key)
    conds = {c: load(results, c) for c in order}
    recs = sorted({r for c in order if c != "alone" for r in conds[c] if r in conds["alone"]})
    if not recs:
        return
    print()
    print("=" * 110)
    print("GPU contention: replay alone vs with a segmenter load next to it")
    print("  seg continuous = seg_load.py back-to-back passes (live-like, ~95% GPU);  seg batched = one 4-image pass per 200 ms;")
    print("  batched @20fps = same, replay paced at 20 fps (frame budget 50 ms, the others 67 ms);")
    print("  svc t=N = the real segformer service fed from the recording, --batch-timeout-ms N (0 = per-arrival, today)")
    for rec in recs:
        print(f"\n{rec}")
        for b in ("baseline", "main"):
            present = [c for c in order if b in conds[c].get(rec, {})]
            if "alone" not in present or len(present) < 2:
                continue
            print(f"  {b}   (runs: " + ", ".join(f"{cond_title(c)} n={len(conds[c][rec][b])}" for c in present) + ")")
            print(f"    {'':24s} " + " ".join(f"{cond_title(c):>15s}" for c in present)
                  + "".join(f" {'vs alone':>9s}" for c in present[1:]))
            for name, g in CONTENTION_ROWS:
                vals = [avg(conds[c][rec][b], g) for c in present]
                rels = [(v / vals[0] - 1) * 100 if vals[0] else float("nan") for v in vals[1:]]
                print(f"    {name:24s} " + " ".join(f"{v:15.2f}" for v in vals) + "".join(f" {r:+8.0f}%" for r in rels))
            for c in present[1:]:
                runs_c = conds[c][rec][b]
                segs = [x for x in (seg_summary(results, m["label"]) for m in runs_c) if x]
                teg = [x for x in (tegra_summary(results, m["label"]) for m in runs_c) if x]
                line = f"    [{cond_title(c)}] "
                if segs:
                    line += (f"loader {np.mean([x['passes_per_s'] for x in segs]):.1f} passes/s, pass latency mean "
                             f"{np.mean([x['latency_ms']['mean'] for x in segs]):.1f} p95 "
                             f"{np.mean([x['latency_ms']['p95'] for x in segs]):.1f} max "
                             f"{np.mean([x['latency_ms']['max'] for x in segs]):.1f} ms; ")
                if teg:
                    line += (f"GPU {np.mean([x['gpu_mean'] for x in teg]):.0f}% mean, "
                             f"CPU {np.mean([x['cpu_mean'] for x in teg]):.0f}% mean")
                if segs or teg:
                    print(line)
                feeds = [x for x in (feed_summary(results, m["label"]) for m in runs_c) if x]
                for fd in feeds:
                    age = fd.get("mask_age_s") or {}
                    print(f"    [{cond_title(c)}] segmenter service: {fd['masks_per_s']:.1f} masks/s, "
                          f"{fd['cycles_per_s']:.1f} passes/s, {fd['masks_per_cycle']:.2f} masks/pass, "
                          f"frames never segmented {fd['lost_pct_overall']:.1f}%, mask age mean "
                          f"{1e3 * age.get('mean', float('nan')):.0f} p95 {1e3 * age.get('p95', float('nan')):.0f} "
                          f"max {1e3 * age.get('max', float('nan')):.0f} ms")
        for c in order[1:]:
            if all(b in conds[c].get(rec, {}) for b in ("baseline", "main")):
                print(f"  main vs baseline, {cond_title(c)}:")
                for name, g in CONTENTION_ROWS[:5]:
                    x, y = avg(conds[c][rec]["baseline"], g), avg(conds[c][rec]["main"], g)
                    print(f"    {name:24s} baseline {x:8.2f}  main {y:8.2f}  "
                          f"{((y / x - 1) * 100 if x else float('nan')):+5.0f}%")


if __name__ == "__main__":
    main()
