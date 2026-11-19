#!/usr/bin/env python3
"""Hold a SegFormer load on the GPU: the real TRTSegmenter on a fixed set of images, no cameras, no ZMQ.

This is the segmenter service's inference call (`TRTSegmenter.infer_class_ids_checked`: GPU resize + normalise, one
TensorRT pass, argmax, copy back) in a loop, without the camera streamers, JPEG decoding, mask encoding or
publishing. It is the lowest-overhead version of the live segmenter, to be run next to `replay_rrd_compare.py` to
see what GPU contention does to cuvslam.

How hard to run it. The live service loops "poll, segment any camera with a fresh frame, publish". Each pass costs
the same (one batch-4 engine pass, ~61 ms on the Orin) whether it carries one frame or four, and frames arrive at
~20/s across the four cameras, so a new frame is always waiting when a pass ends: live, the loop never idles and
holds the GPU ~85-90% of the time. `--hz 0` (the default) reproduces that: back-to-back passes. `--hz N` caps the
rate for a lighter load, e.g. --hz 8 is about half the GPU time.

    python seg_load.py --rrd <recording.rrd> --duration 240

Prints one line per --log-every seconds and a final `SEG_SUMMARY {json}` line. `--ready-file` is created once the
engine is loaded and warm, so a runner knows when to start the replay.

Needs an env with torch + tensorrt + the segformer_segmenter package (the lastfeet envs have it) and rerun >= 0.35
if --rrd is used to take the fixed images from a recording (without it, random-noise images; the cost of a pass does
not depend on the image content).
"""

import argparse
import json
import signal
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)

DEFAULT_MONOREPO = Path.home() / "code" / "lastfeet-monorepo"
DEFAULT_ENGINE = "models/seg-models/model_fp16_batch4_768x768_72a1428.trt"
CAMERAS = ["front_forward", "left", "right", "rear_backward"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--monorepo", type=Path, default=DEFAULT_MONOREPO)
    p.add_argument("--engine", type=Path, default=None, help=f"default <monorepo>/{DEFAULT_ENGINE}")
    p.add_argument("--rrd", type=Path, default=None, help="recording to take one RGB frame per camera from")
    p.add_argument("--frames-per-pass", type=int, default=4,
                   help="images per engine pass (the live service carries ~1.3 on average; the cost is the same)")
    p.add_argument("--hz", type=float, default=0.0, help="engine passes per second; 0 = back-to-back")
    p.add_argument("--duration", type=float, default=0.0, help="seconds to run; 0 = until killed")
    p.add_argument("--log-every", type=float, default=10.0)
    p.add_argument("--ready-file", type=Path, default=None)
    p.add_argument("--summary-json", type=Path, default=None)
    return p.parse_args()


def frames_from_rrd(path, n):
    import cv2
    try:
        from rerun.chunk import RrdReader
    except ImportError:
        from rerun.experimental import RrdReader
    got = {}
    for ch in RrdReader(str(path)).stream():
        ep = str(ch.entity_path)
        for cam in CAMERAS:
            if ep == f"/robot/cameras/{cam}/cam_rgb/img" and ch.num_rows and cam not in got:
                rb = ch.to_record_batch()
                if "EncodedImage:blob" not in rb.schema.names:
                    continue
                blob = rb.column("EncodedImage:blob")[0].values.flatten().to_numpy(zero_copy_only=False)
                bgr = cv2.imdecode(np.frombuffer(blob.tobytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
                if bgr is not None:
                    got[cam] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if len(got) == len(CAMERAS):
            break
    frames = [got[c] for c in CAMERAS if c in got]
    if not frames:
        sys.exit(f"no RGB frames found in {path}")
    return (frames * n)[:n]


def main():
    args = parse_args()
    sys.path.insert(0, str(args.monorepo / "python" / "segformer-segmenter"))
    from segformer_segmenter.engine import TRTSegmenter

    engine = args.engine or args.monorepo / DEFAULT_ENGINE
    seg = TRTSegmenter(str(engine))
    if args.rrd:
        feed = frames_from_rrd(args.rrd, args.frames_per_pass)
        print(f"fixed images: {len(feed)} from {args.rrd.name}, shape {feed[0].shape}", flush=True)
    else:
        rng = np.random.default_rng(0)
        feed = [rng.integers(0, 255, (400, 640, 3), dtype=np.uint8) for _ in range(args.frames_per_pass)]
        print(f"fixed images: {len(feed)} random 640x400", flush=True)

    for _ in range(10):  # warm-up: CUDA context, TensorRT autotune caches, allocator
        seg.infer_class_ids_checked(feed)
    if args.ready_file:
        args.ready_file.write_text("ready\n")
    print(f"engine {engine.name}, batch {seg.batch_size}, hz={'max' if args.hz <= 0 else args.hz}", flush=True)

    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))

    period = 1.0 / args.hz if args.hz > 0 else 0.0
    lat, late = [], 0
    t_start = time.monotonic()
    t_log, n_log, next_t = t_start, 0, t_start
    while not stop["flag"]:
        t0 = time.perf_counter()
        seg.infer_class_ids_checked(feed)
        lat.append((time.perf_counter() - t0) * 1e3)
        n_log += 1
        now = time.monotonic()
        if period:
            next_t += period
            wait = next_t - now
            if wait > 0:
                time.sleep(wait)
            else:
                late += 1
                if wait < -period:
                    next_t = now  # fell far behind: do not try to catch up with a burst
        if now - t_log >= args.log_every:
            recent = lat[-n_log:]
            print(f"seg: {n_log / (now - t_log):5.1f} passes/s  latency mean {np.mean(recent):5.1f} "
                  f"p95 {np.percentile(recent, 95):5.1f} max {np.max(recent):5.1f} ms", flush=True)
            t_log, n_log = now, 0
        if args.duration and now - t_start >= args.duration:
            break

    elapsed = time.monotonic() - t_start
    a = np.asarray(lat)
    summary = {
        "passes": int(len(a)), "elapsed_s": elapsed, "passes_per_s": len(a) / elapsed,
        "latency_ms": {"mean": float(a.mean()), "p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)),
                       "p99": float(np.percentile(a, 99)), "max": float(a.max())},
        "late_passes": late, "hz": args.hz, "frames_per_pass": args.frames_per_pass,
    }
    print("SEG_SUMMARY " + json.dumps(summary), flush=True)
    if args.summary_json:
        args.summary_json.write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
