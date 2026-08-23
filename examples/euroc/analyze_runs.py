# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA software released under the NVIDIA Community License is intended to be used to enable
# the further development of AI and robotics technologies. Such software has been designed, tested,
# and optimized for use with NVIDIA hardware, and this License grants permission to use the software
# solely with such hardware.
# Subject to the terms of this License, NVIDIA confirms that you are free to commercially use,
# modify, and distribute the software with NVIDIA hardware. NVIDIA does not claim ownership of any
# outputs generated using the software or derivative works thereof. Any code contributions that you
# share with NVIDIA are licensed to NVIDIA as feedback under this License and may be incorporated
# in future releases without notice or attribution.
# By using, reproducing, modifying, distributing, performing, or displaying any portion or element
# of the software or derivative works thereof, you agree to be bound by this License.

"""Statistics over per-frame CSVs written by track_euroc_mod.py.

Usage:
    python3 analyze_runs.py [run.csv ...]     # defaults to every CSV in ./dataset/runs

The headline metric is per-frame *error injection*:

    |(pos[i] - pos[i-1]) - (gt[i] - gt[i-1])|

how far the odometry's frame-to-frame displacement differs from ground truth's. It is
motion-invariant -- zero whenever odometry tracks the true motion, at any speed. That matters
because the platform is moving for most of MH_01, so a raw position-step threshold cannot
separate "drifted 40 mm" from "genuinely travelled 40 mm".

Each blackout event is split into two numbers:

    during  - injected while the images are blacked out (IMU coasting; expected, small)
    after   - injected once vision returns (the recovery transient)

The split is what tells you which mechanism a change affected. Anchoring IMU propagation on
prev_pose instead of on the newest keyframe removes the jump at blackout entry but leaves
`after` untouched, because that one comes from the map being rebuilt out of a single stereo
frame once tracking resumes.

Not measurable from these CSVs: which frames commit keyframes. A blackout frame's n_obs and
n_landmarks both drop to zero whether or not a keyframe was committed. For that, run with
VERBOSITY=3 and check whether a frame's timestamp reappears as the next frame's
`[INTEG WINDOW] kf_t=`.
"""

import glob
import os
import sys

import numpy as np

# Frames after a blackout window that still count as part of the event. The untreated
# recovery transient decayed over ~11 frames, so 30 covers it with margin.
RECOVERY_FRAMES = 30


def read_run(path):
    """Read one run CSV into a structured array plus its header metadata.

    The '#' metadata lines are stripped here rather than handed to genfromtxt, which would
    otherwise take the first comment line as the column names.
    """
    metadata = {}
    data_lines = []
    with open(path) as stream:
        for line in stream:
            if line.startswith('#'):
                for token in line.lstrip('#').strip().split():
                    if '=' in token:
                        key, value = token.split('=', 1)
                        metadata[key] = value
            elif line.strip():
                data_lines.append(line)

    table = np.genfromtxt(data_lines, delimiter=',', names=True)
    if table.shape == ():
        table = table.reshape(1)
    return table, metadata


def find_windows(is_blackout):
    """Contiguous [start, end) index ranges where is_blackout is set."""
    flags = np.nan_to_num(is_blackout).astype(int)
    windows = []
    start = None
    for index, flag in enumerate(flags):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            windows.append((start, index))
            start = None
    if start is not None:
        windows.append((start, len(flags)))
    return windows


def error_injection(table):
    """Odometry and ground-truth tracks, plus per-frame |(d odom) - (d gt)|."""
    pos = np.stack([table['tx'], table['ty'], table['tz']], axis=1)
    gt = np.stack([table['gt_x'], table['gt_y'], table['gt_z']], axis=1)
    step = np.linalg.norm(np.diff(pos, axis=0) - np.diff(gt, axis=0), axis=1)
    return pos, gt, np.concatenate([[np.nan], step])


def analyze(path):
    table, meta = read_run(path)
    pos, gt, injection = error_injection(table)
    windows = find_windows(table['is_blackout'])

    blackout = np.nan_to_num(table['is_blackout']).astype(bool)
    failures = table['tracked'] != 1
    gravity = np.nan_to_num(table['has_gravity']).astype(int)

    print(f"\n=== {os.path.basename(path)} ===")
    print(f"  version {meta.get('cuvslam_version', '?')}, period {meta.get('blackout_period', '?')}, "
          f"duration {meta.get('blackout_duration', '?')}, fill {meta.get('blackout_fill', '?')}, "
          f"start_frame {meta.get('start_frame', '0')}")
    print(f"  frames {len(table)}, blackout frames {int(blackout.sum())}, "
          f"track failures {int(failures.sum())} "
          f"(on blackout {int((failures & blackout).sum())}, "
          f"elsewhere {int((failures & ~blackout).sum())})")
    print(f"  gravity coverage {100 * gravity.mean():.1f}%")

    # Baseline envelope: everything outside a blackout event and its recovery tail.
    near = np.zeros(len(table), bool)
    for start, end in windows:
        near[max(0, start - 1):min(len(table), end + RECOVERY_FRAMES + 1)] = True
    clean = injection[~near & ~np.isnan(injection)]
    threshold = np.percentile(clean, 99) if len(clean) else np.inf
    if len(clean):
        print(f"  baseline error injection (n={len(clean)}): "
              f"median {np.median(clean) * 1000:.2f} mm, p99 {threshold * 1000:.2f} mm, "
              f"max {clean.max() * 1000:.2f} mm")

    valid = ~np.isnan(gt).any(axis=1) & ~np.isnan(pos).any(axis=1)
    if valid.any():
        # Compare at the last frame ground truth covers: EuRoC GT stops before the images do.
        last = np.where(valid)[0][-1]
        deviation = np.linalg.norm(pos[valid] - gt[valid], axis=1)
        print(f"  GT deviation: rms {np.sqrt((deviation ** 2).mean()):.3f} m, "
              f"final {np.linalg.norm(pos[last] - gt[last]):.3f} m")

    if not windows:
        print("  no blackout windows (baseline run)")
        return

    print(f"  {len(windows)} windows, nominal length {windows[0][1] - windows[0][0]}")
    print(f"    {'win':>6} {'during':>12} {'after':>12} {'peak/frame':>12} {'settle':>7}")

    total_during = total_after = 0.0
    for start, end in windows:
        before = max(0, start - 1)
        settled = min(len(table) - 1, end + RECOVERY_FRAMES)
        during = np.linalg.norm((pos[end - 1] - pos[before]) - (gt[end - 1] - gt[before]))
        after = np.linalg.norm((pos[settled] - pos[end - 1]) - (gt[settled] - gt[end - 1]))
        tail = injection[end:settled + 1]
        peak = np.nanmax(tail) if len(tail) else np.nan
        # Frames until the injection rate falls back inside the baseline envelope.
        settle = 0
        for offset, value in enumerate(tail):
            if not np.isnan(value) and value > threshold:
                settle = offset + 1
        total_during += during
        total_after += after
        print(f"    {start:>6} {during * 1000:>9.1f} mm {after * 1000:>9.1f} mm "
              f"{peak * 1000:>9.1f} mm {settle:>7}")

    total = total_during + total_after
    print(f"    {'sum':>6} {total_during * 1000:>9.1f} mm {total_after * 1000:>9.1f} mm")
    if total > 0:
        print(f"  transient is {100 * total_after / total:.0f}% of all error "
              f"injected by the {len(windows)} events")


def main():
    paths = sys.argv[1:]
    if not paths:
        default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset', 'runs')
        paths = sorted(glob.glob(os.path.join(default_dir, '*.csv')))
    if not paths:
        print("No run CSVs found. Run track_euroc_mod.py first.")
        return 1
    for path in paths:
        analyze(path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
