#!/usr/bin/env python3
"""Replay a lastfeet control-rs `.rrd` through the installed cuVSLAM and compare it with the live run.

The recording holds everything the live `cuvslam_tracker` consumed -- the front_forward stereo pair, an aux
pair (rear_backward by default), the front camera's 400 Hz IMU and every camera's calibration -- plus what
the tracker produced at the time (`/robot/slam/pose`, `.../trajectory_odom`, `.../loop_closures`). This script
feeds the same inputs to whatever `cuvslam` is installed, with the configuration the live tracker uses
(`python/cuvslam-tracker/cuvslam_tracker/tracker.py` in the lastfeet monorepo): Multisensor + IMU, no depth,
SLAM with background threads, `async_sba`, `use_gpu`, rig anchored at the URDF base, output composed with the
initial world pose. The live poses in the recording are in that same world frame, so the two trajectories are
directly comparable with no alignment step.

What it writes (into --out, named `<recording>__<label>`):

    .rrd    rerun recording: 3D trajectories (live vs replay) plus track() time, divergence, gravity series
    .png    one-page summary of the same
    .json   all numbers (also printed at the end)
    .npz    the raw per-frame arrays, so runs can be overlaid later with --overlay

Typical use (A/B over the two IMU patches, one build, no rebuild):

    python replay_rrd_compare.py logs/.../real_nov-11-2026_12-58-41.rrd --out out/
    CUVSLAM_HOLD_GRAVITY=0 python replay_rrd_compare.py <rrd> --out out/
    CUVSLAM_IMU_SBA_INIT=1 python replay_rrd_compare.py <rrd> --out out/
    python replay_rrd_compare.py <rrd> --out out/ --no-replay --overlay out/*.npz   # side-by-side figure

Needs, besides cuvslam: numpy, scipy, opencv, pyyaml, matplotlib and rerun-sdk >= 0.35 (older rerun cannot
read these files). It imports the live tracker's calibration helpers from the monorepo (--monorepo).

Pacing. By default frames are fed at the recorded rate (`--speed 1`), because track() cost depends on what
else the machine is doing and on the async SBA/SLAM threads, which behave differently when fed as fast as
possible. `--speed 0` feeds as fast as possible (quick functional runs). `lag_ms` in the output is how far
behind its recorded time a frame was when track() started -- a growing lag means the tracker cannot keep up,
which is the live "backlog" symptom.

Not replicated: the live tracker drops stale frames when it falls behind (drain-to-newest); this replays every
recorded frame. The recording itself has gaps of up to ~1 s where the cameras stalled, and those are replayed.
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------------------------------------
# Arguments
# --------------------------------------------------------------------------------------------------------

DEFAULT_MONOREPO = Path.home() / "code" / "lastfeet-monorepo"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("rrd", type=Path, help="control-rs recording (the raw .rrd, not *_policy_pinhole.rrd)")
    p.add_argument("--out", type=Path, default=Path("replay_out"), help="output directory")
    p.add_argument("--label", default=None,
                   help="run label (default: cuvslam version + the two IMU experiment flags)")
    p.add_argument("--monorepo", type=Path, default=DEFAULT_MONOREPO,
                   help="lastfeet monorepo checkout (live tracker helpers, config, URDF)")
    p.add_argument("--config", type=Path, default=None,
                   help="robot config for IMU noise / correction quaternion (default <monorepo>/configs/real_boxer.yaml)")
    p.add_argument("--urdf", type=Path, default=None, help="URDF (default <monorepo>/robot_description/urdf/out.urdf)")
    p.add_argument("--primary", default="front_forward", help="primary camera label (owns the time base and the IMU)")
    p.add_argument("--aux", nargs="*", default=["rear_backward"],
                   help="aux stereo pairs, in rig order. Live used `rear_backward`; add `left right` to try 4 pairs.")
    p.add_argument("--max-seconds", type=float, default=None, help="replay only the first N seconds")
    p.add_argument("--speed", type=float, default=1.0,
                   help="1 = feed at the recorded rate; 2 = twice as fast; 0 = as fast as possible")
    p.add_argument("--no-slam", action="store_true", help="odometry only (Mode.OdometryOnlyRealtime)")
    p.add_argument("--sync", action="store_true",
                   help="offline modes: async_sba off, SLAM in the calling thread (reproducible, not live-like)")
    p.add_argument("--no-imu", action="store_true", help="Multisensor without IMU fusion")
    p.add_argument("--cuvslam-verbosity", type=int, default=2,
                   help="2 (warning) prints the IMU experiment flags the build is running with")
    p.add_argument("--no-replay", action="store_true",
                   help="do not run cuvslam; only draw the live trajectory and the --overlay runs")
    p.add_argument("--overlay", nargs="*", type=Path, default=[], help=".npz files from earlier runs to draw too")
    p.add_argument("--spawn", action="store_true", help="also open the rerun viewer")
    p.add_argument("--no-rerun", action="store_true", help="skip the rerun recording")
    return p.parse_args()


# --------------------------------------------------------------------------------------------------------
# Reading the recording
# --------------------------------------------------------------------------------------------------------

def _open_reader(path):
    try:
        from rerun.chunk import RrdReader
    except ImportError:  # rerun < 0.39 kept it here
        from rerun.experimental import RrdReader
    return RrdReader(str(path))


def _i64(rb, name):
    if name not in rb.schema.names:
        return None
    return rb.column(name).to_numpy(zero_copy_only=False).astype("int64")


def _has(rb, *names):
    """A static chunk (series names, pinhole, ...) shares the entity path but carries none of the data columns."""
    return all(n in rb.schema.names for n in names)


def _blobs(rb):
    col = rb.column("EncodedImage:blob")
    return [col[i].values.flatten().to_numpy(zero_copy_only=False).tobytes() for i in range(rb.num_rows)]


def _vec3(rb, name):
    return np.asarray(rb.column(name).to_pylist(), dtype=np.float64).reshape(rb.num_rows, -1)


def load_recording(path, primary, aux, max_seconds):
    """One streaming pass over the recording; returns plain python/numpy containers."""
    cams = [primary] + list(aux)
    img_paths = {}
    for c in cams:
        for side in ("left", "right"):
            img_paths[f"/robot/cameras/{c}/stereo_pair/{side}/img"] = (c, side)
    imu_paths = {f"/robot/cameras/{primary}/imu/accel": "accel", f"/robot/cameras/{primary}/imu/gyro": "gyro"}
    calib_paths = {f"/meta/cameras/{c}_camera/calib": c for c in cams}

    frames = {(c, s): [] for c in cams for s in ("left", "right")}  # (camera_ts, host_sync_ts, png)
    imu = {"accel": [], "gyro": []}                                  # (camera_ts, xyz)
    calib = {}
    live_pose, live_odom, live_lc = [], [], []

    t0 = time.time()
    for ch in _open_reader(path).stream():
        if ch.num_rows == 0:
            continue
        ep = str(ch.entity_path)
        if ep in img_paths:
            rb = ch.to_record_batch()
            if not _has(rb, "EncodedImage:blob"):
                continue
            cam_ts, host_ts = _i64(rb, "camera_ts"), _i64(rb, "host_sync_ts")
            if cam_ts is None or host_ts is None:
                sys.exit(f"{ep} has no camera_ts/host_sync_ts: recording predates host-synced pairing")
            frames[img_paths[ep]].extend(zip(cam_ts.tolist(), host_ts.tolist(), _blobs(rb)))
        elif ep in imu_paths:
            rb = ch.to_record_batch()
            if not _has(rb, "Scalars:scalars", "camera_ts"):
                continue
            ts = _i64(rb, "camera_ts")
            xyz = _vec3(rb, "Scalars:scalars")
            imu[imu_paths[ep]].append((ts, xyz))
        elif ep in calib_paths:
            rb = ch.to_record_batch()
            if not _has(rb, "TextDocument:text"):
                continue
            calib[calib_paths[ep]] = json.loads(rb.column("TextDocument:text")[0].values[0].as_py())
        elif ep == "/robot/slam/pose":
            rb = ch.to_record_batch()
            if not _has(rb, "Transform3D:translation", "camera_ts"):
                continue
            live_pose.append((_i64(rb, "camera_ts"), _vec3(rb, "Transform3D:translation"),
                              _vec3(rb, "Transform3D:quaternion")))
        elif ep == "/robot/slam/trajectory_odom":
            rb = ch.to_record_batch()
            if not _has(rb, "LineStrips3D:strips", "camera_ts"):
                continue
            # Each row is the newest segment [previous, current]; the current point is the odometry pose.
            pts = [row[0][-1] for row in rb.column("LineStrips3D:strips").to_pylist()]
            live_odom.append((_i64(rb, "camera_ts"), np.asarray(pts, dtype=np.float64)))
        elif ep == "/robot/slam/loop_closures":
            rb = ch.to_record_batch()
            if not _has(rb, "Points3D:positions", "camera_ts"):
                continue
            pos = [row[0] for row in rb.column("Points3D:positions").to_pylist()]
            live_lc.append((_i64(rb, "camera_ts"), np.asarray(pos, dtype=np.float64)))
    print(f"read {path.name} in {time.time() - t0:.0f}s")

    missing = [c for c in cams if c not in calib]
    if missing:
        sys.exit(f"no calibration in the recording for: {missing}")

    def cat(parts, k):
        return np.concatenate([p[k] for p in parts]) if parts else np.empty((0,))

    rec = {"calib": calib, "cams": cams}

    # Stereo pairs: join left/right on camera_ts, sort, drop duplicate timestamps.
    rec["pairs"] = {}
    for c in cams:
        L = {t: (h, b) for t, h, b in frames[(c, "left")]}
        R = {t: (h, b) for t, h, b in frames[(c, "right")]}
        ts = sorted(set(L) & set(R))
        rec["pairs"][c] = {
            "cam_ts": np.asarray(ts, dtype=np.int64),
            "host_ts": np.asarray([L[t][0] for t in ts], dtype=np.int64),
            "left": [L[t][1] for t in ts],
            "right": [R[t][1] for t in ts],
        }
        print(f"  {c}: {len(ts)} stereo pairs")

    # IMU: accel and gyro share timestamps; sort once.
    a_ts, g_ts = cat(imu["accel"], 0), cat(imu["gyro"], 0)
    a_xyz = np.concatenate([p[1] for p in imu["accel"]]) if imu["accel"] else np.empty((0, 3))
    g_xyz = np.concatenate([p[1] for p in imu["gyro"]]) if imu["gyro"] else np.empty((0, 3))
    oa, og = np.argsort(a_ts, kind="stable"), np.argsort(g_ts, kind="stable")
    a_ts, a_xyz, g_ts, g_xyz = a_ts[oa], a_xyz[oa], g_ts[og], g_xyz[og]
    if len(a_ts) != len(g_ts) or (len(a_ts) and not np.array_equal(a_ts, g_ts)):
        n = min(len(a_ts), len(g_ts))
        print(f"  WARNING: accel/gyro timestamps differ ({len(a_ts)} vs {len(g_ts)}); using the first {n} of each")
        a_ts, a_xyz, g_xyz = a_ts[:n], a_xyz[:n], g_xyz[:n]
    rec["imu"] = {"ts": a_ts, "accel": a_xyz, "gyro": g_xyz}
    print(f"  imu: {len(a_ts)} samples")

    # Live outputs.
    if live_pose:
        ts, t, q = cat(live_pose, 0).astype(np.int64), np.concatenate([p[1] for p in live_pose]), \
            np.concatenate([p[2] for p in live_pose])
        o = np.argsort(ts, kind="stable")
        rec["live_pose"] = {"ts": ts[o], "xyz": t[o], "quat": q[o]}
    else:
        rec["live_pose"] = None
    if live_odom:
        ts, xyz = cat(live_odom, 0).astype(np.int64), np.concatenate([p[1] for p in live_odom])
        o = np.argsort(ts, kind="stable")
        rec["live_odom"] = {"ts": ts[o], "xyz": xyz[o]}
    else:
        rec["live_odom"] = None
    if live_lc:
        ts, xyz = cat(live_lc, 0).astype(np.int64), np.concatenate([p[1] for p in live_lc])
        o = np.argsort(ts, kind="stable")
        rec["live_lc"] = {"ts": ts[o], "xyz": xyz[o]}
    else:
        rec["live_lc"] = None

    if max_seconds is not None:
        p = rec["pairs"][primary]
        keep = int(np.searchsorted(p["cam_ts"], p["cam_ts"][0] + int(max_seconds * 1e9), side="right"))
        cut = p["cam_ts"][keep - 1]
        for k in ("cam_ts", "host_ts", "left", "right"):
            p[k] = p[k][:keep]
        h_cut = p["host_ts"][-1]
        for c in aux:
            q = rec["pairs"][c]
            k2 = int(np.searchsorted(q["host_ts"], h_cut + int(0.2e9), side="right"))
            for k in ("cam_ts", "host_ts", "left", "right"):
                q[k] = q[k][:k2]
        m = rec["imu"]["ts"] <= cut
        for k in ("ts", "accel", "gyro"):
            rec["imu"][k] = rec["imu"][k][m]
        for k in ("live_pose", "live_odom"):
            if rec[k] is not None:
                mm = rec[k]["ts"] <= cut
                rec[k] = {kk: vv[mm] for kk, vv in rec[k].items()}
        if rec["live_lc"] is not None:
            mm = rec["live_lc"]["ts"] <= cut
            rec["live_lc"] = {kk: vv[mm] for kk, vv in rec["live_lc"].items()}
    return rec


# --------------------------------------------------------------------------------------------------------
# Rig / config (mirrors cuvslam_tracker/tracker.py)
# --------------------------------------------------------------------------------------------------------

def import_live_helpers(monorepo):
    pkg = monorepo / "python" / "cuvslam-tracker"
    if not (pkg / "cuvslam_tracker" / "calib.py").exists():
        sys.exit(f"cannot find the live tracker helpers under {pkg}; pass --monorepo")
    sys.path.insert(0, str(pkg))
    from cuvslam_tracker import calib, urdf  # noqa: E402
    return calib, urdf


def load_slam_config(path):
    import yaml
    cfg = yaml.safe_load(path.read_text())
    slam = cfg["perception"]["slam"]
    n = slam["imu_noise"]
    noise = (n["accel_noise_density"], n["accel_random_walk"], n["gyro_noise_density"], n["gyro_random_walk"])
    return noise, [float(v) for v in slam["imu_correction_quat"]]


def compose_T_base_from_cam(urdf_mod, urdf_path, camera_link):
    """Same chain as the live tracker: base -> upper_body -> <label>_camera (the camera link is CAM_A, RDF)."""
    return urdf_mod.read_fixed_joint_transform(str(urdf_path), "base", "upper_body") @ \
        urdf_mod.read_fixed_joint_transform(str(urdf_path), "upper_body", camera_link)


def parse_live_log(rrd):
    """Cross-check values the live tracker printed at start-up, from the sibling .log if there is one."""
    log = rrd.with_suffix(".log")
    out = {}
    if not log.exists():
        return out
    for line in log.read_text(errors="replace").splitlines():
        m = re.search(r"T_base_from_camA translation=\[([^\]]+)\]", line)
        if m and "base_from_camA" not in out:
            out["base_from_camA"] = [float(x) for x in re.split(r"[,\s]+", m.group(1).strip())]
        m = re.search(r"T_world_from_base translation=.*quat\(xyzw\)=\[([^\]]+)\]", line)
        if m and "world_from_base_quat" not in out:
            out["world_from_base_quat"] = [float(x) for x in re.split(r"[,\s]+", m.group(1).strip())]
    return out


# --------------------------------------------------------------------------------------------------------
# Replay
# --------------------------------------------------------------------------------------------------------

EMPTY_IMAGE = np.empty((0, 0), dtype=np.uint8)


def match_aux(primary, aux):
    """Per primary tick, the index of the aux pair to use (or -1). Nearest on host_sync_ts within half the aux
    frame period, each aux frame used at most once -- the live AuxPairSet rule without the sockets."""
    h = aux["host_ts"]
    if len(h) < 2:
        return np.full(len(primary["host_ts"]), -1)
    period = float(np.median(np.diff(h)))
    tol = 0.5 * period
    claimed, out = set(), []
    for hp in primary["host_ts"]:
        i = int(np.searchsorted(h, hp))
        cand = [j for j in (i - 1, i) if 0 <= j < len(h)]
        j = min(cand, key=lambda k: abs(h[k] - hp)) if cand else -1
        if j >= 0 and abs(h[j] - hp) <= tol and j not in claimed:
            claimed.add(j)
            out.append(j)
        else:
            out.append(-1)
    return np.asarray(out)


def pose_row(pose):
    return [*map(float, pose.translation), *map(float, pose.rotation)]


def run_replay(args, rec, label):
    import cuvslam
    from scipy.spatial.transform import Rotation as R

    calib_mod, urdf_mod = import_live_helpers(args.monorepo)
    config_path = args.config or args.monorepo / "configs" / "real_boxer.yaml"
    urdf_path = args.urdf or args.monorepo / "robot_description" / "urdf" / "out.urdf"
    noise, correction_quat = load_slam_config(config_path)

    primary, aux_labels = args.primary, list(args.aux)
    cams = [primary] + aux_labels
    P = rec["pairs"][primary]
    if len(P["cam_ts"]) < 2:
        sys.exit("fewer than two primary frames")

    # World frame: the live tracker's T_world_from_base, which is the first live slam pose's rotation (the live
    # tracker emits initial_pose * slam_rel and slam_rel starts at identity).
    live_log = parse_live_log(args.rrd)
    if rec["live_pose"] is not None:
        q0 = rec["live_pose"]["quat"][0].tolist()
        t0 = rec["live_pose"]["xyz"][0].tolist()
    elif "world_from_base_quat" in live_log:
        q0, t0 = live_log["world_from_base_quat"], [0.0, 0.0, 0.0]
    else:
        sys.exit("no live pose and no sibling .log: cannot recover the initial world pose")
    if "world_from_base_quat" in live_log:
        dq = np.abs(np.asarray(q0) - np.asarray(live_log["world_from_base_quat"])).max()
        print(f"initial world pose: from recording; live log agrees to {dq:.1e}" if dq < 1e-4
              else f"WARNING initial pose differs from the live log by {dq:.2e}")
    initial_pose = cuvslam.Pose(translation=t0, rotation=q0)

    T_base_primary = compose_T_base_from_cam(urdf_mod, urdf_path, f"{primary}_camera")
    if "base_from_camA" in live_log:
        dt = np.abs(T_base_primary[:3, 3] - np.asarray(live_log["base_from_camA"])).max()
        print("T_base_from_camA matches the live log" if dt < 1e-4 else f"WARNING T_base_from_camA off by {dt:.2e} m")
    cameras = calib_mod.build_cameras_from_calib(rec["calib"][primary], T_rig_from_camA=T_base_primary)
    for a in aux_labels:
        cameras += calib_mod.build_cameras_from_calib(
            rec["calib"][a], T_rig_from_camA=compose_T_base_from_cam(urdf_mod, urdf_path, f"{a}_camera"))
    imu_calib = None
    if not args.no_imu:
        imu_calib = calib_mod.build_imu_from_calib(rec["calib"][primary], T_base_primary, correction_quat, noise)
        if imu_calib is None:
            print("primary calibration has no imu block: running without IMU fusion")
    rig = cuvslam.Rig(cameras, [imu_calib] if imu_calib is not None else [])

    # Live configuration; --sync switches to the offline modes, which need the opposite sync settings.
    realtime = not args.sync
    odom_cfg = cuvslam.Odometry.Config(
        async_sba=realtime,
        enable_observations_export=False,
        enable_landmarks_export=False,
        enable_final_landmarks_export=False,
        rectified_stereo_camera=False,
        use_gpu=True,
        odometry_mode=cuvslam.Odometry.OdometryMode.Multisensor,
        multisensor_settings=cuvslam.Odometry.MultisensorSettings(depth_camera_ids=[]),
    )
    slam_on = not args.no_slam
    # Builds from before the explicit Tracker.Mode (v17.x, e.g. the one deployed on the robots on 2026-11-11)
    # take (rig, odom_cfg, slam_cfg or None) and derive the mode from the configs; this lets the same script
    # replay them as a baseline.
    has_mode = hasattr(cuvslam.Tracker, "Mode")
    if slam_on:
        slam_cfg = cuvslam.Slam.Config(sync_mode=not realtime)
    else:
        slam_cfg = None
    if has_mode:
        if slam_on:
            mode = (cuvslam.Tracker.Mode.OdometryWithSlamRealtime if realtime
                    else cuvslam.Tracker.Mode.OdometryWithSlamOffline)
        else:
            mode = (cuvslam.Tracker.Mode.OdometryOnlyRealtime if realtime
                    else cuvslam.Tracker.Mode.OdometryOnlyOffline)
    else:
        mode = "legacy (no Tracker.Mode)"
    cuvslam.set_verbosity(args.cuvslam_verbosity)
    try:
        cuvslam.warm_up_gpu()
    except Exception as e:  # noqa: BLE001 -- warm-up is an optimisation only
        print(f"warm_up_gpu failed: {e}")
    tracker = (cuvslam.Tracker(rig, mode, odom_cfg, slam_cfg) if has_mode
               else cuvslam.Tracker(rig, odom_cfg, slam_cfg))
    print(f"tracker: {cuvslam.get_version()[0]} mode={mode} imu={'on' if imu_calib else 'off'} "
          f"cameras={len(cameras)} ({', '.join(cams)})")

    # Per-tick inputs.
    aux_idx = {a: match_aux(P, rec["pairs"][a]) for a in aux_labels}
    for a in aux_labels:
        print(f"  aux {a}: matched on {int((aux_idx[a] >= 0).sum())}/{len(P['cam_ts'])} ticks")
    n = len(P["cam_ts"])
    frame_period_s = float(np.median(np.diff(P["host_ts"]))) / 1e9
    imu = rec["imu"]
    imu_ok = imu_calib is not None and len(imu["ts"]) > 0
    j_imu, floor = 0, None
    pool = ThreadPoolExecutor(max_workers=2 * len(cams))

    def decode(blob):
        return calib_mod.decode_png(blob)

    out = {k: np.full(n, np.nan) for k in ("track_ms", "lag_ms", "gravity_tilt_deg")}
    odom = np.full((n, 7), np.nan)
    slam = np.full((n, 7), np.nan)
    aux_used = np.zeros(n, dtype=np.int64)
    errors, lc_events, seen_lc = [], [], 0
    gravity_frame = None
    wall0, host0 = None, int(P["host_ts"][0])
    t_start = time.time()

    for i in range(n):
        ts = int(P["cam_ts"][i])
        images = [P["left"][i], P["right"][i]]
        slots = [0, 1]
        for k, a in enumerate(aux_labels):
            j = int(aux_idx[a][i])
            if j >= 0:
                images += [rec["pairs"][a]["left"][j], rec["pairs"][a]["right"][j]]
                slots += [2 + 2 * k, 3 + 2 * k]
        decoded = list(pool.map(decode, images))
        full = [EMPTY_IMAGE] * (2 * len(cams))
        for s, im in zip(slots, decoded):
            full[s] = im if im is not None else EMPTY_IMAGE
        if full[0] is EMPTY_IMAGE or full[1] is EMPTY_IMAGE:
            errors.append((i, "corrupt primary PNG"))
            continue
        aux_used[i] = (len(slots) - 2) // 2

        if imu_ok:
            while j_imu < len(imu["ts"]) and imu["ts"][j_imu] <= ts:
                its = int(imu["ts"][j_imu])
                if floor is None or its > floor:
                    tracker.register_imu_measurement(0, cuvslam.ImuMeasurement(
                        timestamp_ns=its, linear_accelerations=imu["accel"][j_imu].tolist(),
                        angular_velocities=imu["gyro"][j_imu].tolist()))
                    floor = its
                j_imu += 1

        now = time.time()
        if args.speed > 0:
            if wall0 is None:
                wall0 = now
            target = wall0 + (int(P["host_ts"][i]) - host0) / 1e9 / args.speed
            if target > now:
                time.sleep(target - now)
                now = time.time()
            out["lag_ms"][i] = (now - target) * 1e3
        t0 = time.perf_counter()
        try:
            odom_est, slam_pose = tracker.track(ts, full)
        except Exception as e:  # noqa: BLE001 -- keep replaying, report at the end
            errors.append((i, f"{type(e).__name__}: {e}"))
            continue
        out["track_ms"][i] = (time.perf_counter() - t0) * 1e3
        floor = ts

        if odom_est.world_from_rig is None:
            continue
        odom_world = calib_mod.combine_poses(initial_pose, odom_est.world_from_rig.pose)
        odom[i] = pose_row(odom_world)
        if slam_on and slam_pose is not None:
            slam[i] = pose_row(calib_mod.combine_poses(initial_pose, slam_pose))
        elif not slam_on:
            slam[i] = odom[i]

        if imu_ok:
            g = tracker.odometry.get_last_gravity()
            if g is not None:
                gw = R.from_quat(odom_world.rotation).apply(np.asarray(g, dtype=float))
                gh = gw / max(np.linalg.norm(gw), 1e-9)
                out["gravity_tilt_deg"][i] = float(np.degrees(np.arccos(np.clip(-gh[2], -1.0, 1.0))))
                if gravity_frame is None:
                    gravity_frame = i
        if slam_on:
            for lc in tracker.slam.get_loop_closure_poses():
                if lc.timestamp_ns > seen_lc:
                    pw = calib_mod.combine_poses(initial_pose, lc.pose)
                    lc_events.append((int(lc.timestamp_ns), *map(float, pw.translation)))
            if lc_events:
                seen_lc = max(e[0] for e in lc_events)
        if i and i % 300 == 0:
            tm = out["track_ms"][max(0, i - 300):i]
            print(f"  tick {i}/{n}  t={(ts - P['cam_ts'][0]) / 1e9:6.1f}s  track mean {np.nanmean(tm):5.1f} ms "
                  f"max {np.nanmax(tm):5.1f}  lag {out['lag_ms'][i]:.0f} ms  LC {len(lc_events)}")
    wall = time.time() - t_start

    final_ts = final_xyz = None
    if slam_on:
        t0 = time.time()
        poses = tracker.slam.get_all_slam_poses()
        print(f"get_all_slam_poses: {len(poses)} poses in {time.time() - t0:.1f}s")
        final_ts = np.asarray([p.timestamp_ns for p in poses], dtype=np.int64)
        final_xyz = np.asarray([calib_mod.combine_poses(initial_pose, p.pose).translation for p in poses])
    if errors:
        print(f"{len(errors)} tick errors, first: {errors[0]}")

    flags = {k: os.environ.get(k, "default") for k in ("CUVSLAM_HOLD_GRAVITY", "CUVSLAM_IMU_SBA_INIT")}
    meta = {
        "label": label, "recording": args.rrd.name, "cuvslam_version": cuvslam.get_version()[0],
        "flags": flags, "mode": str(mode), "speed": args.speed, "imu": bool(imu_calib), "slam": slam_on,
        "cameras": cams, "ticks": n, "tick_errors": len(errors), "wall_s": wall,
        "frame_period_ms": frame_period_s * 1e3, "gravity_first_frame": gravity_frame,
        "max_seconds": args.max_seconds,
    }
    return {
        "meta": meta, "ts": P["cam_ts"], "host_ts": P["host_ts"], "odom": odom, "slam": slam,
        "aux_used": aux_used, "lc": np.asarray(lc_events).reshape(-1, 4),
        "final_ts": final_ts if final_ts is not None else np.empty(0, np.int64),
        "final_xyz": final_xyz if final_xyz is not None else np.empty((0, 3)), **out,
    }


# --------------------------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------------------------

def _path_len(xyz):
    xyz = xyz[~np.isnan(xyz).any(axis=1)]
    return float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()) if len(xyz) > 1 else 0.0


def _end_to_start(xyz):
    xyz = xyz[~np.isnan(xyz).any(axis=1)]
    return float(np.linalg.norm(xyz[-1] - xyz[0])) if len(xyz) > 1 else float("nan")


def live_arrays(rec):
    """Live outputs as (ts, xyz) tuples; slam also has quats."""
    lp, lo = rec["live_pose"], rec["live_odom"]
    return (None if lp is None else (lp["ts"], lp["xyz"]), None if lo is None else (lo["ts"], lo["xyz"]))


def divergence(run_ts, run_xyz, live):
    """Position difference at ticks present in both (camera_ts join)."""
    if live is None:
        return np.full(len(run_ts), np.nan)
    lt, lx = live
    idx = np.searchsorted(lt, run_ts)
    idx_c = np.clip(idx, 0, len(lt) - 1)
    ok = (lt[idx_c] == run_ts) & ~np.isnan(run_xyz).any(axis=1)
    d = np.full(len(run_ts), np.nan)
    d[ok] = np.linalg.norm(run_xyz[ok] - lx[idx_c[ok]], axis=1)
    return d


def metrics_for(run, rec):
    live_slam, live_odom = live_arrays(rec)
    ts = run["ts"]
    t_s = (ts - ts[0]) / 1e9
    tm = run["track_ms"]
    ok = ~np.isnan(tm)
    m = {"label": run["meta"]["label"]}
    tracked = ~np.isnan(run["slam"][:, 0])
    m["ticks"], m["tracked_ticks"] = int(len(ts)), int(tracked.sum())
    if ok.any():
        v = tm[ok]
        m["track_ms"] = {"mean": float(v.mean()), "p50": float(np.percentile(v, 50)),
                         "p95": float(np.percentile(v, 95)), "p99": float(np.percentile(v, 99)),
                         "max": float(v.max())}
        budget = run["meta"]["frame_period_ms"]
        m["track_over_frame_period"] = int((v > budget).sum())
        m["track_over_100ms"] = int((v > 100).sum())
        third = max(1, ok.sum() // 3)
        m["track_ramp_last_over_first_third"] = float(v[-third:].mean() / max(v[:third].mean(), 1e-9))
        win = np.floor(t_s[ok] / 10).astype(int)
        m["track_ms_per_10s_window_mean"] = [float(v[win == w].mean()) for w in np.unique(win)]
    lag = run["lag_ms"][~np.isnan(run["lag_ms"])]
    if len(lag):
        m["lag_ms"] = {"mean": float(lag.mean()), "max": float(lag.max()), "final": float(lag[-1])}
    gt = run["gravity_tilt_deg"]
    m["gravity_first_frame"] = run["meta"]["gravity_first_frame"]
    if (~np.isnan(gt)).any():
        m["gravity_tilt_deg"] = {"median": float(np.nanmedian(gt)), "max": float(np.nanmax(gt)),
                                 "final": float(gt[~np.isnan(gt)][-1])}
    m["loop_closures"] = int(len(run["lc"]))
    m["path_len_m"] = {"slam": _path_len(run["slam"][:, :3]), "odom": _path_len(run["odom"][:, :3])}
    m["end_to_start_m"] = {"slam": _end_to_start(run["slam"][:, :3]), "odom": _end_to_start(run["odom"][:, :3])}
    if len(run["final_xyz"]):
        m["path_len_m"]["slam_final"] = _path_len(run["final_xyz"])
        m["end_to_start_m"]["slam_final"] = _end_to_start(run["final_xyz"])
    d = divergence(ts, run["slam"][:, :3], live_slam)
    run["divergence_m"] = d
    dd = d[~np.isnan(d)]
    if len(dd):
        m["divergence_vs_live_slam_m"] = {"median": float(np.median(dd)), "p95": float(np.percentile(dd, 95)),
                                          "max": float(dd.max()), "final": float(dd[-1])}
    return m


def live_metrics(rec, replay_ticks_ts):
    live_slam, live_odom = live_arrays(rec)
    m = {}
    if live_slam is not None:
        m["slam_poses"] = int(len(live_slam[0]))
        m["path_len_m"] = _path_len(live_slam[1])
        m["end_to_start_m"] = _end_to_start(live_slam[1])
    if live_odom is not None:
        m["odom_path_len_m"] = _path_len(live_odom[1])
        m["odom_end_to_start_m"] = _end_to_start(live_odom[1])
    m["loop_closures"] = int(len(rec["live_lc"]["ts"])) if rec["live_lc"] is not None else None
    return m


def print_summary(rec, runs_metrics, live_m):
    print("\n==== summary ====")
    print(f"live            poses {live_m.get('slam_poses')}  slam path {live_m.get('path_len_m', float('nan')):.1f} m"
          f"  end->start {live_m.get('end_to_start_m', float('nan')):.2f} m (odom {live_m.get('odom_end_to_start_m', float('nan')):.2f} m)"
          f"  loop closures {live_m.get('loop_closures')}")
    for m in runs_metrics:
        t = m.get("track_ms", {})
        dv = m.get("divergence_vs_live_slam_m", {})
        print(f"{m['label']}\n  tracked {m['tracked_ticks']}/{m['ticks']}  track() mean {t.get('mean', float('nan')):.1f} "
              f"p95 {t.get('p95', float('nan')):.1f} p99 {t.get('p99', float('nan')):.1f} max {t.get('max', float('nan')):.1f} ms"
              f"  >frame period {m.get('track_over_frame_period')}  ramp(last/first third) "
              f"{m.get('track_ramp_last_over_first_third', float('nan')):.2f}")
        if "lag_ms" in m:
            print(f"  lag mean {m['lag_ms']['mean']:.0f} max {m['lag_ms']['max']:.0f} final {m['lag_ms']['final']:.0f} ms")
        print(f"  slam path {m['path_len_m']['slam']:.1f} m  end->start {m['end_to_start_m']['slam']:.2f} m "
              f"(odom {m['end_to_start_m']['odom']:.2f} m"
              + (f", final-graph {m['end_to_start_m']['slam_final']:.2f} m" if "slam_final" in m['end_to_start_m'] else "")
              + f")  loop closures {m['loop_closures']}")
        if dv:
            print(f"  vs live slam: median {dv['median']:.2f}  p95 {dv['p95']:.2f}  max {dv['max']:.2f}  "
                  f"final {dv['final']:.2f} m")
        if "gravity_tilt_deg" in m:
            g = m["gravity_tilt_deg"]
            print(f"  gravity converged at tick {m['gravity_first_frame']}, tilt median {g['median']:.1f}  "
                  f"max {g['max']:.1f} deg")


# --------------------------------------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------------------------------------

PALETTE = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189), (140, 86, 75)]


def save_npz(path, run):
    arrays = {k: v for k, v in run.items() if k != "meta"}
    np.savez_compressed(path, meta=json.dumps(run["meta"]), **arrays)


def load_npz(path):
    z = np.load(path, allow_pickle=False)
    run = {k: z[k] for k in z.files if k != "meta"}
    run["meta"] = json.loads(str(z["meta"]))
    return run


def write_png(path, rec, runs, runs_metrics, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    live_slam, live_odom = live_arrays(rec)
    fig, ax = plt.subplots(2, 2, figsize=(16, 11))
    a = ax[0, 0]
    if live_odom is not None:
        a.plot(live_odom[1][:, 0], live_odom[1][:, 1], color="0.75", lw=1, label="live odom")
    if live_slam is not None:
        a.plot(live_slam[1][:, 0], live_slam[1][:, 1], color="k", lw=1.6, label="live slam")
        a.plot(*live_slam[1][0, :2], "ko", ms=7)
    if rec["live_lc"] is not None and len(rec["live_lc"]["xyz"]):
        a.plot(rec["live_lc"]["xyz"][:, 0], rec["live_lc"]["xyz"][:, 1], "kx", ms=5, alpha=0.5, label="live LC")
    for k, run in enumerate(runs):
        c = np.array(PALETTE[k % len(PALETTE)]) / 255
        lab = run["meta"]["label"]
        a.plot(run["odom"][:, 0], run["odom"][:, 1], color=c, lw=0.8, ls=":", alpha=0.8)
        a.plot(run["slam"][:, 0], run["slam"][:, 1], color=c, lw=1.4, label=f"{lab} slam")
        if len(run["lc"]):
            a.plot(run["lc"][:, 1], run["lc"][:, 2], "+", color=c, ms=7)
    a.set_aspect("equal"); a.grid(alpha=0.3); a.set_xlabel("x [m]"); a.set_ylabel("y [m]")
    a.set_title("top-down: solid = slam, dotted = odometry, x/+ = loop closures")
    a.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))

    t0 = runs[0]["ts"][0] if runs else 0
    b = ax[0, 1]
    for k, run in enumerate(runs):
        c = np.array(PALETTE[k % len(PALETTE)]) / 255
        t = (run["ts"] - t0) / 1e9
        b.plot(t, run["track_ms"], color=c, lw=0.5, alpha=0.5)
        k_ = 31
        sm = np.convolve(np.nan_to_num(run["track_ms"], nan=np.nanmedian(run["track_ms"])), np.ones(k_) / k_, "same")
        b.plot(t, sm, color=c, lw=1.8, label=f"{run['meta']['label']}")
    if runs:
        b.axhline(runs[0]["meta"]["frame_period_ms"], color="r", ls="--", lw=1, label="frame period")
    b.set_yscale("log"); b.set_xlabel("t [s]"); b.set_ylabel("track() [ms]"); b.grid(alpha=0.3, which="both")
    b.set_title("track() time (thin = per frame, thick = 31-frame mean)"); b.legend(fontsize=7)

    c_ = ax[1, 0]
    for k, run in enumerate(runs):
        col = np.array(PALETTE[k % len(PALETTE)]) / 255
        c_.plot((run["ts"] - t0) / 1e9, run.get("divergence_m", np.full(len(run["ts"]), np.nan)), color=col, lw=1,
                label=run["meta"]["label"])
    c_.set_xlabel("t [s]"); c_.set_ylabel("|replay slam - live slam| [m]"); c_.grid(alpha=0.3)
    c_.set_title("position difference to the live slam pose, same frame"); c_.legend(fontsize=7)

    d = ax[1, 1]
    for k, run in enumerate(runs):
        col = np.array(PALETTE[k % len(PALETTE)]) / 255
        d.plot((run["ts"] - t0) / 1e9, run["gravity_tilt_deg"], color=col, lw=1, label=run["meta"]["label"])
    d.set_xlabel("t [s]"); d.set_ylabel("gravity off vertical [deg]"); d.grid(alpha=0.3)
    d.set_title("IMU gravity estimate vs world vertical (blank = not converged)"); d.legend(fontsize=7)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def write_rerun(path, rec, runs, spawn):
    import rerun as rr
    import rerun.blueprint as rrb

    rr.init("lastfeet_rrd_replay", spawn=spawn)
    rr.save(str(path))
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    live_slam, live_odom = live_arrays(rec)
    t0 = (runs[0]["ts"][0] if runs else (live_slam[0][0] if live_slam else 0))

    def tsec(ts):
        return (np.asarray(ts) - t0) / 1e9

    def line(entity, xyz, color, radius=0.02):
        xyz = xyz[~np.isnan(xyz).any(axis=1)]
        if len(xyz) > 1:
            rr.log(entity, rr.LineStrips3D([xyz], colors=[color], radii=radius), static=True)

    if live_slam is not None:
        line("world/live/slam", live_slam[1], (255, 255, 255), 0.03)
        for t, xyz, q in zip(tsec(rec["live_pose"]["ts"]), rec["live_pose"]["xyz"], rec["live_pose"]["quat"]):
            rr.set_time("t", duration=float(t))
            rr.log("world/live/slam_pose", rr.Transform3D(translation=xyz, rotation=rr.Quaternion(xyzw=q)))
    if live_odom is not None:
        line("world/live/odom", live_odom[1], (120, 120, 120), 0.015)
    if rec["live_lc"] is not None and len(rec["live_lc"]["xyz"]):
        rr.log("world/live/loop_closures", rr.Points3D(rec["live_lc"]["xyz"], colors=[(255, 255, 255)], radii=0.12),
               static=True)
    for k, run in enumerate(runs):
        col = PALETTE[k % len(PALETTE)]
        lab = re.sub(r"[^A-Za-z0-9_]+", "_", run["meta"]["label"])
        line(f"world/replay/{lab}/slam", run["slam"][:, :3], col, 0.03)
        line(f"world/replay/{lab}/odom", run["odom"][:, :3], tuple(int(c * 0.6) for c in col), 0.015)
        if len(run["final_xyz"]):
            line(f"world/replay/{lab}/slam_final_graph", run["final_xyz"], tuple(min(255, int(c * 1.3)) for c in col), 0.02)
        if len(run["lc"]):
            rr.log(f"world/replay/{lab}/loop_closures", rr.Points3D(run["lc"][:, 1:4], colors=[col], radii=0.12),
                   static=True)
        ts = tsec(run["ts"])
        div = run.get("divergence_m", np.full(len(ts), np.nan))
        for i in range(len(ts)):
            rr.set_time("t", duration=float(ts[i]))
            if not np.isnan(run["slam"][i, 0]):
                rr.log(f"world/replay/{lab}/slam_pose",
                       rr.Transform3D(translation=run["slam"][i, :3], rotation=rr.Quaternion(xyzw=run["slam"][i, 3:])))
            for name, arr in (("track_ms", run["track_ms"]), ("lag_ms", run["lag_ms"]),
                              ("gravity_tilt_deg", run["gravity_tilt_deg"]), ("divergence_m", div)):
                if not np.isnan(arr[i]):
                    rr.log(f"metrics/{name}/{lab}", rr.Scalars(float(arr[i])))
    rr.send_blueprint(rrb.Blueprint(
        rrb.Vertical(
            rrb.Spatial3DView(origin="world", name="live (white) vs replay"),
            rrb.Horizontal(
                rrb.TimeSeriesView(origin="metrics/track_ms", name="track() ms"),
                rrb.TimeSeriesView(origin="metrics/divergence_m", name="distance to live slam [m]"),
                rrb.TimeSeriesView(origin="metrics/gravity_tilt_deg", name="gravity off vertical [deg]"),
                rrb.TimeSeriesView(origin="metrics/lag_ms", name="lag behind recording [ms]"),
            ),
            row_shares=[3, 2],
        ),
        rrb.TimePanel(state="collapsed"),
    ))


# --------------------------------------------------------------------------------------------------------

def default_label(version):
    hg = os.environ.get("CUVSLAM_HOLD_GRAVITY", "def")
    sba = os.environ.get("CUVSLAM_IMU_SBA_INIT", "def")
    return f"{version}_hg{hg}_sba{sba}"


def main():
    args = parse_args()
    if not args.rrd.exists():
        sys.exit(f"{args.rrd} not found")
    if "policy_pinhole" in args.rrd.name:
        sys.exit("use the raw recording, not the *_policy_pinhole.rrd copy")
    args.out.mkdir(parents=True, exist_ok=True)

    rec = load_recording(args.rrd, args.primary, args.aux, args.max_seconds)
    if rec["live_pose"] is None:
        print("WARNING: the recording has no /robot/slam/pose; nothing to compare against")

    runs = []
    stem = args.rrd.stem
    if not args.no_replay:
        import cuvslam
        label = args.label or default_label(cuvslam.get_version()[0])
        print(f"\n== replay: {label}")
        run = run_replay(args, rec, label)
        runs.append(run)
    for path in args.overlay:
        runs.append(load_npz(path))
    if not runs:
        sys.exit("nothing to draw: give --overlay runs or drop --no-replay")

    runs_metrics = [metrics_for(r, rec) for r in runs]
    live_m = live_metrics(rec, runs[0]["ts"])
    print_summary(rec, runs_metrics, live_m)

    tag = "overlay" if args.no_replay else runs[0]["meta"]["label"]
    base = args.out / f"{stem}__{tag}"
    if not args.no_replay:
        save_npz(f"{base}.npz", runs[0])
    with open(f"{base}.json", "w") as f:
        json.dump({"recording": args.rrd.name, "live": live_m, "runs": runs_metrics,
                   "meta": [r["meta"] for r in runs]}, f, indent=1)
    write_png(f"{base}.png", rec, runs, runs_metrics, f"{stem}  --  " + " | ".join(r["meta"]["label"] for r in runs))
    if not args.no_rerun:
        write_rerun(f"{base}.rrd", rec, runs, args.spawn)
    print(f"\nwrote {base}.{{json,png,npz{'' if args.no_rerun else ',rrd'}}}")


if __name__ == "__main__":
    main()
