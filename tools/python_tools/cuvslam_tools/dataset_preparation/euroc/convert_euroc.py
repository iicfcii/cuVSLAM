#!/usr/bin/env python3
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

"""Convert EuRoC MAV dataset from raw zip archives to cuVSLAM converted format.

Input (raw/):
  machine_hall.zip  — MH_01_easy … MH_05_difficult
  vicon_room1.zip   — V1_01_easy … V1_03_difficult
  vicon_room2.zip   — V2_01_easy … V2_03_difficult

Output (converted/):
  <seq>/00/l.NNNNNN.png      — left  camera images (0-indexed, 6-digit)
  <seq>/01/r.NNNNNN.png      — right camera images (0-indexed, 6-digit)
  <seq>/stereo.edex          — calibration in cuVSLAM edex format
  <seq>/frame_metadata.jsonl — per-frame timestamps
  <seq>/IMU.jsonl            — IMU measurements within GT time range
  <seq>/gt.txt               — ground-truth poses relative to first frame
  euroc-slam.cfg             — reporter config: all seqs, SLAM only
  euroc-vio.cfg              — reporter config: all seqs, VIO/ODOM only
  euroc-vio_slam.cfg         — reporter config: all seqs, VIO + SLAM
  euroc_v203_slam.cfg        — reporter config: V2_03_difficult, SLAM only
  euroc_v203_vo.cfg          — reporter config: V2_03_difficult, VIO only

Usage:
  python3 convert_euroc.py [RAW_DIR] [OUT_DIR]

  RAW_DIR  defaults to datasets/euroc/raw  (relative to repo root)
  OUT_DIR  defaults to datasets/euroc/converted (relative to repo root)
"""

import argparse
import bisect, io, json, math, shutil, sys, tempfile, zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Sequence layout
# ---------------------------------------------------------------------------

_OUTER_ZIPS = [
    ("machine_hall.zip", ["MH_01_easy", "MH_02_easy", "MH_03_medium",
                          "MH_04_difficult", "MH_05_difficult"]),
    ("vicon_room1.zip",  ["V1_01_easy", "V1_02_medium", "V1_03_difficult"]),
    ("vicon_room2.zip",  ["V2_01_easy", "V2_02_medium", "V2_03_difficult"]),
]

ALL_SEQS = [seq for _, seqs in _OUTER_ZIPS for seq in seqs]

# ---------------------------------------------------------------------------
# Hardcoded VI-Sensor calibration (same for all EuRoC sequences)
# ---------------------------------------------------------------------------

# stereo.edex template — "___FRAME_END___" is replaced per sequence.
# Calibration: equidistant (fisheye) Kalibr model for the EuRoC VI-Sensor.
# IMU noise params from imu0/sensor.yaml; extrinsics from calibration.
_EDEX_TEMPLATE = """\
[
    {
        "cameras": [
            {
                "intrinsics": {
                    "distortion_model": "fisheye",
                    "distortion_params": [
                        -0.0062748193357009315,
                        0.029005519692414498,
                        -0.03438856012105873,
                        0.014830434499283266
                    ],
                    "focal": [
                        460.9855047976205,
                        459.67892586299877
                    ],
                    "principal": [
                        366.09923470990486,
                        249.22157605207943
                    ],
                    "size": [
                        752,
                        480
                    ]
                },
                "transform": [
                    [
                        1,
                        0,
                        0,
                        0
                    ],
                    [
                        0,
                        1,
                        0,
                        0
                    ],
                    [
                        0,
                        0,
                        1,
                        0
                    ]
                ]
            },
            {
                "intrinsics": {
                    "distortion_model": "fisheye",
                    "distortion_params": [
                        0.0030523152970989243,
                        0.0022729295767180894,
                        -0.0023088086978921007,
                        0.002031411542915807
                    ],
                    "focal": [
                        459.56983030590357,
                        458.20957848757143
                    ],
                    "principal": [
                        379.5888918566419,
                        255.9525258537914
                    ],
                    "size": [
                        752,
                        480
                    ]
                },
                "transform": [
                    [
                        0.9999967,
                        0.0021889,
                        -0.0013548,
                        0.1099839
                    ],
                    [
                        -0.0022078,
                        0.9998979,
                        -0.0141205,
                        0.0005322
                    ],
                    [
                        0.0013237,
                        0.0141234,
                        0.9998994,
                        -0.0004407
                    ]
                ]
            }
        ],
        "frame_end": ___FRAME_END___,
        "frame_start": 0,
        "imu": {
            "g": [
                0.33835679,
                -9.43382516,
                -2.54067297
            ],
            "measurements": "IMU.jsonl",
            "transform": [
                [ 0.0149006,  0.9996865, -0.0201192,  0.0683705],
                [ 0.9998883, -0.014921 , -0.0008608,  0.0158797],
                [-0.0011607, -0.0201041, -0.9997972,  0.0035799]
            ],
            "gyroscope_noise_density": 0.00016968,
            "gyroscope_random_walk": 0.000019393,
            "accelerometer_noise_density": 0.002,
            "accelerometer_random_walk": 0.003,
            "frequency": 200
        },
        "version": "0.9"
    },
    {
        "frame_metadata": "frame_metadata.jsonl",
        "points2d": {},
        "points3d": {},
        "rig_positions": {},
        "sequence": [
            [
                "00/l.000000.png"
            ],
            [
                "01/r.000000.png"
            ]
        ]
    }
]
"""

_EDEX_SENTINEL = "___FRAME_END___"


def _make_edex(frame_end: int) -> str:
    return _EDEX_TEMPLATE.replace(_EDEX_SENTINEL, str(frame_end))


# cam0 T_BS (row-major 3×4) from sensor.yaml: transforms cam0 → IMU/body frame.
# Same for all EuRoC sequences (same VI-Sensor hardware and calibration run).
_T_BS_DATA = [
    0.0148655429818, -0.999880929698,  0.00414029679422, -0.0216401454975,
    0.999557249008,   0.0149672133247,  0.025715529948,  -0.064676986768,
   -0.0257744366974,  0.00375618835797, 0.999660727178,   0.00981073058949,
]
_R_BC0 = [[_T_BS_DATA[i*4 + j] for j in range(3)] for i in range(3)]
_t_BC0  = [_T_BS_DATA[i*4 + 3]                     for i in range(3)]

# ---------------------------------------------------------------------------
# Math helpers (pure Python — no third-party dependencies)
# ---------------------------------------------------------------------------

def _mmul(A, B):
    return [[sum(A[i][k]*B[k][j] for k in range(3)) for j in range(3)]
            for i in range(3)]


def _mv(A, v):
    return [sum(A[i][k]*v[k] for k in range(3)) for i in range(3)]


def _inv34(R, t):
    """Inverse of rigid-body transform (rotation R, translation t)."""
    Rt = [[R[j][i] for j in range(3)] for i in range(3)]
    return Rt, _mv(Rt, [-t[i] for i in range(3)])


def _mul34(R1, t1, R2, t2):
    """Compose two rigid-body transforms."""
    return _mmul(R1, R2), [_mv(R1, t2)[i] + t1[i] for i in range(3)]


def _quat_to_rot(w, x, y, z):
    """Unit quaternion (w,x,y,z) → 3×3 rotation matrix."""
    return [
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ]


def _slerp(q0, q1, t):
    """Spherical linear interpolation between unit quaternions [w,x,y,z]."""
    dot = sum(q0[i]*q1[i] for i in range(4))
    if dot < 0:
        q1 = [-c for c in q1]
        dot = -dot
    if dot > 0.9995:
        r = [q0[i] + t*(q1[i] - q0[i]) for i in range(4)]
        n = math.sqrt(sum(c*c for c in r))
        return [c/n for c in r]
    theta0  = math.acos(dot)
    sin_th0 = math.sin(theta0)
    s0 = math.sin((1 - t)*theta0) / sin_th0
    s1 = math.sin(      t *theta0) / sin_th0
    return [s0*q0[i] + s1*q1[i] for i in range(4)]

# ---------------------------------------------------------------------------
# Ground-truth interpolation
# ---------------------------------------------------------------------------

def _parse_gt_csv(text):
    """Return list of (timestamp_float, [px,py,pz], [qw,qx,qy,qz])."""
    rows = []
    for line in text.strip().splitlines():
        if line.startswith('#'):
            continue
        v = [float(x) for x in line.split(',')]
        rows.append((v[0], v[1:4], v[4:8]))
    return rows


def _interp_body_pose(gt_data, gt_times, ts):
    """Interpolate body position (linear) and orientation (slerp) at ts."""
    ts = float(ts)
    idx = bisect.bisect_left(gt_times, ts)
    if idx == 0:
        return gt_data[0][1], gt_data[0][2]
    if idx >= len(gt_data):
        return gt_data[-1][1], gt_data[-1][2]
    t0, p0, q0 = gt_data[idx - 1]
    t1, p1, q1 = gt_data[idx]
    alpha = (ts - t0) / (t1 - t0)
    pos = [p0[i] + alpha*(p1[i] - p0[i]) for i in range(3)]
    return pos, _slerp(q0, q1, alpha)


def _cam0_world_pose(gt_data, gt_times, cam_ts):
    """(R_WC0, t_WC0): camera0 pose in world, at the given camera timestamp."""
    pos_WB, q_WB = _interp_body_pose(gt_data, gt_times, cam_ts)
    R_WB = _quat_to_rot(*q_WB)
    # T_WC0 = T_WB * T_BC0  where T_BC0 is cam0/sensor.yaml T_BS
    return _mul34(R_WB, pos_WB, _R_BC0, _t_BC0)

# ---------------------------------------------------------------------------
# Config file generation
# ---------------------------------------------------------------------------

def _json_val(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    return f'"{v}"'


def _seq_entry(seq, mode):
    """Ordered field list for one reporter cfg sequence entry."""
    label = "SLAM" if mode == "slam" else "ODOM"
    title = "EUROC-" + seq.replace("_", "-") + f"-{label}"
    fields = [
        ("enable",                True),
        ("sequence_folder",       seq),
        ("precompute_2d_tracks",  False),
        ("precompute_key_frames", False),
        ("use_gt_scale",          False),
        ("sequence_title",        title),
    ]
    if mode == "slam":
        fields.append(("use_slam", True))
    return fields


def _format_cfg(entries):
    lines = ["{"]
    lines.append('    "version": "0.1",')
    lines.append('    "write_cache": false,')
    lines.append('    "use_cuda": false,')
    lines.append('    "dataset_folder": "euroc/",')
    lines.append('    "use_icp_scaling": false,')
    lines.append('    "segment_lengths": [1, 2, 3, 5, 7.5, 10, 15, 20, 25, 35, 45],')
    lines.append('    "sequence_cfgs": [')
    for ei, fields in enumerate(entries):
        lines.append("        {")
        for fi, (k, v) in enumerate(fields):
            comma = "," if fi < len(fields) - 1 else ""
            lines.append(f'            "{k}": {_json_val(v)}{comma}')
        comma = "," if ei < len(entries) - 1 else ""
        lines.append(f"        }}{comma}")
    lines.append("  ]")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def _write_configs(out_dir, seqs):
    slam_only  = [_seq_entry(s, "slam") for s in seqs]
    odom_only  = [_seq_entry(s, "odom") for s in seqs]
    odom_slam  = [e for s in seqs
                    for e in (_seq_entry(s, "odom"), _seq_entry(s, "slam"))]

    configs = {
        "euroc-slam.cfg":     _format_cfg(slam_only),
        "euroc-vio.cfg":      _format_cfg(odom_only),
        "euroc-vio_slam.cfg": _format_cfg(odom_slam),
    }
    if "V2_03_difficult" in seqs:
        configs["euroc_v203_slam.cfg"] = _format_cfg([_seq_entry("V2_03_difficult", "slam")])
        configs["euroc_v203_vo.cfg"] = _format_cfg([_seq_entry("V2_03_difficult", "odom")])
    for name, text in configs.items():
        (out_dir / name).write_text(text)
        print(f"  wrote {name}")

# ---------------------------------------------------------------------------
# Per-sequence conversion
# ---------------------------------------------------------------------------

def _convert_sequence(seq_name, inner_zip_path, out_dir):
    print(f"\nProcessing {seq_name} …")

    with zipfile.ZipFile(inner_zip_path) as zf:

        # ---- load CSV metadata ----
        def _parse_cam_csv(name):
            lines = zf.read(name).decode().strip().splitlines()
            return [(int(l.split(",")[0]), l.split(",")[1].strip())
                    for l in lines if not l.startswith("#")]

        cam0_entries = _parse_cam_csv("mav0/cam0/data.csv")
        cam1_entries = _parse_cam_csv("mav0/cam1/data.csv")

        imu_lines = zf.read("mav0/imu0/data.csv").decode().strip().splitlines()
        imu_entries = []
        for line in imu_lines:
            if line.startswith("#"):
                continue
            p = line.split(",")
            imu_entries.append((int(p[0]), [float(x) for x in p[1:]]))

        gt_text  = zf.read("mav0/state_groundtruth_estimate0/data.csv").decode()
        gt_data  = _parse_gt_csv(gt_text)
        gt_times = [g[0] for g in gt_data]
        gt_ts0   = gt_data[0][0]
        gt_ts1   = gt_data[-1][0]

        # ---- select camera frames within GT time range ----
        cam0_valid = [(ts, fn) for ts, fn in cam0_entries if gt_ts0 <= ts <= gt_ts1]
        cam1_valid = [(ts, fn) for ts, fn in cam1_entries if gt_ts0 <= ts <= gt_ts1]

        num_frames = min(len(cam0_valid), len(cam1_valid))
        if len(cam0_valid) != len(cam1_valid):
            print(f"  WARNING: cam0={len(cam0_valid)}, cam1={len(cam1_valid)}"
                  f" — using {num_frames}")
        cam0_valid = cam0_valid[:num_frames]
        cam1_valid = cam1_valid[:num_frames]
        print(f"  {num_frames} valid frames")

        # ---- output directories ----
        seq_dir  = out_dir / seq_name
        cam0_dir = seq_dir / "00"
        cam1_dir = seq_dir / "01"
        cam0_dir.mkdir(parents=True, exist_ok=True)
        cam1_dir.mkdir(parents=True, exist_ok=True)

        # ---- copy and rename images ----
        print("  extracting images …")
        for i, (_, orig_fn) in enumerate(cam0_valid):
            (cam0_dir / f"l.{i:06d}.png").write_bytes(
                zf.read(f"mav0/cam0/data/{orig_fn}"))

        for i, (_, orig_fn) in enumerate(cam1_valid):
            (cam1_dir / f"r.{i:06d}.png").write_bytes(
                zf.read(f"mav0/cam1/data/{orig_fn}"))

        # ---- frame_metadata.jsonl (compact JSON, integer timestamps) ----
        # Both cameras share the cam0 timestamp (canonical stereo pair timestamp).
        meta_lines = []
        for i, (ts0, _) in enumerate(cam0_valid):
            entry = {
                "frame_id": i,
                "cams": [
                    {"id": 0, "filename": f"00/l.{i:06d}.png", "timestamp": ts0},
                    {"id": 1, "filename": f"01/r.{i:06d}.png", "timestamp": ts0},
                ],
            }
            meta_lines.append(json.dumps(entry, separators=(",", ":")))
        (seq_dir / "frame_metadata.jsonl").write_text("\n".join(meta_lines) + "\n")

        # ---- IMU.jsonl — entries strictly inside GT time range (open interval) ----
        imu_out = []
        for ts, vals in imu_entries:
            if ts <= gt_ts0 or ts >= gt_ts1:
                continue
            wx, wy, wz, ax, ay, az = vals
            rec = {
                "AngularVelocityX":    wx,
                "AngularVelocityY":    wy,
                "AngularVelocityZ":    wz,
                "LinearAccelerationX": ax,
                "LinearAccelerationY": ay,
                "LinearAccelerationZ": az,
                "timestamp":           float(ts),
                "type":                "imu_data",
            }
            imu_out.append(json.dumps(rec))
        # No trailing newline (matches reference)
        (seq_dir / "IMU.jsonl").write_text("\n".join(imu_out))

        # ---- stereo.edex ----
        (seq_dir / "stereo.edex").write_text(_make_edex(num_frames - 1))

        # ---- gt.txt — cam0 poses relative to first frame ----
        R0, t0    = _cam0_world_pose(gt_data, gt_times, cam0_valid[0][0])
        R0inv, t0inv = _inv34(R0, t0)

        _IDENTITY_ROW = ("1.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 "
                         "0.000000e+00 1.000000e+00 0.000000e+00 0.000000e+00 "
                         "0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00")
        gt_rows = []
        for k, (cam_ts, _) in enumerate(cam0_valid):
            if k == 0:
                gt_rows.append(_IDENTITY_ROW)
                continue
            Ri, ti   = _cam0_world_pose(gt_data, gt_times, cam_ts)
            R_rel, t_rel = _mul34(R0inv, t0inv, Ri, ti)
            vals = (R_rel[0] + [t_rel[0]] +
                    R_rel[1] + [t_rel[1]] +
                    R_rel[2] + [t_rel[2]])
            gt_rows.append(" ".join(f"{v:.6e}" for v in vals))
        (seq_dir / "gt.txt").write_text("\n".join(gt_rows) + "\n")

    print(f"  done → {seq_dir}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def convert(raw_dir: Path, out_dir: Path, sequences: list[str] | None = None):
    selected = set(sequences) if sequences else set(ALL_SEQS)
    unknown = selected - set(ALL_SEQS)
    if unknown:
        sys.exit(f"ERROR: unknown sequence(s): {', '.join(sorted(unknown))}")

    out_dir.mkdir(parents=True, exist_ok=True)
    converted: list[str] = []

    with tempfile.TemporaryDirectory(prefix="euroc_convert_") as tmp:
        tmp_path = Path(tmp)

        for outer_name, seqs in _OUTER_ZIPS:
            seqs_to_convert = [s for s in seqs if s in selected]
            if not seqs_to_convert:
                continue

            outer_path = raw_dir / outer_name
            if not outer_path.exists():
                sys.exit(f"ERROR: {outer_path} not found (needed for {', '.join(seqs_to_convert)})")

            group = outer_name.replace(".zip", "")
            print(f"\n=== Opening {outer_name} ===")

            with zipfile.ZipFile(outer_path) as outer_zip:
                for seq in seqs_to_convert:
                    inner_zip_entry = f"{group}/{seq}/{seq}.zip"
                    inner_zip_path  = tmp_path / f"{seq}.zip"

                    print(f"  extracting inner zip for {seq} …")
                    with outer_zip.open(inner_zip_entry) as src, \
                         open(inner_zip_path, "wb") as dst:
                        shutil.copyfileobj(src, dst)

                    _convert_sequence(seq, inner_zip_path, out_dir)
                    converted.append(seq)
                    inner_zip_path.unlink()

    if not converted:
        sys.exit("ERROR: no sequences converted")

    print("\nGenerating config files …")
    _write_configs(out_dir, converted)
    print(f"\nDone.  Output written to: {out_dir}")


if __name__ == "__main__":
    _script_dir = Path(__file__).resolve().parent
    _repo_root  = _script_dir.parents[2]   # tools/datasets/euroc → repo root

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "raw_dir",
        nargs="?",
        default=str(_repo_root / "datasets" / "euroc" / "raw"),
        help="directory containing EuRoC zip archives",
    )
    parser.add_argument(
        "out_dir",
        nargs="?",
        default=str(_repo_root / "datasets" / "euroc" / "converted"),
        help="directory for converted sequences and reporter configs",
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        metavar="SEQ",
        help=f"convert only these sequences (default: all). Known: {', '.join(ALL_SEQS)}",
    )
    args = parser.parse_args()

    raw = Path(args.raw_dir)
    out = Path(args.out_dir)

    print(f"RAW dir : {raw}")
    print(f"OUT dir : {out}")
    if args.sequences:
        print(f"SEQ     : {', '.join(args.sequences)}")
    print()

    convert(raw, out, args.sequences)
