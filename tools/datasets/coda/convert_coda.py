#!/usr/bin/env python3
"""Convert CODa (UT Campus Object Dataset) per-sequence zip archives to cuVSLAM converted format.

Reads zip entries directly via the `zipfile` module — does NOT extract archives to disk.

Input (raw/):
  0.zip, 1.zip, …, 22.zip   — one per CODa sequence (user supplies; see download_coda.sh)

Output (converted/):
  <seq>/00/<seq>.0.XXXXX.png   — left rectified camera images, 1-indexed
  <seq>/01/<seq>.1.XXXXX.png   — right rectified camera images, 1-indexed
  <seq>/stereo.edex            — calibration in cuVSLAM edex format
  <seq>/gt.txt                 — ground-truth poses, 12-float row-major SE(3), cam0 frame
  coda-slam_gt.cfg             — reporter config: SLAM mode, all sequences
  coda-vio_gt.cfg              — reporter config: VIO mode, all sequences
  coda-vio_slam.cfg            — reporter config: both modes, all sequences
  coda-vio_slam_gt.cfg         — alias for coda-vio_slam.cfg (all sequences have GT)

Usage:
  python3 convert_coda.py [RAW_DIR] [OUT_DIR]

  RAW_DIR  defaults to the directory containing this script
  OUT_DIR  defaults to <parent_of_RAW_DIR>/converted
"""

import re
import sys
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# CODa-specific constants
# ---------------------------------------------------------------------------
GT_PREFERRED = "poses/dense_global"
GT_FALLBACK = "poses/dense"
# Sequences for which CODa only ships poses/dense/ (no globally optimized
# poses/dense_global/); see DATA_REPORT.md "Poses".
DENSE_ONLY_SEQS = {8, 14, 15}

# Filename width for output PNGs. CODa's longest sequence (#21) has ~21k
# frames at 10 Hz, so 5 digits are required (KITTI uses 4).
FRAME_WIDTH = 5


# ---------------------------------------------------------------------------
# Minimal targeted YAML reader (CODa calib files are a fixed, simple shape)
# ---------------------------------------------------------------------------

def _extract_data_array(text, parent_key):
    """Return the float list under '<parent_key>: ... data: [...]' (handles multi-line list)."""
    lines = text.splitlines()
    i = 0
    in_section = False
    while i < len(lines):
        line = lines[i]
        if re.match(r'^' + re.escape(parent_key) + r':\s*$', line):
            in_section = True
            i += 1
            continue
        if in_section and line and not line[0].isspace():
            in_section = False
        if in_section:
            m = re.match(r'^\s+data:\s*\[(.*)$', line)
            if m:
                buf = m.group(1)
                while ']' not in buf and i + 1 < len(lines):
                    i += 1
                    buf += ' ' + lines[i].strip()
                inner = buf[: buf.index(']')]
                return [float(x.strip()) for x in inner.split(',') if x.strip()]
        i += 1
    return None


def _extract_int(text, key):
    m = re.search(r'^' + re.escape(key) + r':\s*(\d+)\s*$', text, re.M)
    if not m:
        return None
    return int(m.group(1))


# ---------------------------------------------------------------------------
# Small linalg helpers (avoid numpy/scipy dependency)
# ---------------------------------------------------------------------------

def quat_to_rotation(qw, qx, qy, qz):
    """Return 3x3 rotation matrix from quaternion (qw, qx, qy, qz)."""
    n = (qw * qw + qx * qx + qy * qy + qz * qz) ** 0.5
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return [
        [1 - 2 * (qy * qy + qz * qz),     2 * (qx * qy - qz * qw),     2 * (qx * qz + qy * qw)],
        [    2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz),     2 * (qy * qz - qx * qw)],
        [    2 * (qx * qz - qy * qw),     2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ]


def mat4_mul(A, B):
    out = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            out[i][j] = A[i][0] * B[0][j] + A[i][1] * B[1][j] + A[i][2] * B[2][j] + A[i][3] * B[3][j]
    return out


def mat4_inv_rigid(M):
    """Inverse of a rigid 4x4 transform [R t; 0 1]:  R^T, -R^T t."""
    R_inv = [[M[j][i] for j in range(3)] for i in range(3)]
    t = [M[i][3] for i in range(3)]
    t_inv = [-(R_inv[i][0] * t[0] + R_inv[i][1] * t[1] + R_inv[i][2] * t[2]) for i in range(3)]
    return [
        [R_inv[0][0], R_inv[0][1], R_inv[0][2], t_inv[0]],
        [R_inv[1][0], R_inv[1][1], R_inv[1][2], t_inv[1]],
        [R_inv[2][0], R_inv[2][1], R_inv[2][2], t_inv[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


# ---------------------------------------------------------------------------
# Edex template
# ---------------------------------------------------------------------------

_EDEX_TEMPLATE = """\
[
    {{
        "cameras": [
            {{
                "intrinsics": {{
                    "distortion_model": "pinhole",
                    "distortion_params": [],
                    "focal": [{focal_x}, {focal_y}],
                    "principal": [{cx}, {cy}],
                    "size": [{width}, {height}]
                }},
                "transform": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0]
                ]
            }},
            {{
                "intrinsics": {{
                    "distortion_model": "pinhole",
                    "distortion_params": [],
                    "focal": [{focal_x}, {focal_y}],
                    "principal": [{cx}, {cy}],
                    "size": [{width}, {height}]
                }},
                "transform": [
                    [1.0, 0.0, 0.0, {baseline}],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0]
                ]
            }}
        ],
        "frame_end": {frame_end},
        "frame_start": 1,
        "version": "0.9"
    }},
    {{
        "fps": 10,
        "points2d": {{}},
        "points3d": {{}},
        "rig_positions": {{}},
        "sequence": [["00/{left_first}"], ["01/{right_first}"]]
    }}
]"""


def make_edex(seq, fx, fy, cx, cy, w, h, baseline, num_frames):
    first = f"{1:0{FRAME_WIDTH}d}"
    return _EDEX_TEMPLATE.format(
        focal_x=repr(fx),
        focal_y=repr(fy),
        cx=repr(cx),
        cy=repr(cy),
        width=w,
        height=h,
        baseline=repr(baseline),
        frame_end=num_frames,
        left_first=f"{seq}.0.{first}.png",
        right_first=f"{seq}.1.{first}.png",
    )


# ---------------------------------------------------------------------------
# Reporter cfg generation (mirrors tools/datasets/kitti/convert_kitti.py)
# ---------------------------------------------------------------------------

def _json_val(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    return f'"{v}"'


def _seq_cfg_entry(seq, mode, has_gt):
    is_slam = (mode == "slam")
    label = "SLAM" if is_slam else "ODOM"
    fields = [
        ("enable", True),
        ("sequence_folder", str(seq)),
        ("edex_file", "stereo.edex"),
        ("precompute_2d_tracks", False),
        ("precompute_key_frames", False),
        ("use_gt_scale", False),
        ("sequence_title", f"CODa-{int(seq):02d}-{label}"),
    ]
    if has_gt:
        fields.append(("gt_file_path", "gt.txt"))
    if is_slam:
        fields.append(("use_slam", True))
    return fields


def _format_cfg(sequences, gt_seqs, slam, odom):
    lines = ["{"]
    lines.append('    "version": "0.1",')
    lines.append('    "write_cache": false,')
    lines.append('    "use_cuda": false,')
    lines.append('    "dataset_folder": "coda/",')
    lines.append('    "use_icp_scaling": false,')
    lines.append('    "segment_lengths": [100, 200, 300, 400, 500, 600, 700, 800],')
    lines.append('    "sequence_cfgs": [')

    entries = []
    for seq in sequences:
        has_gt = seq in gt_seqs
        if odom:
            entries.append(_seq_cfg_entry(seq, "odom", has_gt))
        if slam:
            entries.append(_seq_cfg_entry(seq, "slam", has_gt))

    for ei, fields in enumerate(entries):
        lines.append("        {")
        for fi, (k, v) in enumerate(fields):
            comma = "," if fi < len(fields) - 1 else ""
            lines.append(f'            "{k}": {_json_val(v)}{comma}')
        comma = "," if ei < len(entries) - 1 else ""
        lines.append(f"        }}{comma}")

    lines.append("  ]")
    lines.append("}")
    lines.append("")  # trailing newline
    return "\n".join(lines)


def write_configs(out_dir, sequences, gt_sequences):
    gt_set = set(gt_sequences)
    gt_only = [s for s in sequences if s in gt_set]
    configs = {
        "coda-slam_gt.cfg":     _format_cfg(gt_only,   gt_set, slam=True,  odom=False),
        "coda-vio_gt.cfg":      _format_cfg(gt_only,   gt_set, slam=False, odom=True),
        "coda-vio_slam.cfg":    _format_cfg(sequences, gt_set, slam=True,  odom=True),
        "coda-vio_slam_gt.cfg": _format_cfg(gt_only,   gt_set, slam=True,  odom=True),
    }
    for name, text in configs.items():
        (out_dir / name).write_text(text)
        print(f"  wrote {name}")


# ---------------------------------------------------------------------------
# Per-sequence conversion (streams everything out of one zip)
# ---------------------------------------------------------------------------

def _gt_path_for(zf, seq):
    primary = f"{GT_PREFERRED}/{seq}.txt"
    fallback = f"{GT_FALLBACK}/{seq}.txt"
    names = set(zf.namelist())
    if int(seq) not in DENSE_ONLY_SEQS and primary in names:
        return primary
    if fallback in names:
        return fallback
    return None


def convert_sequence(zip_path, out_root):
    seq = zip_path.stem  # e.g. "0", "5"
    seq_dst = out_root / seq
    cam0_dst = seq_dst / "00"
    cam1_dst = seq_dst / "01"
    cam0_dst.mkdir(parents=True, exist_ok=True)
    cam1_dst.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        cam0_yaml = zf.read(f"calibrations/{seq}/calib_cam0_intrinsics.yaml").decode()
        cam1_yaml = zf.read(f"calibrations/{seq}/calib_cam1_intrinsics.yaml").decode()
        os1_to_cam0_yaml = zf.read(f"calibrations/{seq}/calib_os1_to_cam0.yaml").decode()

        P0 = _extract_data_array(cam0_yaml, "projection_matrix")
        if P0 is None or len(P0) < 8:
            sys.exit(f"ERROR: seq {seq}: cannot parse cam0 projection_matrix")
        fx, fy = P0[0], P0[5]
        cx, cy = P0[2], P0[6]
        width = _extract_int(cam0_yaml, "image_width")
        height = _extract_int(cam0_yaml, "image_height")

        # Baseline from cam0 disparity_matrix Q:  Q[3,2] = -1/Tx  (data[14] in row-major)
        # See DATA_REPORT.md "Calibration Files" and OpenCV stereoRectify docs.
        # We trust Q over P[0,3] because CODa's projection_matrix appears to encode
        # Tx*fx in inconsistent units (off by ~1000x vs the cam-to-cam extrinsic),
        # whereas the disparity_matrix is consistent with the physical baseline.
        Q = _extract_data_array(cam0_yaml, "disparity_matrix")
        if Q is None or len(Q) < 15:
            sys.exit(f"ERROR: seq {seq}: cannot parse cam0 disparity_matrix")
        if abs(Q[14]) < 1e-9:
            sys.exit(f"ERROR: seq {seq}: cam0 disparity_matrix has zero/invalid Q[14] "
                     f"(={Q[14]}); cannot derive baseline")
        baseline = abs(1.0 / Q[14])

        ext_data = _extract_data_array(os1_to_cam0_yaml, "extrinsic_matrix")
        if ext_data is None or len(ext_data) != 16:
            sys.exit(f"ERROR: seq {seq}: cannot parse os1_to_cam0 extrinsic_matrix")
        T_cam0_from_os1 = [ext_data[i * 4:(i + 1) * 4] for i in range(4)]
        T_os1_from_cam0 = mat4_inv_rigid(T_cam0_from_os1)

        # Index images
        cam0_re = re.compile(r'^2d_rect/cam0/' + re.escape(seq) +
                             r'/2d_rect_cam0_' + re.escape(seq) + r'_(\d+)\.png$')
        cam1_re = re.compile(r'^2d_rect/cam1/' + re.escape(seq) +
                             r'/2d_rect_cam1_' + re.escape(seq) + r'_(\d+)\.png$')
        cam0_map, cam1_map = {}, {}
        for info in zf.infolist():
            m = cam0_re.match(info.filename)
            if m:
                cam0_map[int(m.group(1))] = info.filename
                continue
            m = cam1_re.match(info.filename)
            if m:
                cam1_map[int(m.group(1))] = info.filename

        frame_ids = sorted(set(cam0_map) & set(cam1_map))
        if len(frame_ids) != len(cam0_map) or len(frame_ids) != len(cam1_map):
            print(f"  WARNING: seq {seq}: {len(cam0_map)} cam0 / {len(cam1_map)} cam1 / "
                  f"{len(frame_ids)} paired — using paired only")

        if not frame_ids:
            sys.exit(f"ERROR: seq {seq}: no paired stereo frames found in {zip_path}")

        # Read GT (if present) up front so we can drop any frame whose pose row
        # is out of range *before* writing images — this keeps image-pair count
        # and gt.txt row count strictly 1:1, which the reporter assumes.
        gt_zip_path = _gt_path_for(zf, seq)
        poses_lines = None
        if gt_zip_path is None:
            print(f"  WARNING: seq {seq}: no GT pose file found, skipping gt.txt")
        else:
            poses_lines = zf.read(gt_zip_path).decode().strip().splitlines()
            print(f"  reading GT from {gt_zip_path} ({len(poses_lines)} rows)")
            dropped = [f for f in frame_ids if f >= len(poses_lines)]
            if dropped:
                print(f"  WARNING: seq {seq}: {len(dropped)} frame(s) out of pose range — "
                      f"dropping from both image and GT output")
                frame_ids = [f for f in frame_ids if f < len(poses_lines)]
                if not frame_ids:
                    sys.exit(f"ERROR: seq {seq}: all frames out of pose range")

        num_frames = len(frame_ids)
        print(f"Sequence {seq}: {num_frames} frames, {width}x{height}, "
              f"fx={fx:.4f}, fy={fy:.4f}, cx={cx:.4f}, cy={cy:.4f}, baseline={baseline:.6f} m")

        # Stream images: rename to 1-indexed sequential filenames
        for idx, frame in enumerate(frame_ids, start=1):
            (cam0_dst / f"{seq}.0.{idx:0{FRAME_WIDTH}d}.png").write_bytes(zf.read(cam0_map[frame]))
            (cam1_dst / f"{seq}.1.{idx:0{FRAME_WIDTH}d}.png").write_bytes(zf.read(cam1_map[frame]))

        # Ground truth poses: CODa stores T_osWorld_from_os1(t) in the LiDAR FLU
        # frame.  cuVSLAM tracks in the cam0(0) RDF frame with the first pose
        # pinned to identity (KITTI convention), so we re-ground via
        #   T_cam0(0)_from_cam0(t) = inv(T_osWorld_from_cam0(0)) @ T_osWorld_from_cam0(t).
        has_gt = poses_lines is not None
        if has_gt:
            poses_cam0 = []
            for frame in frame_ids:
                row = poses_lines[frame].split()
                # ts x y z qw qx qy qz
                x = float(row[1]); y = float(row[2]); z = float(row[3])
                qw = float(row[4]); qx = float(row[5]); qy = float(row[6]); qz = float(row[7])
                R = quat_to_rotation(qw, qx, qy, qz)
                T_world_from_os1 = [
                    [R[0][0], R[0][1], R[0][2], x],
                    [R[1][0], R[1][1], R[1][2], y],
                    [R[2][0], R[2][1], R[2][2], z],
                    [0.0,    0.0,    0.0,    1.0],
                ]
                poses_cam0.append(mat4_mul(T_world_from_os1, T_os1_from_cam0))

            T0_inv = mat4_inv_rigid(poses_cam0[0])
            gt_out_lines = []
            for pose in poses_cam0:
                rel = mat4_mul(T0_inv, pose)
                vals = rel[0][:4] + rel[1][:4] + rel[2][:4]
                gt_out_lines.append(" ".join(repr(v) for v in vals))
            (seq_dst / "gt.txt").write_text("\n".join(gt_out_lines) + "\n")

        # stereo.edex
        edex_text = make_edex(seq, fx, fy, cx, cy, width, height, baseline, num_frames)
        (seq_dst / "stereo.edex").write_text(edex_text)

        return has_gt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def convert(raw_dir, out_dir):
    zips = sorted(raw_dir.glob("*.zip"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
    if not zips:
        sys.exit(f"ERROR: no *.zip archives found in {raw_dir}\n"
                 f"       CODa requires registration on the Texas Dataverse — see download_coda.sh.")

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(zips)} CODa sequence zip(s):", ", ".join(z.name for z in zips), "\n")

    sequences = []
    gt_sequences = []
    for zp in zips:
        if not zp.stem.isdigit():
            print(f"  skipping {zp.name}: filename is not a sequence number")
            continue
        if convert_sequence(zp, out_dir):
            gt_sequences.append(zp.stem)
        sequences.append(zp.stem)

    print("\nGenerating config files …")
    write_configs(out_dir, sequences, gt_sequences)
    print(f"\nDone.  Output written to: {out_dir}")


if __name__ == "__main__":
    _script_dir = Path(__file__).resolve().parent
    _repo_root = _script_dir.parents[2]   # tools/datasets/coda → repo root

    args = sys.argv[1:]
    raw = Path(args[0]) if len(args) >= 1 else _script_dir
    out = Path(args[1]) if len(args) >= 2 else _repo_root / "datasets" / "converted"

    print(f"RAW dir : {raw}")
    print(f"OUT dir : {out}\n")
    convert(raw, out)
