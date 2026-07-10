#!/usr/bin/env python3
"""Create cuvslam_app reporter inputs for the prepared TUM RGB-D sequence."""

import json
import re
import sys
from pathlib import Path


def parse_calibration(path: Path) -> dict:
    data = path.read_text()

    def list_value(name: str) -> list[float]:
        match = re.search(rf"^{re.escape(name)}:\s*\[([^\]]+)\]", data, re.MULTILINE)
        if not match:
            raise ValueError(f"missing {name} in {path}")
        return [float(v.strip()) for v in match.group(1).split(",")]

    def scalar_value(name: str) -> float:
        match = re.search(rf"^{re.escape(name)}:\s*([0-9.]+)", data, re.MULTILINE)
        if not match:
            raise ValueError(f"missing {name} in {path}")
        return float(match.group(1))

    return {
        "focal": list_value("  focal_length"),
        "principal": list_value("  principal_point"),
        "size": [int(scalar_value("  image_width")), int(scalar_value("  image_height"))],
        "depth_scale": scalar_value("  scale"),
    }


def read_timestamp_file(path: Path) -> list[tuple[float, str]]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        timestamp, filename = line.split()[:2]
        rows.append((float(timestamp), filename))
    return rows


def matched_pairs(rgb_rows: list[tuple[float, str]], depth_rows: list[tuple[float, str]]) -> list[tuple[float, str, float, str]]:
    pairs = []
    depth_idx = 0
    for rgb_ts, rgb_path in rgb_rows:
        while depth_idx + 1 < len(depth_rows) and abs(depth_rows[depth_idx + 1][0] - rgb_ts) <= abs(depth_rows[depth_idx][0] - rgb_ts):
            depth_idx += 1
        depth_ts, depth_path = depth_rows[depth_idx]
        if abs(depth_ts - rgb_ts) <= 0.02:
            pairs.append((rgb_ts, rgb_path, depth_ts, depth_path))
    return pairs


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <tum-sequence-dir>", file=sys.stderr)
        return 2

    sequence_dir = Path(sys.argv[1])
    calibration = parse_calibration(sequence_dir / "freiburg3_rig.yaml")
    pairs = matched_pairs(
        read_timestamp_file(sequence_dir / "rgb.txt"),
        read_timestamp_file(sequence_dir / "depth.txt"),
    )
    if not pairs:
        print("error: no synchronized RGB-D pairs found", file=sys.stderr)
        return 1

    stereo_edex = [
        {
            "cameras": [
                {
                    "intrinsics": {
                        "distortion_model": "pinhole",
                        "distortion_params": [],
                        "focal": calibration["focal"],
                        "principal": calibration["principal"],
                        "size": calibration["size"],
                    },
                    "transform": [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                    ],
                    "depth_id": 0,
                    "depth_scale_factor": 1.0 / calibration["depth_scale"],
                }
            ],
            "frame_start": 0,
            "frame_end": len(pairs) - 1,
        },
        {
            "fps": 30,
            "frame_metadata": "frame_metadata.jsonl",
        },
    ]
    (sequence_dir / "stereo.edex").write_text(json.dumps(stereo_edex, indent=4) + "\n")

    with (sequence_dir / "frame_metadata.jsonl").open("w") as out:
        for frame_id, (rgb_ts, rgb_path, depth_ts, depth_path) in enumerate(pairs):
            out.write(json.dumps({
                "frame_id": frame_id,
                "cams": [{
                    "id": 0,
                    "filename": rgb_path,
                    "timestamp": int(rgb_ts * 1_000_000_000),
                }],
                "depth": [{
                    "id": 0,
                    "filename": depth_path,
                    "timestamp": int(depth_ts * 1_000_000_000),
                }],
            }) + "\n")

    print(f"wrote stereo.edex and frame_metadata.jsonl for {len(pairs)} RGB-D pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
