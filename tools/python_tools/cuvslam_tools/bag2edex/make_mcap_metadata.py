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

import argparse
import re
import subprocess
from pathlib import Path


def ns_from_seconds(seconds: str) -> int:
    return int(round(float(seconds) * 1_000_000_000))


def ns_from_epoch(sec: str, nsec: str) -> int:
    return int(sec) * 1_000_000_000 + int(nsec.ljust(9, "0")[:9])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mcap", help="Path to .mcap file")
    parser.add_argument("--version", type=int, default=9)
    args = parser.parse_args()

    mcap_path = Path(args.mcap).resolve()
    bag_dir = mcap_path.parent
    metadata_path = bag_dir / "metadata.yaml"

    result = subprocess.run(
        ["ros2", "bag", "info", "--storage", "mcap", str(mcap_path)],
        check=True,
        text=True,
        capture_output=True,
    )

    text = result.stdout

    duration = re.search(r"Duration:\s+([0-9.]+)s", text)
    start = re.search(r"Start:.*\((\d+)\.(\d+)\)", text)
    messages = re.search(r"Messages:\s+(\d+)", text)

    if not duration or not start or not messages:
        raise RuntimeError("Could not parse ros2 bag info output")

    duration_ns = ns_from_seconds(duration.group(1))
    start_ns = ns_from_epoch(start.group(1), start.group(2))
    total_messages = int(messages.group(1))

    topic_re = re.compile(
        r"Topic:\s+(.+?)\s+\|\s+Type:\s+(.+?)\s+\|\s+Count:\s+(\d+)\s+\|\s+Serialization Format:\s+(\S+)"
    )

    topics = topic_re.findall(text)

    if not topics:
        raise RuntimeError("No topics found in ros2 bag info output")

    rel_file = mcap_path.name

    yaml = []
    yaml.append("rosbag2_bagfile_information:")
    yaml.append(f"  version: {args.version}")
    yaml.append("  storage_identifier: mcap")
    yaml.append("  duration:")
    yaml.append(f"    nanoseconds: {duration_ns}")
    yaml.append("  starting_time:")
    yaml.append(f"    nanoseconds_since_epoch: {start_ns}")
    yaml.append(f"  message_count: {total_messages}")
    yaml.append("  relative_file_paths:")
    yaml.append(f"    - {rel_file}")
    yaml.append("  files:")
    yaml.append(f"    - path: {rel_file}")
    yaml.append("      starting_time:")
    yaml.append(f"        nanoseconds_since_epoch: {start_ns}")
    yaml.append("      duration:")
    yaml.append(f"        nanoseconds: {duration_ns}")
    yaml.append(f"      message_count: {total_messages}")
    yaml.append("  topics_with_message_count:")

    for name, msg_type, count, serialization in topics:
        yaml.append("    - topic_metadata:")
        yaml.append(f"        name: {name}")
        yaml.append(f"        type: {msg_type}")
        yaml.append(f"        serialization_format: {serialization}")
        yaml.append('        offered_qos_profiles: ""')
        yaml.append(f"      message_count: {count}")

    yaml.append('  compression_format: ""')
    yaml.append('  compression_mode: ""')
    yaml.append("  custom_data: {}")
    yaml.append("")

    metadata_path.write_text("\n".join(yaml))
    print(f"Wrote: {metadata_path}")


if __name__ == "__main__":
    main()
