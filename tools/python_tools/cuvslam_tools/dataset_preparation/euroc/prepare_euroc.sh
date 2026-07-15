#!/usr/bin/env bash
# Download EuRoC MH_01_easy and convert it to the cuVSLAM reporter layout.
#
# Usage: prepare_euroc.sh [OPTIONS]
#
# Options:
#   --raw-dir DIR        Directory for raw archives.   Default: <repo>/datasets/euroc/raw
#   --output-dir DIR     Directory for converted data. Default: <repo>/datasets/converted
#   --force-download     Re-download archives even when they already exist.
#   --download-only      Download archives but skip dataset layout.
#   -h, --help           Show this help.
#
# Provisioning uploads ${output_dir}/euroc/ to S3. convert_euroc.py writes copied
# images, stereo.edex, and reporter configs (euroc-vio_slam.cfg, etc.) into that tree.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "${script_dir}/../../../../.." && pwd -P)"

raw_dir="${repo_root}/datasets/euroc/raw"
output_dir="${repo_root}/datasets/converted"
force_download=0
download_only=0
seq_name="MH_01_easy"

usage() {
    sed -n '2,/^$/p' "$0" | grep '^#' | sed 's/^# \?//'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --raw-dir)
            [[ $# -lt 2 ]] && { echo "error: --raw-dir requires a value" >&2; exit 2; }
            raw_dir="$2"; shift 2 ;;
        --output-dir)
            [[ $# -lt 2 ]] && { echo "error: --output-dir requires a value" >&2; exit 2; }
            output_dir="$2"; shift 2 ;;
        --force-download)
            force_download=1; shift ;;
        --download-only)
            download_only=1; shift ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            echo "error: unknown option '$1'" >&2; exit 2 ;;
    esac
done

echo "Raw dir    : ${raw_dir}"
echo "Output dir : ${output_dir}"
echo ""

download_args=("${raw_dir}")
[[ "${force_download}" -eq 1 ]] && download_args+=(--force)
bash "${script_dir}/download_euroc.sh" "${download_args[@]}"

[[ "${download_only}" -eq 1 ]] && exit 0

dataset_dir="${output_dir}/euroc"
echo ""
echo "Converting ${seq_name} via convert_euroc.py …"
python3 "${script_dir}/convert_euroc.py" \
    --sequences "${seq_name}" \
    "${raw_dir}" "${dataset_dir}"

for artifact in \
    "euroc-vio_slam.cfg" \
    "${seq_name}/stereo.edex" \
    "${seq_name}/frame_metadata.jsonl" \
    "${seq_name}/IMU.jsonl" \
    "${seq_name}/gt.txt" \
    "${seq_name}/00/l.000000.png" \
    "${seq_name}/01/r.000000.png"; do
    if [[ ! -s "${dataset_dir}/${artifact}" ]]; then
        echo "error: convert_euroc did not produce ${dataset_dir}/${artifact}" >&2
        exit 1
    fi
done

echo ""
echo "done — dataset ready at ${dataset_dir}/${seq_name}"
