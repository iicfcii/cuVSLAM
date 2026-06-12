#!/usr/bin/env bash
# Convert CODa per-sequence zip archives to cuVSLAM format.
#
# CODa is not redistributable — see download_coda.sh for how to obtain the zips.
#
# Usage: prepare_coda.sh [OPTIONS]
#
# Options:
#   --raw-dir DIR        Directory holding the CODa zips.       Default: <repo>/datasets/coda/raw
#   --output-dir DIR     Directory for converted data.          Default: <repo>/datasets/converted
#   --force-download     Accepted for parity with prepare_kitti.sh; has no effect.
#   --download-only      Check for zips but skip conversion.
#   -h, --help           Show this help.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd -P)"

raw_dir="${repo_root}/datasets/coda/raw"
output_dir="${repo_root}/datasets/converted"
force_download=0
download_only=0

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
bash "${script_dir}/download_coda.sh" "${download_args[@]}"

[[ "${download_only}" -eq 1 ]] && exit 0

echo ""
python3 "${script_dir}/convert_coda.py" "${raw_dir}" "${output_dir}"
