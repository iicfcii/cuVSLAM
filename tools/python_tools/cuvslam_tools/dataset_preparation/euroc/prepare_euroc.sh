#!/usr/bin/env bash
# Download and convert the official EuRoC MAV sequence bundles to the portable
# cuVSLAM reporter layout.
#
# Usage: prepare_euroc.sh [OPTIONS]
#
# Options:
#   --raw-dir DIR        Directory for raw archives.   Default: <repo>/datasets/euroc/raw
#   --output-dir DIR     Directory for converted data. Default: <repo>/datasets/converted
#   --sequences SEQ...   Convert an explicit sequence subset. The default is all
#                        11 official sequences.
#   --force-download     Re-download archives even when they already exist.
#   --download-only      Download archives but skip conversion.
#   -h, --help           Show this help.
#
# The converted root contains copied media, EDEX/metadata/ground truth files,
# smoke/full reporter configs, and deterministic dataset metadata.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "${script_dir}/../../../../.." && pwd -P)"

raw_dir="${repo_root}/datasets/euroc/raw"
output_dir="${repo_root}/datasets/converted"
force_download=0
download_only=0
sequences=()

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
        --sequences)
            shift
            if [[ $# -eq 0 || "$1" == -* ]]; then
                echo "error: --sequences requires at least one sequence" >&2
                exit 2
            fi
            while [[ $# -gt 0 && "$1" != -* ]]; do
                sequences+=("$1")
                shift
            done ;;
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

if [[ ${#sequences[@]} -gt 0 ]]; then
    declare -A required_archives=()
    for sequence in "${sequences[@]}"; do
        case "${sequence}" in
            MH_01_easy|MH_02_easy|MH_03_medium|MH_04_difficult|MH_05_difficult)
                required_archives["machine_hall.zip"]=1 ;;
            V1_01_easy|V1_02_medium|V1_03_difficult)
                required_archives["vicon_room1.zip"]=1 ;;
            V2_01_easy|V2_02_medium|V2_03_difficult)
                required_archives["vicon_room2.zip"]=1 ;;
            *)
                echo "error: unknown EuRoC sequence '${sequence}'" >&2
                exit 2 ;;
        esac
    done
    for archive in machine_hall.zip vicon_room1.zip vicon_room2.zip; do
        if [[ -n "${required_archives[${archive}]:-}" ]]; then
            download_args+=(--archive "${archive}")
        fi
    done
fi

bash "${script_dir}/download_euroc.sh" "${download_args[@]}"

[[ "${download_only}" -eq 1 ]] && exit 0

dataset_dir="${output_dir}/euroc"
converter_args=("${raw_dir}" "${dataset_dir}")
if [[ ${#sequences[@]} -gt 0 ]]; then
    converter_args+=(--sequences "${sequences[@]}")
fi

echo ""
echo "Converting EuRoC data to ${dataset_dir} …"
"${PYTHON_BIN:-python3}" "${script_dir}/convert_euroc.py" "${converter_args[@]}"

selected_sequences=("${sequences[@]}")
if [[ ${#selected_sequences[@]} -eq 0 ]]; then
    selected_sequences=(
        MH_01_easy MH_02_easy MH_03_medium MH_04_difficult MH_05_difficult
        V1_01_easy V1_02_medium V1_03_difficult
        V2_01_easy V2_02_medium V2_03_difficult
    )
fi

for artifact in \
    dataset_metadata.json \
    euroc-vio.cfg \
    euroc-slam.cfg \
    euroc-vio_slam.cfg; do
    if [[ ! -s "${dataset_dir}/${artifact}" ]]; then
        echo "error: converter did not produce ${dataset_dir}/${artifact}" >&2
        exit 1
    fi
done

for sequence in "${selected_sequences[@]}"; do
    for artifact in \
        stereo.edex \
        frame_metadata.jsonl \
        IMU.jsonl \
        gt.txt \
        00/l.000000.png \
        01/r.000000.png; do
        if [[ ! -s "${dataset_dir}/${sequence}/${artifact}" ]]; then
            echo "error: converter did not produce ${dataset_dir}/${sequence}/${artifact}" >&2
            exit 1
        fi
    done
done

echo ""
echo "done — portable EuRoC dataset ready at ${dataset_dir}"
