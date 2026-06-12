#!/usr/bin/env bash
# Check for CODa sequence zip archives.
#
# CODa is not redistributable: every user must accept the dataset license on
# the Texas Dataverse and download the per-sequence archives manually.  This
# script does NOT download anything — it only verifies that the user has
# already placed the zips in OUT_DIR and prints a registration walkthrough
# otherwise.  The interface mirrors download_kitti.sh so prepare_coda.sh can
# call it with the same flags.
#
# Usage: download_coda.sh [OPTIONS] [OUT_DIR]
#
#   OUT_DIR        Directory holding the CODa sequence zips.
#                  Defaults to <repo_root>/datasets/coda/raw
#   --force        Accepted for parity with download_kitti.sh; has no effect.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd -P)"
out_dir="${repo_root}/datasets/coda/raw"
force=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force) force=1; shift ;;
        -*) echo "error: unknown option '$1'" >&2; exit 2 ;;
        *)  out_dir="$1"; shift ;;
    esac
done

if [[ "${force}" -eq 1 ]]; then
    echo "note: --force has no effect — CODa zips are not auto-downloaded"
fi

shopt -s nullglob
zips=("${out_dir}"/*.zip)
shopt -u nullglob

if [[ ${#zips[@]} -gt 0 ]]; then
    echo "Found ${#zips[@]} CODa sequence zip(s) in ${out_dir} — skipping download."
    exit 0
fi

cat >&2 <<EOF
ERROR: CODa requires registration on the Texas Dataverse, so the user supplies the zips manually.

  1. Register and accept the dataset license at:
       https://dataverse.tdl.org/dataset.xhtml?persistentId=doi:10.18738/T8/BBOQMV
  2. Download the per-sequence archives (0.zip, 1.zip, ..., 22.zip) you want.
  3. Place them in: ${out_dir}
  4. Re-run this script (or run convert_coda.py directly).
EOF
exit 1
