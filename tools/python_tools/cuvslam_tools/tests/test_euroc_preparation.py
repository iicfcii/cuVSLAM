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

"""EuRoC bundle selection, plus the download script's cache and user-agent handling.

Selecting the wrong bundle means downloading 6-12 GB that is not needed, so the
sequence-to-archive mapping is worth pinning down.
"""

import io
import os
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from cuvslam_tools.dataset_preparation.euroc import convert_euroc
from cuvslam_tools.dataset_preparation.euroc import prepare as euroc_prepare

_MACHINE_HALL_MD5 = "363f5c2502b469cdd97ef85997714806"


def _download_script() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "dataset_preparation"
        / "euroc"
        / "download_euroc.sh"
    )


def _write_fake_md5sum(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    md5sum = directory / "md5sum"
    md5sum.write_text(
        "#!/usr/bin/env bash\n"
        f"printf '{_MACHINE_HALL_MD5}  %s\\n' \"$2\"\n"
    )
    md5sum.chmod(0o755)


class TestEurocBundleSelection(unittest.TestCase):
    def test_all_bundles_are_required_by_default(self):
        self.assertEqual(
            convert_euroc.required_archives(),
            ["machine_hall.zip", "vicon_room1.zip", "vicon_room2.zip"],
        )

    def test_each_sequence_group_maps_to_its_own_bundle(self):
        cases = {
            "MH_03_medium": ["machine_hall.zip"],
            "V1_02_medium": ["vicon_room1.zip"],
            "V2_03_difficult": ["vicon_room2.zip"],
        }
        for sequence, expected in cases.items():
            with self.subTest(sequence=sequence):
                self.assertEqual(convert_euroc.required_archives([sequence]), expected)

    def test_mixed_subset_keeps_bundle_order(self):
        self.assertEqual(
            convert_euroc.required_archives(["V2_01_easy", "MH_01_easy"]),
            ["machine_hall.zip", "vicon_room2.zip"],
        )

    def test_unknown_sequence_is_rejected(self):
        with self.assertRaisesRegex(convert_euroc.ConversionError, "unknown sequence"):
            convert_euroc.required_archives(["MH_99_easy"])

    def test_explicit_subset_downloads_only_its_bundles(self):
        with tempfile.TemporaryDirectory() as temporary:
            raw_dir = Path(temporary) / "raw"
            with mock.patch.object(euroc_prepare, "run_download_script") as download, redirect_stdout(
                io.StringIO()
            ):
                euroc_prepare.prepare(
                    raw_dir=raw_dir,
                    output_dir=Path(temporary) / "converted",
                    sequences=["MH_01_easy", "V1_01_easy"],
                    download_only=True,
                )

        self.assertEqual(
            download.call_args.args[1],
            [str(raw_dir), "--archive", "machine_hall.zip", "--archive", "vicon_room1.zip"],
        )

    def test_default_run_leaves_bundle_selection_to_the_download_script(self):
        with tempfile.TemporaryDirectory() as temporary:
            raw_dir = Path(temporary) / "raw"
            with mock.patch.object(euroc_prepare, "run_download_script") as download, redirect_stdout(
                io.StringIO()
            ):
                euroc_prepare.prepare(
                    raw_dir=raw_dir,
                    output_dir=Path(temporary) / "converted",
                    download_only=True,
                )

        self.assertEqual(download.call_args.args[1], [str(raw_dir)])


class TestEurocDownloadScript(unittest.TestCase):
    def test_corrupt_cached_archive_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            raw_dir = Path(temporary)
            (raw_dir / "machine_hall.zip").write_bytes(b"corrupt")

            completed = subprocess.run(
                ["bash", str(_download_script()), "--archive", "machine_hall.zip", str(raw_dir)],
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 1)
        self.assertIn("md5 mismatch for existing", completed.stderr)
        self.assertIn("re-run with --force", completed.stderr)

    def test_downloads_set_an_explicit_user_agent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake_bin = root / "bin"
            _write_fake_md5sum(fake_bin)
            curl = fake_bin / "curl"
            curl.write_text(
                "#!/usr/bin/env bash\n"
                "user_agent=\n"
                "output=\n"
                "while [[ $# -gt 0 ]]; do\n"
                "    case \"$1\" in\n"
                "        -A) user_agent=\"$2\"; shift 2 ;;\n"
                "        -o) output=\"$2\"; shift 2 ;;\n"
                "        *) shift ;;\n"
                "    esac\n"
                "done\n"
                "[[ \"$user_agent\" == cuVSLAM-dataset-preparation/1.0 ]] || exit 42\n"
                "printf 'synthetic archive' >\"$output\"\n"
            )
            curl.chmod(0o755)
            env = dict(os.environ)
            env["PATH"] = f"{fake_bin}:{env['PATH']}"

            completed = subprocess.run(
                ["bash", str(_download_script()), "--archive", "machine_hall.zip", str(root / "raw")],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("done", completed.stdout)

    def test_subset_preparation_verifies_the_cached_bundle_without_downloading(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            (raw_dir / "machine_hall.zip").write_bytes(b"synthetic archive")
            fake_bin = root / "bin"
            _write_fake_md5sum(fake_bin)
            curl = fake_bin / "curl"
            curl.write_text("#!/usr/bin/env bash\nexit 99\n")
            curl.chmod(0o755)

            with mock.patch.dict(os.environ, {"PATH": f"{fake_bin}:{os.environ['PATH']}"}):
                with redirect_stdout(io.StringIO()):
                    prepared = euroc_prepare.prepare(
                        raw_dir=raw_dir,
                        output_dir=root / "output",
                        sequences=["MH_01_easy"],
                        download_only=True,
                    )

            self.assertEqual(prepared, raw_dir)
            self.assertFalse((raw_dir / "vicon_room1.zip").exists())
            self.assertFalse((raw_dir / "vicon_room2.zip").exists())


if __name__ == "__main__":
    unittest.main()
