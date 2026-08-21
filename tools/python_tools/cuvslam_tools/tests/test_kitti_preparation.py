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

import io
import json
import os
import struct
import tempfile
import unittest
import zipfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from cuvslam_tools.dataset_preparation.common import PreparationError
from cuvslam_tools.dataset_preparation.kitti import convert_kitti
from cuvslam_tools.dataset_preparation.kitti import prepare as kitti_prepare

_CALIB_TEXT = (
    "P0: 718.856 0.0 607.1928 0.0 0.0 718.856 185.2157 0.0 0.0 0.0 1.0 0.0\n"
    "P1: 718.856 0.0 607.1928 -386.1448 0.0 718.856 185.2157 0.0 0.0 0.0 1.0 0.0\n"
)


def _png_bytes(width: int, height: int) -> bytes:
    """Return the leading PNG bytes the converter reads to learn image size."""
    return (
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", 13)
        + b"IHDR"
        + struct.pack(">I", width)
        + struct.pack(">I", height)
    )


def _write_zip(path: Path, members: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)


def _write_raw_archives(raw_dir: Path, sequences=("00", "11"), frames: int = 2) -> None:
    """Write minimal but structurally valid KITTI odometry archives."""
    gray_members = {}
    calib_members = {}
    poses_members = {}
    for sequence in sequences:
        for frame in range(frames):
            image = _png_bytes(1241, 376)
            gray_members[f"dataset/sequences/{sequence}/image_0/{frame:06d}.png"] = image
            gray_members[f"dataset/sequences/{sequence}/image_1/{frame:06d}.png"] = image
        calib_members[f"dataset/sequences/{sequence}/calib.txt"] = _CALIB_TEXT
        if sequence in convert_kitti.GT_SEQS:
            poses_members[f"dataset/poses/{sequence}.txt"] = "1 0 0 0 0 1 0 0 0 0 1 0\n"

    _write_zip(raw_dir / "data_odometry_gray.zip", gray_members)
    _write_zip(raw_dir / "data_odometry_calib.zip", calib_members)
    _write_zip(raw_dir / "data_odometry_poses.zip", poses_members)


class TestKittiPrepareApi(unittest.TestCase):
    """prepare() is callable directly, without argparse or subprocess mocking."""

    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.raw_dir = self.root / "raw"
        self.output_dir = self.root / "converted"

    def tearDown(self):
        self._temporary.cleanup()

    def test_download_only_skips_conversion_and_returns_raw_dir(self):
        self.raw_dir.mkdir(parents=True)
        with mock.patch.object(kitti_prepare, "run_download_script") as download, mock.patch.object(
            kitti_prepare.convert_kitti, "convert"
        ) as convert, redirect_stdout(io.StringIO()):
            prepared = kitti_prepare.prepare(
                raw_dir=self.raw_dir,
                output_dir=self.output_dir,
                download_only=True,
            )

        self.assertEqual(prepared, self.raw_dir)
        convert.assert_not_called()
        download.assert_called_once()
        script, arguments = download.call_args.args
        self.assertEqual(script.name, "download_kitti.sh")
        self.assertEqual(arguments, [str(self.raw_dir)])

    def test_force_download_reaches_the_download_script(self):
        with mock.patch.object(kitti_prepare, "run_download_script") as download, mock.patch.object(
            kitti_prepare.convert_kitti, "convert"
        ), redirect_stdout(io.StringIO()):
            kitti_prepare.prepare(
                raw_dir=self.raw_dir,
                output_dir=self.output_dir,
                force_download=True,
            )

        self.assertEqual(download.call_args.args[1], [str(self.raw_dir), "--force"])

    def test_conversion_receives_resolved_directories(self):
        with mock.patch.object(kitti_prepare, "run_download_script"), mock.patch.object(
            kitti_prepare.convert_kitti, "convert"
        ) as convert, redirect_stdout(io.StringIO()):
            prepared = kitti_prepare.prepare(raw_dir=self.raw_dir, output_dir=self.output_dir)

        self.assertEqual(prepared, self.output_dir)
        convert.assert_called_once_with(self.raw_dir, self.output_dir)

    def test_defaults_are_relative_to_the_current_directory(self):
        previous = Path.cwd()
        os.chdir(self.root)
        try:
            with mock.patch.object(kitti_prepare, "run_download_script"), mock.patch.object(
                kitti_prepare.convert_kitti, "convert"
            ) as convert, redirect_stdout(io.StringIO()):
                kitti_prepare.prepare()
        finally:
            os.chdir(previous)

        convert.assert_called_once_with(
            self.root / "datasets" / "kitti" / "raw",
            self.root / "datasets" / "converted",
        )

    def test_missing_archives_raise_a_preparation_error(self):
        self.raw_dir.mkdir(parents=True)
        with mock.patch.object(kitti_prepare, "run_download_script"), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(PreparationError, "data_odometry_gray.zip not found"):
                kitti_prepare.prepare(raw_dir=self.raw_dir, output_dir=self.output_dir)


class TestKittiPrepareCli(unittest.TestCase):
    def test_main_forwards_every_flag_to_prepare(self):
        with mock.patch.object(kitti_prepare, "prepare") as prepare:
            self.assertEqual(
                kitti_prepare.main(
                    ["--raw-dir", "/raw", "--output-dir", "/out", "--force-download", "--download-only"]
                ),
                0,
            )

        prepare.assert_called_once_with(
            raw_dir=Path("/raw"),
            output_dir=Path("/out"),
            force_download=True,
            download_only=True,
        )

    def test_main_defaults_leave_path_resolution_to_prepare(self):
        with mock.patch.object(kitti_prepare, "prepare") as prepare:
            self.assertEqual(kitti_prepare.main([]), 0)

        prepare.assert_called_once_with(
            raw_dir=None,
            output_dir=None,
            force_download=False,
            download_only=False,
        )

    def test_main_reports_preparation_errors_on_stderr(self):
        with mock.patch.object(kitti_prepare, "prepare", side_effect=PreparationError("archive missing")):
            with mock.patch("sys.stderr", new_callable=io.StringIO) as stderr:
                self.assertEqual(kitti_prepare.main([]), 1)

        self.assertIn("archive missing", stderr.getvalue())


class TestKittiPrepareEndToEnd(unittest.TestCase):
    def test_cached_archives_convert_to_the_expected_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_dir = root / "raw"
            output_dir = root / "converted"
            _write_raw_archives(raw_dir)

            # The real download script reuses the cached archives and never fetches.
            with redirect_stdout(io.StringIO()):
                prepared = kitti_prepare.prepare(raw_dir=raw_dir, output_dir=output_dir)

            self.assertEqual(prepared, output_dir)
            self.assertTrue((output_dir / "00" / "00" / "00.0.0001.png").is_file())
            self.assertTrue((output_dir / "00" / "01" / "00.1.0002.png").is_file())
            self.assertTrue((output_dir / "00" / "gt.txt").is_file())
            self.assertFalse((output_dir / "11" / "gt.txt").exists())

            edex = json.loads((output_dir / "00" / "stereo.edex").read_text())
            self.assertEqual(edex[0]["frame_end"], 2)
            self.assertEqual(edex[0]["cameras"][0]["intrinsics"]["size"], [1241, 376])

            for name in (
                "kitti-slam_gt.cfg",
                "kitti-vio_gt.cfg",
                "kitti-vio_slam.cfg",
                "kitti-vio_slam_gt.cfg",
            ):
                self.assertTrue((output_dir / name).is_file(), name)

            all_sequences = json.loads((output_dir / "kitti-vio_slam.cfg").read_text())
            self.assertEqual(
                {entry["sequence_folder"] for entry in all_sequences["sequence_cfgs"]},
                {"00", "11"},
            )
            gt_only = json.loads((output_dir / "kitti-vio_gt.cfg").read_text())
            self.assertEqual(
                {entry["sequence_folder"] for entry in gt_only["sequence_cfgs"]},
                {"00"},
            )


if __name__ == "__main__":
    unittest.main()
