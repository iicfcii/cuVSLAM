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
import os
import tarfile
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from cuvslam_tools.dataset_preparation.common import PreparationError
from cuvslam_tools.dataset_preparation.tum import prepare as tum_prepare

SEQUENCE = tum_prepare.SEQUENCE_NAME


def _add_file(tar: tarfile.TarFile, name: str, contents: bytes = b"payload") -> None:
    info = tarfile.TarInfo(name)
    info.size = len(contents)
    tar.addfile(info, io.BytesIO(contents))


def _add_link(tar: tarfile.TarFile, name: str, target: str, symlink: bool = True) -> None:
    info = tarfile.TarInfo(name)
    info.type = tarfile.SYMTYPE if symlink else tarfile.LNKTYPE
    info.linkname = target
    tar.addfile(info)


def _write_sequence_archive(raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / f"{SEQUENCE}.tgz"
    with tarfile.open(archive, "w:gz") as tar:
        _add_file(tar, f"{SEQUENCE}/rgb.txt", b"# rgb\n")
        _add_file(tar, f"{SEQUENCE}/depth.txt", b"# depth\n")
        _add_file(tar, f"{SEQUENCE}/rgb/1341847980.722988.png", b"png")
    return archive


class TestTumSafeExtraction(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)

    def tearDown(self):
        self._temporary.cleanup()

    def _archive(self, build) -> Path:
        archive = self.root / "archive.tgz"
        with tarfile.open(archive, "w:gz") as tar:
            build(tar)
        return archive

    def test_regular_members_are_extracted(self):
        archive = self._archive(lambda tar: _add_file(tar, f"{SEQUENCE}/rgb.txt", b"# rgb\n"))
        destination = self.root / "out"
        destination.mkdir()

        tum_prepare.extract_archive(archive, destination)

        self.assertEqual((destination / SEQUENCE / "rgb.txt").read_text(), "# rgb\n")

    def test_parent_traversal_member_is_rejected(self):
        archive = self._archive(lambda tar: _add_file(tar, "../escaped.txt"))
        destination = self.root / "out"
        destination.mkdir()

        with self.assertRaisesRegex(PreparationError, "unsafe member path"):
            tum_prepare.extract_archive(archive, destination)
        self.assertFalse((self.root / "escaped.txt").exists())

    def test_absolute_member_is_rejected(self):
        archive = self._archive(lambda tar: _add_file(tar, "/etc/escaped.txt"))
        destination = self.root / "out"
        destination.mkdir()

        with self.assertRaisesRegex(PreparationError, "unsafe member path"):
            tum_prepare.extract_archive(archive, destination)

    def test_traversing_symlink_target_is_rejected(self):
        archive = self._archive(lambda tar: _add_link(tar, f"{SEQUENCE}/link", "../../outside"))
        destination = self.root / "out"
        destination.mkdir()

        with self.assertRaisesRegex(PreparationError, "unsafe link target"):
            tum_prepare.extract_archive(archive, destination)

    def test_absolute_hardlink_target_is_rejected(self):
        archive = self._archive(
            lambda tar: _add_link(tar, f"{SEQUENCE}/link", "/etc/passwd", symlink=False)
        )
        destination = self.root / "out"
        destination.mkdir()

        with self.assertRaisesRegex(PreparationError, "unsafe link target"):
            tum_prepare.extract_archive(archive, destination)

    def test_corrupt_archive_is_reported(self):
        archive = self.root / "broken.tgz"
        archive.write_bytes(b"not a gzip archive")
        destination = self.root / "out"
        destination.mkdir()

        with self.assertRaisesRegex(PreparationError, "failed to extract"):
            tum_prepare.extract_archive(archive, destination)


class TestTumPrepareApi(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.raw_dir = self.root / "raw"
        self.output_dir = self.root / "converted"

    def tearDown(self):
        self._temporary.cleanup()

    def test_layout_places_extracted_data_and_rig_calibration(self):
        _write_sequence_archive(self.raw_dir)

        with mock.patch.object(tum_prepare, "run_download_script"), redirect_stdout(io.StringIO()):
            prepared = tum_prepare.prepare(raw_dir=self.raw_dir, output_dir=self.output_dir)

        sequence_dir = self.output_dir / "tum" / SEQUENCE
        self.assertEqual(prepared, sequence_dir)
        self.assertTrue((sequence_dir / "rgb.txt").is_file())
        self.assertTrue((sequence_dir / "rgb" / "1341847980.722988.png").is_file())
        self.assertEqual(
            (sequence_dir / "freiburg3_rig.yaml").read_text(),
            (Path(tum_prepare.__file__).with_name("freiburg3_rig.yaml")).read_text(),
        )

    def test_download_only_skips_extraction_and_returns_raw_dir(self):
        with mock.patch.object(tum_prepare, "run_download_script") as download, redirect_stdout(
            io.StringIO()
        ):
            prepared = tum_prepare.prepare(
                raw_dir=self.raw_dir,
                output_dir=self.output_dir,
                download_only=True,
            )

        self.assertEqual(prepared, self.raw_dir)
        self.assertFalse(self.output_dir.exists())
        self.assertEqual(download.call_args.args[1], [str(self.raw_dir)])

    def test_force_download_reaches_the_download_script(self):
        _write_sequence_archive(self.raw_dir)

        with mock.patch.object(tum_prepare, "run_download_script") as download, redirect_stdout(
            io.StringIO()
        ):
            tum_prepare.prepare(
                raw_dir=self.raw_dir,
                output_dir=self.output_dir,
                force_download=True,
            )

        self.assertEqual(download.call_args.args[1], [str(self.raw_dir), "--force"])

    def test_missing_archive_is_reported(self):
        self.raw_dir.mkdir(parents=True)

        with mock.patch.object(tum_prepare, "run_download_script"), redirect_stdout(io.StringIO()):
            with self.assertRaises(OSError):
                tum_prepare.prepare(raw_dir=self.raw_dir, output_dir=self.output_dir)

    def test_archive_without_the_expected_sequence_is_reported(self):
        self.raw_dir.mkdir(parents=True)
        with tarfile.open(self.raw_dir / f"{SEQUENCE}.tgz", "w:gz") as tar:
            _add_file(tar, "some_other_sequence/rgb.txt")

        with mock.patch.object(tum_prepare, "run_download_script"), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(PreparationError, "after extraction"):
                tum_prepare.prepare(raw_dir=self.raw_dir, output_dir=self.output_dir)

    def test_defaults_are_relative_to_the_current_directory(self):
        previous = Path.cwd()
        os.chdir(self.root)
        try:
            with mock.patch.object(tum_prepare, "run_download_script") as download, redirect_stdout(
                io.StringIO()
            ):
                prepared = tum_prepare.prepare(download_only=True)
        finally:
            os.chdir(previous)

        self.assertEqual(prepared, self.root / "datasets" / "tum" / "raw")
        self.assertEqual(
            download.call_args.args[1],
            [str(self.root / "datasets" / "tum" / "raw")],
        )


class TestTumPrepareCli(unittest.TestCase):
    def test_main_forwards_every_flag_to_prepare(self):
        with mock.patch.object(tum_prepare, "prepare") as prepare:
            self.assertEqual(
                tum_prepare.main(
                    ["--raw-dir", "/raw", "--output-dir", "/converted", "--download-only"]
                ),
                0,
            )

        prepare.assert_called_once_with(
            raw_dir=Path("/raw"),
            output_dir=Path("/converted"),
            force_download=False,
            download_only=True,
        )

    def test_main_reports_preparation_errors_on_stderr(self):
        with mock.patch.object(tum_prepare, "prepare", side_effect=PreparationError("archive missing")):
            with mock.patch("sys.stderr", new_callable=io.StringIO) as stderr:
                self.assertEqual(tum_prepare.main([]), 1)

        self.assertIn("archive missing", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
