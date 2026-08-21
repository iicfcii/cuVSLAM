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
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from cuvslam_tools.dataset_preparation.common import PreparationError
from cuvslam_tools.dataset_preparation.tartan import download as tartan_download
from cuvslam_tools.dataset_preparation.tartan import prepare as tartan_prepare
from cuvslam_tools.dataset_preparation.tartan import stage_sequences as tartan_stage

_ORIENTATIONS = ("front", "left", "right", "back", "top", "bottom")


def _write_tartanground_sequence(sequence_dir: Path, orientations=("front",)) -> None:
    """Create a minimal TartanGround sequence with complete stereo pairs."""
    sequence_dir.mkdir(parents=True, exist_ok=True)
    for orientation in orientations:
        for side in ("lcam", "rcam"):
            images = sequence_dir / f"image_{side}_{orientation}"
            images.mkdir(parents=True, exist_ok=True)
            (images / f"000000_{side}_{orientation}.png").write_text(f"{side}-{orientation}")
            (sequence_dir / f"pose_{side}_{orientation}.txt").write_text("0 0 0 0 0 0 1\n")


class TestTartanPrepareDownload(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.raw_dir = self.root / "raw"
        self.output_dir = self.root / "converted"

    def tearDown(self):
        self._temporary.cleanup()

    def test_download_only_multisensor_never_stages_or_converts(self):
        with mock.patch.object(tartan_prepare, "download_tartan_ground") as download, mock.patch.object(
            tartan_prepare, "stage_sequences"
        ) as stage, redirect_stdout(io.StringIO()):
            prepared = tartan_prepare.prepare(
                raw_dir=self.raw_dir,
                output_dir=self.output_dir,
                variant="multisensor",
                download_only=True,
            )

        self.assertEqual(prepared, self.raw_dir)
        stage.assert_not_called()
        download.assert_called_once_with(
            "multisensor",
            str(self.raw_dir / "dataset" / "tartan_ground"),
        )

    def test_downloader_receives_an_absolute_data_root(self):
        with mock.patch.object(tartan_prepare, "download_tartan_ground") as download, redirect_stdout(
            io.StringIO()
        ):
            tartan_prepare.prepare(
                raw_dir=self.raw_dir,
                output_dir=self.output_dir,
                download_only=True,
            )

        data_root = Path(download.call_args.args[1])
        self.assertTrue(data_root.is_absolute())
        self.assertEqual(data_root, self.raw_dir / "dataset" / "tartan_ground")

    def test_force_download_discards_the_previous_download(self):
        sequence_root = self.raw_dir / "dataset" / "tartan_ground"
        sequence_root.mkdir(parents=True)
        (sequence_root / "stale.zip").write_text("stale")

        with mock.patch.object(tartan_prepare, "download_tartan_ground") as download, redirect_stdout(
            io.StringIO()
        ):
            tartan_prepare.prepare(
                raw_dir=self.raw_dir,
                output_dir=self.output_dir,
                variant="multicamera",
                force_download=True,
                download_only=True,
            )

        self.assertFalse((sequence_root / "stale.zip").exists())
        download.assert_called_once_with("multicamera", str(sequence_root))

    def test_existing_download_is_kept_without_force(self):
        sequence_root = self.raw_dir / "dataset" / "tartan_ground"
        sequence_root.mkdir(parents=True)
        (sequence_root / "cached.zip").write_text("cached")

        with mock.patch.object(tartan_prepare, "download_tartan_ground"), redirect_stdout(io.StringIO()):
            tartan_prepare.prepare(
                raw_dir=self.raw_dir,
                output_dir=self.output_dir,
                download_only=True,
            )

        self.assertEqual((sequence_root / "cached.zip").read_text(), "cached")

    def test_unknown_variant_is_rejected_before_downloading(self):
        with mock.patch.object(tartan_prepare, "download_tartan_ground") as download:
            with self.assertRaisesRegex(PreparationError, "unknown variant"):
                tartan_prepare.prepare(raw_dir=self.raw_dir, variant="stereo")

        download.assert_not_called()

    def test_defaults_are_relative_to_the_current_directory(self):
        previous = Path.cwd()
        os.chdir(self.root)
        try:
            with mock.patch.object(tartan_prepare, "download_tartan_ground") as download, redirect_stdout(
                io.StringIO()
            ):
                prepared = tartan_prepare.prepare(download_only=True)
        finally:
            os.chdir(previous)

        expected_raw = self.root / "datasets" / "tartan" / "raw"
        self.assertEqual(prepared, expected_raw)
        self.assertEqual(
            download.call_args.args[1],
            str(expected_raw / "dataset" / "tartan_ground"),
        )

    def test_missing_tartanair_package_is_reported_with_install_instructions(self):
        with mock.patch.dict(sys.modules, {"tartanair": None}):
            with self.assertRaisesRegex(PreparationError, "pip install tartanair"):
                tartan_download.download_tartan_ground("multisensor", str(self.raw_dir))


class TestTartanPrepareConversion(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.raw_dir = self.root / "raw"
        self.output_dir = self.root / "converted"
        self.sequence_dir = self.raw_dir / "dataset" / "tartan_ground" / "OldTownFall" / "Data_anymal" / "P2000"

    def tearDown(self):
        self._temporary.cleanup()

    def _prepare(self, converter, download=None, **kwargs):
        """Run prepare() with a stubbed download step and converter."""
        with mock.patch.object(
            tartan_prepare, "download_tartan_ground", side_effect=download
        ), mock.patch.dict(
            sys.modules,
            {"cuvslam_tools.dataset_preparation.tartan.dataset_converter.convert": converter},
        ), redirect_stdout(io.StringIO()):
            return tartan_prepare.prepare(
                raw_dir=self.raw_dir,
                output_dir=self.output_dir,
                **kwargs,
            )

    def _converter_writing_cfg_edex(self):
        """Return a converter stub that leaves a cfg.edex in every staged sequence."""

        def convert_sequences(seq_path, save_gt_folder, save_edex_folder):
            sequences = []
            for pose in Path(seq_path).rglob("pose_left.txt"):
                (pose.parent / "cfg.edex").write_text("{}")
                sequences.append(str(pose.parent))
            return sequences

        return mock.Mock(convert_sequences=mock.Mock(side_effect=convert_sequences))

    def test_multicamera_stages_every_complete_stereo_pair(self):
        _write_tartanground_sequence(self.sequence_dir, orientations=_ORIENTATIONS)
        converter = self._converter_writing_cfg_edex()

        converted_dir = self._prepare(converter, variant="multicamera")

        self.assertEqual(converted_dir, self.output_dir / "tartan" / "multicamera")
        staged = sorted(path.name for path in converted_dir.rglob("P2000_*"))
        self.assertEqual(staged, sorted(f"P2000_{orientation}" for orientation in _ORIENTATIONS))
        for orientation in _ORIENTATIONS:
            sequence = converted_dir / "OldTownFall" / "Data_anymal" / f"P2000_{orientation}"
            self.assertTrue((sequence / "image_left" / "000000_left.png").is_file())
            self.assertTrue((sequence / "image_right" / "000000_right.png").is_file())
            self.assertTrue((sequence / "cfg.edex").is_file())

    def test_incomplete_stereo_pairs_are_skipped(self):
        _write_tartanground_sequence(self.sequence_dir, orientations=("front", "left"))
        (self.sequence_dir / "pose_rcam_left.txt").unlink()
        converter = self._converter_writing_cfg_edex()

        converted_dir = self._prepare(converter, variant="multicamera")

        self.assertEqual([path.name for path in converted_dir.rglob("P2000_*")], ["P2000_front"])

    def test_raw_data_is_left_untouched(self):
        _write_tartanground_sequence(self.sequence_dir)
        before = sorted(str(path.relative_to(self.raw_dir)) for path in self.raw_dir.rglob("*"))

        self._prepare(self._converter_writing_cfg_edex(), variant="multicamera")

        after = sorted(str(path.relative_to(self.raw_dir)) for path in self.raw_dir.rglob("*"))
        self.assertEqual(before, after)

    def test_existing_output_is_kept_unless_forced(self):
        _write_tartanground_sequence(self.sequence_dir)
        converted_dir = self.output_dir / "tartan" / "multicamera"
        converted_dir.mkdir(parents=True)
        (converted_dir / "previous.txt").write_text("keep me")

        with self.assertRaisesRegex(PreparationError, "already exists"):
            self._prepare(self._converter_writing_cfg_edex(), variant="multicamera")

        self.assertTrue((converted_dir / "previous.txt").is_file())

    def test_force_download_replaces_previous_output(self):
        _write_tartanground_sequence(self.sequence_dir)
        converted_dir = self.output_dir / "tartan" / "multicamera"
        converted_dir.mkdir(parents=True)
        (converted_dir / "previous.txt").write_text("stale")

        # force_download discards the raw download first, so the stub re-fetches it.
        def refetch(variant, data_root):
            _write_tartanground_sequence(self.sequence_dir)

        self._prepare(
            self._converter_writing_cfg_edex(),
            download=refetch,
            variant="multicamera",
            force_download=True,
        )

        self.assertFalse((converted_dir / "previous.txt").exists())
        self.assertIsNotNone(next(converted_dir.rglob("cfg.edex"), None))

    def test_missing_download_is_reported(self):
        with self.assertRaisesRegex(PreparationError, "no downloaded TartanGround data"):
            self._prepare(self._converter_writing_cfg_edex(), variant="multicamera")

    def test_unconvertible_layout_is_reported(self):
        (self.sequence_dir / "image_lcam_front").mkdir(parents=True)

        with self.assertRaisesRegex(PreparationError, "no convertible"):
            self._prepare(self._converter_writing_cfg_edex(), variant="multicamera")

    def test_conversion_without_edex_output_fails(self):
        _write_tartanground_sequence(self.sequence_dir)
        converter = mock.Mock(convert_sequences=mock.Mock(return_value=[]))

        with self.assertRaisesRegex(PreparationError, "produced no cfg.edex"):
            self._prepare(converter, variant="multicamera")


class TestTartanPrepareEndToEnd(unittest.TestCase):
    """Exercise the real converter pipeline, which needs numpy and scipy."""

    @classmethod
    def setUpClass(cls):
        try:
            import numpy  # noqa: F401
            import scipy  # noqa: F401
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise unittest.SkipTest(f"converter dependency unavailable: {exc}")

    def test_staged_sequences_convert_to_the_edex_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_dir = root / "raw"
            sequence_dir = raw_dir / "dataset" / "tartan_ground" / "OldTownFall" / "Data_anymal" / "P2000"
            _write_tartanground_sequence(sequence_dir, orientations=("front", "left"))

            with mock.patch.object(tartan_prepare, "download_tartan_ground"), redirect_stdout(io.StringIO()):
                converted_dir = tartan_prepare.prepare(
                    raw_dir=raw_dir,
                    output_dir=root / "converted",
                    variant="multicamera",
                )

            for orientation in ("front", "left"):
                sequence = converted_dir / "OldTownFall" / "Data_anymal" / f"P2000_{orientation}"
                self.assertTrue((sequence / "00" / "00.0.000000.png").is_file())
                self.assertTrue((sequence / "01" / "00.0.000000.png").is_file())
                self.assertTrue((sequence / "gt.txt").is_file())
                self.assertTrue((sequence / "cfg.edex").is_file())
                self.assertFalse((sequence / "image_left").exists())


class TestTartanPrepareCli(unittest.TestCase):
    def test_main_forwards_every_flag_to_prepare(self):
        with mock.patch.object(tartan_prepare, "prepare") as prepare:
            self.assertEqual(
                tartan_prepare.main(
                    [
                        "--variant",
                        "multicamera",
                        "--raw-dir",
                        "/raw",
                        "--output-dir",
                        "/converted",
                        "--force-download",
                    ]
                ),
                0,
            )

        prepare.assert_called_once_with(
            raw_dir=Path("/raw"),
            output_dir=Path("/converted"),
            variant="multicamera",
            force_download=True,
            download_only=False,
        )

    def test_main_defaults_to_the_multisensor_variant(self):
        with mock.patch.object(tartan_prepare, "prepare") as prepare:
            self.assertEqual(tartan_prepare.main([]), 0)

        self.assertEqual(prepare.call_args.kwargs["variant"], "multisensor")

    def test_main_rejects_an_unknown_variant_as_a_usage_error(self):
        with mock.patch("sys.stderr", new_callable=io.StringIO):
            with self.assertRaises(SystemExit) as raised:
                tartan_prepare.main(["--variant", "stereo"])

        self.assertEqual(raised.exception.code, 2)

    def test_main_reports_preparation_errors_on_stderr(self):
        with mock.patch.object(
            tartan_prepare, "prepare", side_effect=PreparationError("conversion produced no cfg.edex")
        ):
            with mock.patch("sys.stderr", new_callable=io.StringIO) as stderr:
                self.assertEqual(tartan_prepare.main([]), 1)

        self.assertIn("conversion produced no cfg.edex", stderr.getvalue())


class TestTartanDownloadVariants(unittest.TestCase):
    def test_download_variants_share_download_implementation(self):
        expected_variants = {
            "multicamera": {
                "modality": ["image", "meta"],
                "camera_name": [
                    "lcam_front",
                    "rcam_front",
                    "lcam_left",
                    "rcam_left",
                    "lcam_right",
                    "rcam_right",
                    "lcam_back",
                    "rcam_back",
                    "lcam_top",
                    "rcam_top",
                    "lcam_bottom",
                    "rcam_bottom",
                ],
            },
            "multisensor": {
                "modality": ["image", "depth", "imu", "meta"],
                "camera_name": ["lcam_front", "lcam_back"],
            },
        }

        for variant, config in expected_variants.items():
            with self.subTest(variant=variant):
                fake_tartanair = mock.Mock()

                with mock.patch.dict("sys.modules", {"tartanair": fake_tartanair}), mock.patch.object(
                    tartan_download.os,
                    "makedirs",
                ) as makedirs:
                    tartan_download.download_tartan_ground(variant, "/tmp/tartan_ground")

                makedirs.assert_called_once_with("/tmp/tartan_ground", exist_ok=True)
                fake_tartanair.init.assert_called_once_with("/tmp/tartan_ground")
                fake_tartanair.download_ground.assert_called_once_with(
                    env=["OldTownFall"],
                    version=["anymal"],
                    modality=config["modality"],
                    traj=["P2000"],
                    camera_name=config["camera_name"],
                    unzip=True,
                )

    def test_download_main_passes_variant_and_data_root(self):
        with mock.patch.object(tartan_download, "download_tartan_ground") as download_tartan_ground:
            self.assertEqual(
                tartan_download.main(["--variant", "multicamera", "--data-root", "/tmp/tartan_ground"]),
                0,
            )

        download_tartan_ground.assert_called_once_with("multicamera", "/tmp/tartan_ground")

    def test_download_main_reports_a_missing_package_on_stderr(self):
        with mock.patch.object(
            tartan_download,
            "download_tartan_ground",
            side_effect=PreparationError("the tartanair package is required"),
        ):
            with mock.patch("sys.stderr", new_callable=io.StringIO) as stderr:
                self.assertEqual(tartan_download.main([]), 1)

        self.assertIn("tartanair package is required", stderr.getvalue())


class TestTartanStaging(unittest.TestCase):
    def test_staging_maps_stereo_pair_to_classic_layout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            raw_sequence = temp_path / "raw" / "OldTownFall" / "Data_anymal" / "P2000"
            _write_tartanground_sequence(raw_sequence)

            staged = tartan_stage.stage_sequences(temp_path / "raw", temp_path / "converted")

            self.assertEqual([temp_path / "converted" / "OldTownFall" / "Data_anymal" / "P2000_front"], staged)
            staged_sequence = staged[0]
            self.assertEqual((staged_sequence / "image_left" / "000000_left.png").read_text(), "lcam-front")
            self.assertEqual((staged_sequence / "image_right" / "000000_right.png").read_text(), "rcam-front")
            self.assertTrue((staged_sequence / "pose_left.txt").is_file())
            self.assertTrue((staged_sequence / "pose_right.txt").is_file())


if __name__ == "__main__":
    unittest.main()
