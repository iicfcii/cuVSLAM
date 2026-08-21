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

"""Destructive-path behaviour for TartanGround preparation, plus download variants.

``force_download`` deletes both the raw download and the converted tree, and
without it an existing converted tree must be refused rather than overwritten.
"""

import io
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


def _write_tartanground_sequence(sequence_dir: Path, orientations=("front",)) -> None:
    """Create a minimal TartanGround sequence with complete stereo pairs."""
    sequence_dir.mkdir(parents=True, exist_ok=True)
    for orientation in orientations:
        for side in ("lcam", "rcam"):
            images = sequence_dir / f"image_{side}_{orientation}"
            images.mkdir(parents=True, exist_ok=True)
            (images / f"000000_{side}_{orientation}.png").write_text(f"{side}-{orientation}")
            (sequence_dir / f"pose_{side}_{orientation}.txt").write_text("0 0 0 0 0 0 1\n")


class TestTartanForceAndExistingOutput(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.raw_dir = self.root / "raw"
        self.output_dir = self.root / "converted"
        self.sequence_root = self.raw_dir / "dataset" / "tartan_ground"
        self.sequence_dir = self.sequence_root / "OldTownFall" / "Data_anymal" / "P2000"
        self.converted_dir = self.output_dir / "tartan" / "multicamera"

    def tearDown(self):
        self._temporary.cleanup()

    def _converter_writing_cfg_edex(self):
        """Return a converter stub that leaves a cfg.edex in every staged sequence."""

        def convert_sequences(seq_path, save_gt_folder, save_edex_folder):
            sequences = []
            for pose in Path(seq_path).rglob("pose_left.txt"):
                (pose.parent / "cfg.edex").write_text("{}")
                sequences.append(str(pose.parent))
            return sequences

        return mock.Mock(convert_sequences=mock.Mock(side_effect=convert_sequences))

    def _prepare(self, download=None, **kwargs):
        """Run prepare() with a stubbed download step and converter."""
        with mock.patch.object(
            tartan_prepare, "download_tartan_ground", side_effect=download
        ), mock.patch.dict(
            sys.modules,
            {
                "cuvslam_tools.dataset_preparation.tartan.dataset_converter.convert":
                    self._converter_writing_cfg_edex(),
            },
        ), redirect_stdout(io.StringIO()):
            return tartan_prepare.prepare(
                raw_dir=self.raw_dir,
                output_dir=self.output_dir,
                variant="multicamera",
                **kwargs,
            )

    def test_force_download_discards_the_previous_raw_download(self):
        self.sequence_root.mkdir(parents=True)
        (self.sequence_root / "stale.zip").write_text("stale")

        self._prepare(force_download=True, download_only=True)

        self.assertFalse((self.sequence_root / "stale.zip").exists())

    def test_existing_raw_download_is_kept_without_force(self):
        self.sequence_root.mkdir(parents=True)
        (self.sequence_root / "cached.zip").write_text("cached")

        self._prepare(download_only=True)

        self.assertEqual((self.sequence_root / "cached.zip").read_text(), "cached")

    def test_existing_converted_output_is_refused_without_force(self):
        _write_tartanground_sequence(self.sequence_dir)
        self.converted_dir.mkdir(parents=True)
        (self.converted_dir / "previous.txt").write_text("keep me")

        with self.assertRaisesRegex(PreparationError, "already exists"):
            self._prepare()

        self.assertTrue((self.converted_dir / "previous.txt").is_file())

    def test_force_download_replaces_converted_output(self):
        _write_tartanground_sequence(self.sequence_dir)
        self.converted_dir.mkdir(parents=True)
        (self.converted_dir / "previous.txt").write_text("stale")

        # force_download discards the raw download first, so the stub re-fetches it.
        def refetch(variant, data_root):
            _write_tartanground_sequence(self.sequence_dir)

        self._prepare(download=refetch, force_download=True)

        self.assertFalse((self.converted_dir / "previous.txt").exists())
        self.assertIsNotNone(next(self.converted_dir.rglob("cfg.edex"), None))

    def test_raw_data_is_left_untouched_by_conversion(self):
        _write_tartanground_sequence(self.sequence_dir)
        before = sorted(str(path.relative_to(self.raw_dir)) for path in self.raw_dir.rglob("*"))

        self._prepare()

        after = sorted(str(path.relative_to(self.raw_dir)) for path in self.raw_dir.rglob("*"))
        self.assertEqual(before, after)


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

    def test_missing_tartanair_package_is_reported_with_install_instructions(self):
        with mock.patch.dict(sys.modules, {"tartanair": None}):
            with self.assertRaisesRegex(PreparationError, "pip install tartanair"):
                tartan_download.download_tartan_ground("multisensor", "/tmp/tartan_ground")


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
