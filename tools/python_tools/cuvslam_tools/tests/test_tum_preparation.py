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

"""Archive extraction safety for the TUM RGB-D preparation.

Extraction is hand-rolled on Python tarfile instead of tar, so member and link
paths that would escape the destination need to stay rejected.
"""

import io
import tarfile
import tempfile
import unittest
from pathlib import Path

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


class TestTumSafeExtraction(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.destination = self.root / "out"
        self.destination.mkdir()

    def tearDown(self):
        self._temporary.cleanup()

    def _archive(self, build) -> Path:
        archive = self.root / "archive.tgz"
        with tarfile.open(archive, "w:gz") as tar:
            build(tar)
        return archive

    def test_regular_members_are_extracted(self):
        archive = self._archive(lambda tar: _add_file(tar, f"{SEQUENCE}/rgb.txt", b"# rgb\n"))

        tum_prepare.extract_archive(archive, self.destination)

        self.assertEqual((self.destination / SEQUENCE / "rgb.txt").read_text(), "# rgb\n")

    def test_parent_traversal_member_is_rejected(self):
        archive = self._archive(lambda tar: _add_file(tar, "../escaped.txt"))

        with self.assertRaisesRegex(PreparationError, "unsafe member path"):
            tum_prepare.extract_archive(archive, self.destination)
        self.assertFalse((self.root / "escaped.txt").exists())

    def test_absolute_member_is_rejected(self):
        archive = self._archive(lambda tar: _add_file(tar, "/etc/escaped.txt"))

        with self.assertRaisesRegex(PreparationError, "unsafe member path"):
            tum_prepare.extract_archive(archive, self.destination)

    def test_traversing_symlink_target_is_rejected(self):
        archive = self._archive(lambda tar: _add_link(tar, f"{SEQUENCE}/link", "../../outside"))

        with self.assertRaisesRegex(PreparationError, "unsafe link target"):
            tum_prepare.extract_archive(archive, self.destination)

    def test_absolute_hardlink_target_is_rejected(self):
        archive = self._archive(
            lambda tar: _add_link(tar, f"{SEQUENCE}/link", "/etc/passwd", symlink=False)
        )

        with self.assertRaisesRegex(PreparationError, "unsafe link target"):
            tum_prepare.extract_archive(archive, self.destination)

    def test_corrupt_archive_is_reported(self):
        archive = self.root / "broken.tgz"
        archive.write_bytes(b"not a gzip archive")

        with self.assertRaisesRegex(PreparationError, "failed to extract"):
            tum_prepare.extract_archive(archive, self.destination)


if __name__ == "__main__":
    unittest.main()
