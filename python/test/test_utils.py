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

import tempfile
import unittest

import cuvslam as vslam


class TestUtils(unittest.TestCase):
    def test_load_multisensor_odometry_config(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as config_file:
            config_file.write(
                "odometry:\n"
                "  odometry_mode: Multisensor\n"
                "  multisensor_settings:\n"
                "    depth_camera_ids: [0, 2]\n"
                "    depth_scale_factor: 1000.0\n"
                "    enable_depth_stereo_tracking: false\n"
            )
            config_file.flush()

            config = vslam.utils.load_odometry_config_from_file(config_file.name)

        self.assertEqual(config.odometry_mode, vslam.Tracker.OdometryMode.Multisensor)
        self.assertEqual(config.multisensor_settings.depth_camera_ids, [0, 2])
        self.assertEqual(config.multisensor_settings.depth_scale_factor, 1000.0)
        self.assertFalse(config.multisensor_settings.enable_depth_stereo_tracking)


if __name__ == "__main__":
    unittest.main()
