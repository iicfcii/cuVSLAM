/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * NVIDIA software released under the NVIDIA Community License is intended to be used to enable
 * the further development of AI and robotics technologies. Such software has been designed, tested,
 * and optimized for use with NVIDIA hardware, and this License grants permission to use the software
 * solely with such hardware.
 * Subject to the terms of this License, NVIDIA confirms that you are free to commercially use,
 * modify, and distribute the software with NVIDIA hardware. NVIDIA does not claim ownership of any
 * outputs generated using the software or derivative works thereof. Any code contributions that you
 * share with NVIDIA are licensed to NVIDIA as feedback under this License and may be incorporated
 * in future releases without notice or attribution.
 * By using, reproducing, modifying, distributing, performing, or displaying any portion or element
 * of the software or derivative works thereof, you agree to be bound by this License.
 */

#include "utils/cuvslam_yaml_config.h"

#include <fstream>
#include <string>
#include <vector>

#include "common/include_gtest.h"
#include "cuvslam/cuvslam2_internal.h"

namespace test::utils {

TEST(CuvslamYamlConfig, LoadsPerFrameKeyframeOverrideInternals) {
  const std::string path = ::testing::TempDir() + "/internals_keyframe_override.yaml";
  {
    std::ofstream file(path);
    file << "internals:\n"
         << "  kf_override_frame_selection: true\n";
  }

  cuvslam::internal::Internals internals;
  ASSERT_TRUE(cuvslam::LoadInternalsFromFile(path.c_str(), internals));

  ASSERT_TRUE(internals.kf_override_frame_selection.has_value());
  EXPECT_TRUE(*internals.kf_override_frame_selection);
}

TEST(CuvslamYamlConfig, LoadsMultisensorSettings) {
  const std::string path = ::testing::TempDir() + "/multisensor_odometry.yaml";
  {
    std::ofstream file(path);
    file << "odometry:\n"
         << "  odometry_mode: Multisensor\n"
         << "  multisensor_settings:\n"
         << "    depth_camera_ids: [0, 2]\n"
         << "    depth_scale_factor: 1000.0\n"
         << "    enable_depth_stereo_tracking: false\n";
  }

  cuvslam::Odometry::Config config;
  ASSERT_TRUE(cuvslam::LoadOdometryConfigFromFile(path.c_str(), config));

  EXPECT_EQ(config.odometry_mode, cuvslam::Odometry::OdometryMode::Multisensor);
  EXPECT_EQ(config.multisensor_settings.depth_camera_ids, (std::vector<int32_t>{0, 2}));
  EXPECT_FLOAT_EQ(config.multisensor_settings.depth_scale_factor, 1000.f);
  EXPECT_FALSE(config.multisensor_settings.enable_depth_stereo_tracking);
}

}  // namespace test::utils
