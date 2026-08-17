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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "common/include_gtest.h"
#include "cuvslam/cuvslam2.h"

namespace {

using cuvslam::Odometry;
using cuvslam::Pose;
using cuvslam::Rig;
using cuvslam::Slam;
using cuvslam::Tracker;

constexpr int32_t kWidth = 640;
constexpr int32_t kHeight = 480;
constexpr float kFocal = 500.f;
constexpr float kBaseline = 0.1f;
/// Disparity of the synthetic scene: a fronto-parallel plane at kFocal * kBaseline / kDisparity metres
constexpr int32_t kDisparity = 25;
/// Sideways shift of the rig between consecutive frames, in pixels. Large enough that the sequence
/// spans several keyframes, so SLAM produces a trajectory rather than sitting at the origin.
constexpr int32_t kShiftPerFrame = 8;
constexpr int64_t kFramePeriodNs = 33'333'333;
constexpr size_t kNumFrames = 40;

/// Distance to the synthetic plane, and the sideways travel the tracker is expected to recover
constexpr float kDepthM = kFocal * kBaseline / kDisparity;
constexpr float kTravelPerFrameM = kShiftPerFrame * kDepthM / kFocal;
constexpr float kExpectedTravelM = (kNumFrames - 1) * kTravelPerFrameM;

Rig MakeStereoRig() {
  cuvslam::Camera camera;
  camera.size = {kWidth, kHeight};
  camera.focal = {kFocal, kFocal};
  camera.principal = {kWidth / 2.f, kHeight / 2.f};

  Rig rig;
  rig.cameras.push_back(camera);
  rig.cameras.push_back(camera);
  rig.cameras[1].rig_from_camera.translation = {kBaseline, 0.f, 0.f};
  return rig;
}

/**
 * Deterministic synthetic stereo sequence.
 *
 * A wide texture is generated once, and each frame crops a window out of it: the left camera at an
 * offset that grows with the frame index, the right camera at a further constant disparity. That
 * imitates a rig translating sideways in front of a fronto-parallel plane, which gives the tracker
 * real features to work with while staying reproducible run to run.
 */
class SyntheticStereoSequence {
public:
  explicit SyntheticStereoSequence(size_t num_frames)
      : texture_width_{kWidth + kDisparity + static_cast<int32_t>(num_frames) * kShiftPerFrame} {
    texture_.resize(static_cast<size_t>(texture_width_) * kHeight);
    uint32_t seed = 12345u;
    for (int32_t y = 0; y < kHeight; ++y) {
      for (int32_t x = 0; x < texture_width_; ++x) {
        // Incommensurable frequencies keep the pattern from repeating, so feature matches stay
        // unambiguous; the noise term adds the high-frequency detail corner detectors need.
        const float wave = std::sin(x * 0.07f) + std::sin(y * 0.11f) + std::sin((x + y) * 0.031f);
        seed = seed * 1664525u + 1013904223u;
        const float noise = static_cast<float>(seed >> 24) / 255.f;
        const float value = 128.f + 30.f * wave + 40.f * noise;
        texture_[static_cast<size_t>(y) * texture_width_ + x] = static_cast<uint8_t>(std::lround(value));
      }
    }
    left_.resize(static_cast<size_t>(kWidth) * kHeight);
    right_.resize(static_cast<size_t>(kWidth) * kHeight);
  }

  /// Fill the internal buffers with frame `index` and return images referencing them. The returned
  /// images are valid until the next GetFrame() call.
  Odometry::ImageSet GetFrame(size_t index) {
    const int32_t left_offset = static_cast<int32_t>(index) * kShiftPerFrame;
    CropInto(left_, left_offset);
    CropInto(right_, left_offset + kDisparity);

    const int64_t timestamp_ns = static_cast<int64_t>(index + 1) * kFramePeriodNs;
    Odometry::ImageSet images;
    images.push_back(MakeImage(left_, timestamp_ns, 0));
    images.push_back(MakeImage(right_, timestamp_ns, 1));
    return images;
  }

private:
  void CropInto(std::vector<uint8_t>& dst, int32_t x_offset) const {
    for (int32_t y = 0; y < kHeight; ++y) {
      const uint8_t* src_row = texture_.data() + static_cast<size_t>(y) * texture_width_ + x_offset;
      std::copy_n(src_row, kWidth, dst.data() + static_cast<size_t>(y) * kWidth);
    }
  }

  static cuvslam::Image MakeImage(const std::vector<uint8_t>& pixels, int64_t timestamp_ns, uint32_t camera_index) {
    return cuvslam::Image{{pixels.data(), kWidth, kHeight, kWidth, cuvslam::ImageData::Encoding::MONO,
                           cuvslam::ImageData::DataType::UINT8, false /* is_gpu_mem */},
                          timestamp_ns,
                          camera_index};
  }

  int32_t texture_width_;
  std::vector<uint8_t> texture_;
  std::vector<uint8_t> left_;
  std::vector<uint8_t> right_;
};

float TranslationNorm(const Pose& pose) {
  return std::sqrt(pose.translation[0] * pose.translation[0] + pose.translation[1] * pose.translation[1] +
                   pose.translation[2] * pose.translation[2]);
}

void ExpectPoseNear(const Pose& actual, const Pose& expected, float tolerance) {
  for (size_t i = 0; i < expected.rotation.size(); ++i) {
    EXPECT_NEAR(actual.rotation[i], expected.rotation[i], tolerance) << "rotation[" << i << "]";
  }
  for (size_t i = 0; i < expected.translation.size(); ++i) {
    EXPECT_NEAR(actual.translation[i], expected.translation[i], tolerance) << "translation[" << i << "]";
  }
}

/// Deterministic configuration: the asynchronous bundler and the SLAM worker thread both make
/// results depend on timing, which would make a run-to-run comparison meaningless.
Tracker::Config MakeSyncConfig(bool enable_slam) {
  Tracker::Config cfg;
  cfg.odometry.async_sba = false;
  if (enable_slam) {
    cfg.slam = Slam::Config{};
    cfg.slam->sync_mode = true;
    cfg.slam->enable_reading_internals = true;
  }
  return cfg;
}

class TrackerTest : public testing::Test {
protected:
  Rig rig{MakeStereoRig()};
  SyntheticStereoSequence sequence{kNumFrames};
};

TEST_F(TrackerTest, SlamIsDisabledByDefault) {
  Tracker tracker{rig};

  EXPECT_FALSE(tracker.IsSlamEnabled());
  EXPECT_EQ(tracker.GetSlam(), nullptr);

  const auto result = tracker.Track(sequence.GetFrame(0));
  EXPECT_TRUE(result.odometry.world_from_rig.has_value());
  EXPECT_FALSE(result.slam.has_value());

  EXPECT_TRUE(tracker.GetAllSlamPoses().empty());
  EXPECT_TRUE(tracker.GetLoopClosurePoses().empty());
  EXPECT_FALSE(tracker.GetSlamMetrics().has_value());
}

TEST_F(TrackerTest, SaveMapWithoutSlamReportsFailure) {
  Tracker tracker{rig};

  bool called = false;
  bool success = true;
  tracker.SaveMap("unused_folder", [&](bool result) {
    called = true;
    success = result;
  });

  EXPECT_TRUE(called);
  EXPECT_FALSE(success);
}

TEST_F(TrackerTest, LocalizeInMapWithoutSlamThrows) {
  Tracker tracker{rig};

  EXPECT_THROW(tracker.LocalizeInMap(
                   "unused_folder", kFramePeriodNs, Pose{}, sequence.GetFrame(0), Slam::LocalizationSettings{}, [] {},
                   [](const cuvslam::Result<Pose>&) {}),
               std::invalid_argument);
}

TEST_F(TrackerTest, SlamIsTrackedWhenEnabled) {
  Tracker tracker{rig, MakeSyncConfig(/*enable_slam=*/true)};

  EXPECT_TRUE(tracker.IsSlamEnabled());
  ASSERT_NE(tracker.GetSlam(), nullptr);

  Pose last_odometry_pose;
  Pose last_slam_pose;
  for (size_t frame = 0; frame < kNumFrames; ++frame) {
    const auto result = tracker.Track(sequence.GetFrame(frame));
    ASSERT_TRUE(result.odometry.world_from_rig.has_value()) << "odometry lost tracking at frame " << frame;
    ASSERT_TRUE(result.slam.has_value()) << "no SLAM pose at frame " << frame;
    last_odometry_pose = result.odometry.world_from_rig->pose;
    last_slam_pose = *result.slam;
  }

  EXPECT_FALSE(tracker.GetAllSlamPoses().empty());
  EXPECT_TRUE(tracker.GetSlamMetrics().has_value());

  // Guard against the sequence degenerating into a featureless scene, which would make the
  // comparison in MatchesManualOrchestration vacuous.
  EXPECT_FALSE(tracker.GetLastObservations(0).empty());
  EXPECT_FALSE(tracker.GetLastLandmarks().empty());
  EXPECT_GT(TranslationNorm(last_odometry_pose), kExpectedTravelM / 2.f)
      << "the synthetic rig should have travelled about " << kExpectedTravelM << " m";
  EXPECT_GT(TranslationNorm(last_slam_pose), kExpectedTravelM / 2.f)
      << "SLAM should follow the odometry trajectory rather than stay at the origin";
}

// A SLAM configuration must turn on the exports SLAM depends on, whatever the odometry config said.
TEST_F(TrackerTest, SlamConfigEnablesRequiredExports) {
  Tracker::Config cfg = MakeSyncConfig(/*enable_slam=*/true);
  cfg.odometry.enable_observations_export = false;
  cfg.odometry.enable_landmarks_export = false;

  Tracker tracker{rig, cfg};
  tracker.Track(sequence.GetFrame(0));

  EXPECT_NO_THROW(tracker.GetLastObservations(0));
  EXPECT_NO_THROW(tracker.GetLastLandmarks());

  // The caller's config is an input, not scratch space.
  EXPECT_FALSE(cfg.odometry.enable_observations_export);
  EXPECT_FALSE(cfg.odometry.enable_landmarks_export);
}

// Without SLAM the exports stay as configured, which is what makes the test above meaningful.
TEST_F(TrackerTest, ExportsStayDisabledWithoutSlam) {
  Tracker::Config cfg = MakeSyncConfig(/*enable_slam=*/false);
  cfg.odometry.enable_observations_export = false;
  cfg.odometry.enable_landmarks_export = false;

  Tracker tracker{rig, cfg};
  tracker.Track(sequence.GetFrame(0));

  EXPECT_THROW(tracker.GetLastObservations(0), std::invalid_argument);
  EXPECT_THROW(tracker.GetLastLandmarks(), std::invalid_argument);
}

// The whole point of Tracker is to run the sequence users would otherwise write by hand, so both
// paths must produce the same trajectory.
TEST_F(TrackerTest, MatchesManualOrchestration) {
  constexpr float kTolerance = 1e-4f;

  const Tracker::Config cfg = MakeSyncConfig(/*enable_slam=*/true);

  std::vector<Tracker::TrackResult> expected;
  expected.reserve(kNumFrames);
  {
    Odometry odometry{rig, [&] {
                        Odometry::Config odom_cfg = cfg.odometry;
                        odom_cfg.enable_observations_export = true;
                        odom_cfg.enable_landmarks_export = true;
                        return odom_cfg;
                      }()};
    Slam slam{rig, odometry.GetPrimaryCameras(), *cfg.slam};
    SyntheticStereoSequence manual_sequence{kNumFrames};

    for (size_t frame = 0; frame < kNumFrames; ++frame) {
      Tracker::TrackResult result;
      result.odometry = odometry.Track(manual_sequence.GetFrame(frame));
      if (result.odometry.world_from_rig.has_value()) {
        Odometry::State state;
        odometry.GetState(state);
        slam.Track(state);
        result.slam = slam.GetPose();
      }
      expected.push_back(result);
    }
  }

  Tracker tracker{rig, cfg};
  float max_odometry_travel = 0.f;
  float max_slam_travel = 0.f;
  for (size_t frame = 0; frame < kNumFrames; ++frame) {
    const auto actual = tracker.Track(sequence.GetFrame(frame));

    ASSERT_EQ(actual.odometry.world_from_rig.has_value(), expected[frame].odometry.world_from_rig.has_value())
        << "odometry validity differs at frame " << frame;
    ASSERT_EQ(actual.slam.has_value(), expected[frame].slam.has_value()) << "SLAM validity differs at frame " << frame;
    EXPECT_EQ(actual.odometry.timestamp_ns, expected[frame].odometry.timestamp_ns);

    if (actual.odometry.world_from_rig.has_value()) {
      SCOPED_TRACE("odometry pose at frame " + std::to_string(frame));
      ExpectPoseNear(actual.odometry.world_from_rig->pose, expected[frame].odometry.world_from_rig->pose, kTolerance);
      max_odometry_travel = std::max(max_odometry_travel, TranslationNorm(actual.odometry.world_from_rig->pose));
    }
    if (actual.slam.has_value()) {
      SCOPED_TRACE("SLAM pose at frame " + std::to_string(frame));
      ExpectPoseNear(*actual.slam, *expected[frame].slam, kTolerance);
      max_slam_travel = std::max(max_slam_travel, TranslationNorm(*actual.slam));
    }
  }

  // Comparing two stationary trajectories would prove nothing.
  EXPECT_GT(max_odometry_travel, kExpectedTravelM / 2.f);
  EXPECT_GT(max_slam_travel, kExpectedTravelM / 2.f);
}

TEST_F(TrackerTest, ExposesUnderlyingComponents) {
  Tracker tracker{rig, MakeSyncConfig(/*enable_slam=*/true)};

  EXPECT_FALSE(tracker.GetOdometry().GetPrimaryCameras().empty());

  tracker.Track(sequence.GetFrame(0));

  // Data layer reading is reached through the SLAM accessor rather than mirrored on Tracker.
  Slam* slam = tracker.GetSlam();
  ASSERT_NE(slam, nullptr);
  slam->EnableReadingData(Slam::DataLayer::Landmarks, 1000);
  EXPECT_NO_THROW(slam->ReadLandmarks(Slam::DataLayer::Landmarks));
}

TEST_F(TrackerTest, MovedTrackerKeepsTracking) {
  Tracker tracker{rig, MakeSyncConfig(/*enable_slam=*/true)};
  tracker.Track(sequence.GetFrame(0));

  Tracker moved{std::move(tracker)};

  EXPECT_TRUE(moved.IsSlamEnabled());
  const auto result = moved.Track(sequence.GetFrame(1));
  EXPECT_TRUE(result.odometry.world_from_rig.has_value());
  EXPECT_TRUE(result.slam.has_value());
}

}  // namespace
