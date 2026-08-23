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

#include "pipelines/service_sba.h"
#include "common/include_gtest.h"

namespace cuvslam::pipelines {
namespace {

using map::LandmarkPtr;
using map::UnifiedMap;

// Landmarks are identified only by pointer identity in FindNewestVisualComponentStart, but they
// must carry a pose to count: a landmark without one is skipped exactly as it is in run_imu_sba.
LandmarkPtr MakeLandmark(TrackId id) { return std::make_shared<map::Landmark>(id, Vector3T::Zero()); }

LandmarkPtr MakePoselessLandmark(TrackId id) { return std::make_shared<map::Landmark>(id); }

// Builds a window from a per-keyframe list of landmarks. Observations are left empty; the helper
// only looks at landmark identity.
UnifiedMap::SubMap MakeWindow(const std::vector<std::vector<LandmarkPtr>>& per_keyframe) {
  UnifiedMap::SubMap sub_map;
  sub_map.landmark_and_obs.resize(per_keyframe.size());
  for (size_t i = 0; i < per_keyframe.size(); i++) {
    sub_map.consecutive_keyframes.push_back({std::make_shared<map::KeyFrame>(static_cast<int64_t>(i)), nullptr});
    for (const auto& landmark : per_keyframe[i]) {
      sub_map.landmark_and_obs[i].try_push_back({landmark, {}});
    }
  }
  return sub_map;
}

TEST(FindNewestVisualComponentStart, EmptyWindowIsConnected) {
  EXPECT_EQ(FindNewestVisualComponentStart(MakeWindow({})), 0u);
}

TEST(FindNewestVisualComponentStart, SingleKeyframeIsConnected) {
  auto a = MakeLandmark(1);
  EXPECT_EQ(FindNewestVisualComponentStart(MakeWindow({{a}})), 0u);
}

TEST(FindNewestVisualComponentStart, FullyConnectedWindowReturnsZero) {
  auto a = MakeLandmark(1);
  auto b = MakeLandmark(2);
  auto c = MakeLandmark(3);
  // Each keyframe shares at least one landmark with its predecessor.
  EXPECT_EQ(FindNewestVisualComponentStart(MakeWindow({{a, b}, {b, c}, {c, a}})), 0u);
}

TEST(FindNewestVisualComponentStart, TransitiveSharingCountsAsConnected) {
  auto a = MakeLandmark(1);
  auto b = MakeLandmark(2);
  // The third keyframe shares nothing with the second but does with the first, which is already
  // in the component -- connectivity is to the component, not to the immediate predecessor.
  EXPECT_EQ(FindNewestVisualComponentStart(MakeWindow({{a}, {a, b}, {a}})), 0u);
}

TEST(FindNewestVisualComponentStart, IsolatedNewestKeyframeSplitsAtTheEnd) {
  auto a = MakeLandmark(1);
  auto fresh = MakeLandmark(2);
  // The post-blackout case: every TrackId died, so the recovery keyframe shares nothing.
  EXPECT_EQ(FindNewestVisualComponentStart(MakeWindow({{a}, {a}, {a}, {fresh}})), 3u);
}

TEST(FindNewestVisualComponentStart, TrailingPairSharingOnlyWithEachOther) {
  auto a = MakeLandmark(1);
  auto fresh1 = MakeLandmark(2);
  auto fresh2 = MakeLandmark(3);
  // One keyframe after the recovery: {fresh1,fresh2} is connected internally but still cut off
  // from the anchor. A "drop only the last lonely keyframe" rule would miss this.
  EXPECT_EQ(FindNewestVisualComponentStart(MakeWindow({{a}, {a}, {fresh1}, {fresh1, fresh2}})), 2u);
}

TEST(FindNewestVisualComponentStart, TwoBreaksReturnsTheLast) {
  auto a = MakeLandmark(1);
  auto b = MakeLandmark(2);
  auto c = MakeLandmark(3);
  EXPECT_EQ(FindNewestVisualComponentStart(MakeWindow({{a}, {b}, {b}, {c}})), 3u);
}

TEST(FindNewestVisualComponentStart, LandmarksWithoutAPoseDoNotConnect) {
  auto shared_no_pose = MakePoselessLandmark(1);
  auto a = MakeLandmark(2);
  auto b = MakeLandmark(3);
  // Untriangulated landmarks are skipped by run_imu_sba, so they cannot tie two keyframes
  // together here either.
  EXPECT_EQ(FindNewestVisualComponentStart(MakeWindow({{a, shared_no_pose}, {b, shared_no_pose}})), 1u);
}

TEST(FindNewestVisualComponentStart, EmptyKeyframeSplitsTheWindow) {
  auto a = MakeLandmark(1);
  // A keyframe with no usable landmarks shares nothing by definition.
  EXPECT_EQ(FindNewestVisualComponentStart(MakeWindow({{a}, {}})), 1u);
}

}  // namespace
}  // namespace cuvslam::pipelines
