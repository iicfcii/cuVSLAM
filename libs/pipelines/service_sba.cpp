
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

#include <unordered_set>

namespace cuvslam::pipelines {

int CalcNumFixedKeyframes(size_t map_size, size_t numFixedKeyFrames) {
  if (map_size < 2) {
    return 0;
  }

  if (map_size < numFixedKeyFrames + 1) {
    // We don't have enough key frames to fix numFixedKeyFrames of them.
    // Fix just one to constrain the problem.
    return 1;
  }

  return numFixedKeyFrames;
}

size_t FindNewestVisualComponentStart(const UnifiedMap::SubMap& sub_map) {
  const size_t n = sub_map.landmark_and_obs.size();
  if (n < 2) {
    return 0;
  }

  // Walk the window forward, carrying the landmark set of the component built so far. A keyframe
  // sharing nothing with that set cannot be tied to it by any reprojection residual, so it opens a
  // new component; the last one opened is the one containing the newest keyframe.
  size_t start = 0;
  std::unordered_set<LandmarkPtr> component;

  const auto insert_landmarks = [&component](const auto& landmarks) {
    for (const auto& [landmark, obs] : landmarks) {
      if (landmark->get_pose()) {
        component.insert(landmark);
      }
    }
  };
  const auto shares_landmark = [&component](const auto& landmarks) {
    for (const auto& [landmark, obs] : landmarks) {
      if (landmark->get_pose() && component.count(landmark) > 0) {
        return true;
      }
    }
    return false;
  };

  insert_landmarks(sub_map.landmark_and_obs[0]);
  for (size_t i = 1; i < n; i++) {
    if (!shares_landmark(sub_map.landmark_and_obs[i])) {
      start = i;
      component.clear();
    }
    insert_landmarks(sub_map.landmark_and_obs[i]);
  }
  return start;
}
}  // namespace cuvslam::pipelines
