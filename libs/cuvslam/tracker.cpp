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

#include "cuvslam/cuvslam2.h"

#include <memory>
#include <stdexcept>
#include <utility>

namespace cuvslam {

namespace {

Odometry::Config WithSlamExports(Odometry::Config cfg) {
  cfg.enable_observations_export = true;
  cfg.enable_landmarks_export = true;
  return cfg;
}

}  // namespace

class Tracker::Impl {
public:
  Impl(const Rig& rig, const Config& cfg)
      : odometry{rig, cfg.slam.has_value() ? WithSlamExports(cfg.odometry) : cfg.odometry} {
    if (cfg.slam.has_value()) {
      slam = std::make_unique<Slam>(rig, odometry.GetPrimaryCameras(), *cfg.slam);
    }
  }

  Odometry odometry;
  std::unique_ptr<Slam> slam;
  // Reused across frames so that the per-frame observation and landmark vectors keep their capacity
  Odometry::State state;
};

Tracker::Tracker(const Rig& rig, const Config& cfg) : impl{std::make_unique<Impl>(rig, cfg)} {}

Tracker::Tracker(Tracker&&) noexcept = default;

Tracker::~Tracker() = default;

Tracker::TrackResult Tracker::Track(const ImageSet& images, const ImageSet& masks, const ImageSet& depths,
                                    const Pose* gt_pose, const cuvslam::internal::Internals* internals) {
  TrackResult result;
  result.odometry = impl->odometry.Track(images, masks, depths, internals);

  // Odometry state is only meaningful once odometry has produced a pose, so SLAM stays untouched
  // on a lost frame and keeps its previous pose.
  if (impl->slam && result.odometry.world_from_rig.has_value()) {
    impl->odometry.GetState(impl->state);
    impl->slam->Track(impl->state, gt_pose);
    result.slam = impl->slam->GetPose();
  }

  return result;
}

void Tracker::RegisterImuMeasurement(uint32_t sensor_index, const ImuMeasurement& imu) {
  impl->odometry.RegisterImuMeasurement(sensor_index, imu);
}

std::vector<Observation> Tracker::GetLastObservations(uint32_t camera_index) const {
  return impl->odometry.GetLastObservations(camera_index);
}

std::vector<Landmark> Tracker::GetLastLandmarks() const { return impl->odometry.GetLastLandmarks(); }

std::optional<Odometry::Gravity> Tracker::GetLastGravity() const { return impl->odometry.GetLastGravity(); }

std::unordered_map<uint64_t, Vector3f> Tracker::GetFinalLandmarks() const { return impl->odometry.GetFinalLandmarks(); }

bool Tracker::IsSlamEnabled() const { return impl->slam != nullptr; }

std::vector<PoseStamped> Tracker::GetAllSlamPoses(uint32_t max_poses_count) const {
  std::vector<PoseStamped> poses;
  if (impl->slam) {
    impl->slam->GetAllSlamPoses(poses, max_poses_count);
  }
  return poses;
}

void Tracker::SaveMap(const std::string_view& folder_name, std::function<void(bool success)> callback) const {
  if (!impl->slam) {
    callback(false);
    return;
  }
  impl->slam->SaveMap(folder_name, std::move(callback));
}

void Tracker::LocalizeInMap(const std::string_view& folder_name, int64_t timestamp_ns, const Pose& guess_pose,
                            const ImageSet& images, const Slam::LocalizationSettings& settings,
                            Slam::LocalizeStartCB start_cb, Slam::LocalizeFinishCB finish_cb) {
  if (!impl->slam) {
    throw std::invalid_argument{"SLAM is not enabled"};
  }
  impl->slam->LocalizeInMap(folder_name, timestamp_ns, guess_pose, images, settings, std::move(start_cb),
                            std::move(finish_cb));
}

std::optional<Slam::Metrics> Tracker::GetSlamMetrics() const {
  if (!impl->slam) {
    return std::nullopt;
  }
  Slam::Metrics metrics;
  impl->slam->GetSlamMetrics(metrics);
  return metrics;
}

std::vector<PoseStamped> Tracker::GetLoopClosurePoses() const {
  std::vector<PoseStamped> poses;
  if (impl->slam) {
    impl->slam->GetLoopClosurePoses(poses);
  }
  return poses;
}

Odometry& Tracker::GetOdometry() { return impl->odometry; }

const Odometry& Tracker::GetOdometry() const { return impl->odometry; }

Slam* Tracker::GetSlam() { return impl->slam.get(); }

const Slam* Tracker::GetSlam() const { return impl->slam.get(); }

}  // namespace cuvslam
