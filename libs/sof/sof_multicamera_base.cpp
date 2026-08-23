
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

#include "sof/internal/sof_multicamera_base.h"

namespace cuvslam::sof {

// Minimum number of primary->secondary matches for a frame to be usable as a keyframe. Only
// matched pairs can be triangulated, so below this the keyframe would contribute no landmarks.
// Kept deliberately low: this rejects frames that can contribute nothing, it does not try to
// judge whether a frame is a *good* anchor. See the call site in trackNextFrame().
constexpr size_t kMinStereoPairsForKeyframe = 10;

MultiSOFBase::MultiSOFBase(const camera::Rig& rig, const camera::FrustumIntersectionGraph& fid,
                           const Settings& sof_settings, const odom::KeyFrameSettings& keyframe_settings)
    : rig_(rig), fid_(fid), box_prefilter_(sof_settings.box3_prefilter), kf_selector_(keyframe_settings) {}

void MultiSOFBase::reset_keyframe_selector() {
  kf_selector_.reset();
  last_kf_tracks_.clear();
  last_kf_timestamp_ = 0;
}

bool MultiSOFBase::trackNextFrame(const Sources& curr_sources, Images& curr_images, const Images& prev_images,
                                  const Sources& masks_sources, const Isometry3T& predicted_world_from_rig,
                                  MulticamObservations& observations, FrameState& state,
                                  const odom::TrackPerFrameSettings& per_frame) {
  const sof::Settings& sof_settings = per_frame.sof;
  const odom::KeyFrameSettings& kf_settings = per_frame.kf;
  // is_mono_mode=false: the keyframe override is applied once globally in KFSelector::select
  // (see MultiSOFBase::is_keyframe), so the per-camera MonoSOF must not apply it again.
  const sof::MonoSOFFrameSettings sof_frame_settings{sof_settings, kf_settings, /*is_mono_mode=*/false};
  box_prefilter_ = sof_settings.box3_prefilter;
  TRACE_EVENT ev = profiler_domain_.trace_event("trackNextFrame", profiler_color_);

  state = FrameState::None;

  for (auto& obs : observations) {
    obs.clear();
  }
  int64_t current_time_ns = -1;
  size_t num_failed_sofs = 0;

  for (auto& sof : mono_sof_) {
    TRACE_EVENT ev1 = profiler_domain_.trace_event("mono start", profiler_color_);
    assert(sof);
    const CameraId cam_id = sof->camera_id();

    if (cam_id >= curr_images.size() || curr_images[cam_id] == nullptr) {
      num_failed_sofs++;
      continue;
    }

    const ImageSource& curr_source = curr_sources[cam_id];
    const ImageContextPtr curr_image = curr_images[cam_id];

    // should be same for all images
    current_time_ns = curr_image->get_image_meta().timestamp;

    ImageContextPtr prev_image;
    if (cam_id < prev_images.size()) {
      prev_image = prev_images[cam_id];
    }

    const ImageSource& mask_src = masks_sources[cam_id];
    sof->track({curr_source, curr_image}, prev_image, predicted_world_from_rig, sof_frame_settings, &mask_src);
  }

  if (num_failed_sofs == mono_sof_.size()) {
    return false;
  }

  // Ensure all mono-stream GPU work (pyramid builds, LK tracking) is complete
  // before reading results. Required for cross-stream memory visibility on
  // architectures like Blackwell (sm_121).
  cudaDeviceSynchronize();

  MulticamTracksVector primary_tracks;
  for (auto& sof : mono_sof_) {
    TRACE_EVENT ev1 = profiler_domain_.trace_event("mono finish", profiler_color_);
    const CameraId cam_id = sof->camera_id();
    if (cam_id >= curr_images.size() || curr_images[cam_id] == nullptr) {
      continue;
    }
    FrameState mono_state;  // TODO: launch l->r tracking for this stereo camera only if it keyframe
    const TracksVector& tracks_vector = sof->finish(mono_state, sof_frame_settings);
    primary_tracks.push_back({cam_id, std::cref(tracks_vector)});
    tracks_vector.export_to_observations_vector(*rig_.intrinsics[cam_id], observations[cam_id]);
  }

  if (is_keyframe(primary_tracks, current_time_ns, kf_settings)) {
    TRACE_EVENT ev1 = profiler_domain_.trace_event("prim->sec", profiler_color_);
    // Ensure all mono-stream GPU work (pyramid builds, feature detection) is fully
    // visible before launching stereo tracking on separate streams. Without this,
    // cross-stream memory visibility is not guaranteed on some GPU architectures
    // (e.g. Blackwell sm_121), leading to CUDA error 700 after ~200 frames.
    cudaDeviceSynchronize();
    StartKeyframe();

    // Count how many tracks get matched from a primary camera into a secondary one. Only those
    // can be triangulated into landmarks, so the count is what this frame is able to contribute
    // to the map. Taken as a before/after difference on the total observation count because the
    // two backends emit their matches in different places: the GPU path defers to
    // GetTrackingResults(), while the CPU path pushes directly from LaunchTrackingPrimaryToSecondary
    // and leaves GetTrackingResults() a no-op. Bracketing the whole section covers both.
    size_t observations_before = 0;
    for (const auto& cam_observations : observations) {
      observations_before += cam_observations.size();
    }

    const auto& primary_cams = fid_.primary_cameras();
    for (CameraId primary_cam_id : primary_cams) {
      if (primary_cam_id >= curr_images.size() || curr_images[primary_cam_id] == nullptr) {
        continue;
      }
      const auto& secondary_cams = fid_.secondary_cameras(primary_cam_id);
      for (CameraId secondary_cam_id : secondary_cams) {
        if (secondary_cam_id >= curr_images.size() || curr_images[secondary_cam_id] == nullptr) {
          continue;
        }
        LaunchTrackingPrimaryToSecondary(primary_cam_id, secondary_cam_id, curr_sources, curr_images,
                                         observations[primary_cam_id], &observations[secondary_cam_id]);
      }
    }
    GetTrackingResults(observations);

    size_t observations_after = 0;
    for (const auto& cam_observations : observations) {
      observations_after += cam_observations.size();
    }
    const size_t stereo_pairs = observations_after - observations_before;

    if (stereo_pairs < kMinStereoPairsForKeyframe) {
      // No track was seen by two cameras, so triangulate() would produce nothing and this frame
      // cannot anchor anything. Degraded images are the common cause, but a failed or badly
      // calibrated secondary camera looks the same and is handled identically.
      //
      // reset_keyframe_selector() un-consumes the request that is_keyframe() just recorded, so the
      // next frame is offered again instead of waiting for 41% track attrition or the 60 s
      // backstop. Note this is not a quality gate: SOF cannot see which tracks already carry
      // landmarks (that lives in the solver's MulticamTriangulator), so it can only tell whether
      // stereo is working at all. Judging whether a keyframe is worth committing belongs in
      // SolverSfMInertial, which can see the map.
      reset_keyframe_selector();
      return true;  // state stays FrameState::None
    }
    state = FrameState::Key;
  }
  return true;
}

bool MultiSOFBase::is_keyframe(const MulticamTracksVector& tracks, const int64_t current_timestamp_ns,
                               const odom::KeyFrameSettings& kf_settings) {
  last_kf_tracks_vec_.reset();
  all_tracks_vec_.reset();
  for (const auto& [cam_id, tracks_vector] : tracks) {
    all_tracks_vec_.add(tracks_vector);

    const auto& last_kf_vec = last_kf_tracks_[cam_id];
    last_kf_tracks_vec_.add(last_kf_vec);
  }

  all_tracks_vec_.sort();
  last_kf_tracks_vec_.sort();

  if (kf_selector_.select(all_tracks_vec_, current_timestamp_ns, last_kf_tracks_vec_, last_kf_timestamp_,
                          kf_settings)) {
    for (const auto& [cam_id, tracks_vector] : tracks) {
      last_kf_tracks_[cam_id] = tracks_vector;
    }
    last_kf_timestamp_ = current_timestamp_ns;
    return true;
  }
  return false;
}

}  // namespace cuvslam::sof
