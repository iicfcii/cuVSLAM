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

import os
import time
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Tuple
import numpy as np

import cuvslam as vslam
import conversions as conv


class Processing(Protocol):
    """Protocol defining interface for data processing."""

    def process_images(
        self,
        frame_id: int,
        timestamps: Sequence[int],
        images: Sequence[np.ndarray],
        masks: Sequence[np.ndarray],
        depths: Optional[Sequence[np.ndarray]] = None
    ) -> None:
        """Process image data for a single frame."""
        ...

    def process_imu(
        self,
        timestamp: int,
        linear_accelerations: Sequence[float],
        angular_velocities: Sequence[float]
    ) -> None:
        """Process IMU data."""
        ...

    def get_camera_pose(self, frame_id: int) -> Optional[vslam.Pose]:
        """Get camera pose for given frame."""
        ...

    def set_frame_metadata(self, frame_id: int, metadata: Dict) -> None:
        """Set metadata for a frame."""
        ...


class ReplayScheduler:
    """Pace replay against a monotonic clock and select the latest due frame."""

    def __init__(self, target_fps: float, drop_late_frames: bool,
                 clock: Callable[[], float] = time.monotonic,
                 sleeper: Callable[[float], None] = time.sleep):
        if target_fps <= 0:
            raise ValueError("target_fps must be greater than zero")
        self.period_s = 1.0 / target_fps
        self.drop_late_frames = drop_late_frames
        self.clock = clock
        self.sleeper = sleeper
        self.reset()

    def reset(self) -> None:
        self.start_time: Optional[float] = None
        self.next_tick = 0

    def wait_for_frame(self) -> Tuple[int, int]:
        """Return (number of stale frames to drop, scheduled tick to process)."""
        now = self.clock()
        if self.start_time is None:
            self.start_time = now

        scheduled_tick = self.next_tick
        deadline = self.start_time + scheduled_tick * self.period_s
        if now < deadline:
            self.sleeper(deadline - now)
            now = self.clock()

        dropped_frames = 0
        if self.drop_late_frames:
            due_tick = int(max(0.0, now - self.start_time) / self.period_s)
            if due_tick > scheduled_tick:
                dropped_frames = due_tick - scheduled_tick
                scheduled_tick = due_tick

        self.next_tick = scheduled_tick + 1
        return dropped_frames, scheduled_tick


class DatasetReader:
    def __init__(self, dataset_path: str, stereo_edex: Optional[str] = None,
                 num_loops: int = 0, repeat_type: str = "none",
                 gt_path: Optional[str] = None, target_fps: float = 0.0,
                 drop_late_frames: bool = False):
        repeat_type = (repeat_type or "none").lower()
        if repeat_type not in ("none", "repeat", "shuttle"):
            raise ValueError(f"Invalid repeat_type {repeat_type!r}; expected none|repeat|shuttle")
        if target_fps < 0:
            raise ValueError("target_fps must be non-negative")
        if drop_late_frames and target_fps == 0:
            raise ValueError("drop_late_frames requires target_fps to be greater than zero")
        self.dataset_path = dataset_path
        self.stereo_edex = stereo_edex
        self.num_loops = num_loops
        self.repeat_type = repeat_type
        self.replay_scheduler = (
            ReplayScheduler(target_fps, drop_late_frames) if target_fps > 0 else None
        )
        self.num_dropped_frames = 0
        self.max_consecutive_dropped_frames = 0
        self.rig: Optional[vslam.Rig] = None
        self.total_frames = 0
        self.fps = 0
        self.max_ts = 0
        self.sequence_duration = 0
        self.current_frame = 0
        self.frame_id_start = 0
        self.current_loop = 0
        self.replay_forward = True
        self.gt_from_shuttle = False
        self.gt_transforms = []

        if gt_path is None:
            self.gt_path = os.path.join(dataset_path, 'gt.txt')
        elif os.path.isabs(gt_path):
            self.gt_path = gt_path
        else:
            self.gt_path = os.path.join(dataset_path, gt_path)
        if os.path.exists(self.gt_path):
            with open(self.gt_path, 'r') as f:
                for line in f:
                    transform_12 = [float(value) for value in line.split()]
                    assert len(transform_12) == 12, "Each line should have 12 float values"
                    transform = np.vstack((np.array(transform_12).reshape(3, 4), np.array([0, 0, 0, 1])))
                    self.gt_transforms.append(transform)
        elif self.repeat_type == "shuttle" and self.num_loops > 0:
            # consider first forward run in shuttle mode as ground truth
            self.gt_from_shuttle = True

    @staticmethod
    def get_camera(intrinsics: Dict, transform: List[List[float]]) -> vslam.Camera:
        """Create camera from intrinsics and transform."""
        if not all(len(intrinsics[k]) == 2 for k in ['focal', 'principal', 'size']):
            raise ValueError("Camera intrinsics are invalid.")

        cam = vslam.Camera()
        cam.distortion = vslam.Distortion(
            conv.to_distortion_model(intrinsics['distortion_model']),
            intrinsics['distortion_params']
        )
        cam.focal = intrinsics['focal']
        cam.principal = intrinsics['principal']
        cam.size = intrinsics['size']
        cam.rig_from_camera = conv.transform_to_pose(transform)
        return cam

    @staticmethod
    def get_imu_calibration(imu_data: Dict) -> vslam.ImuCalibration:
        """Create IMU calibration from data."""
        if 'measurements' not in imu_data:
            raise ValueError("IMU calibration is provided, but data is absent.")

        imu = vslam.ImuCalibration(rig_from_imu=conv.transform_to_pose(imu_data['transform']),
                                   gyroscope_noise_density=0.00016968,
                                   gyroscope_random_walk=0.000019393,
                                   accelerometer_noise_density=0.002,
                                   accelerometer_random_walk=0.003,
                                   frequency=200)
        if 'gyroscope_noise_density' in imu_data:
            imu.gyroscope_noise_density = imu_data['gyroscope_noise_density']
        if 'gyroscope_random_walk' in imu_data:
            imu.gyroscope_random_walk = imu_data['gyroscope_random_walk']
        if 'accelerometer_noise_density' in imu_data:
            imu.accelerometer_noise_density = imu_data['accelerometer_noise_density']
        if 'accelerometer_random_walk' in imu_data:
            imu.accelerometer_random_walk = imu_data['accelerometer_random_walk']
        if 'frequency' in imu_data:
            imu.frequency = imu_data['frequency']
        if 'measurements' not in imu_data:
            raise ValueError("IMU calibration is provided, but data is absent.")
        return imu

    @staticmethod
    def get_rig(config: Dict) -> vslam.Rig:
        """Create rig from configuration."""
        rig = vslam.Rig()
        rig.cameras = [
            DatasetReader.get_camera(cam['intrinsics'], cam['transform'])
            for cam in config['cameras']
        ]
        rig.imus = [DatasetReader.get_imu_calibration(config['imu'])] if 'imu' in config else []
        return rig

    @staticmethod
    def parse_config(config_data: dict) -> vslam.Rig:
        try:
            rig = DatasetReader.get_rig(config_data[0])
            print("Rig initialized successfully.")
            return rig
        except Exception as e:
            print(f"Error initializing Rig: {e}")
            raise

    def validate_rig(self):
        """Validate rig parameters against requirements."""
        if self.rig is None:
            return False

        for i, camera in enumerate(self.rig.cameras):
            if not all(p >= 0 for p in camera.principal):  # type: ignore
                print(f"Camera {i}: Principal point must be >= 0.0 ({camera.principal})")
                return False

            if not all(f > 0 for f in camera.focal):  # type: ignore
                print(f"Camera {i}: Focal length must be > 0.0 ({camera.focal})")
                return False

            if not all(s > 0 for s in camera.size):  # type: ignore
                print(f"Camera {i}: Image size must be > 0.0 ({camera.size})")
                return False

        return True

    def reset(self) -> None:
        """Reset reader state for a new replay."""
        self.current_frame = 0
        self.current_loop = 0
        self.replay_forward = True
        self.num_dropped_frames = 0
        self.max_consecutive_dropped_frames = 0
        if self.replay_scheduler is not None:
            self.replay_scheduler.reset()
        if self.gt_from_shuttle:
            self.gt_transforms = []

    def check_end_of_sequence(self) -> bool:
        """Check if we should continue playing or switch direction."""
        # Single playback
        if self.repeat_type == "none" or self.num_loops == 0:
            if self.current_frame >= self.total_frames:
                return False
            return True

        if self.repeat_type == "repeat":
            # Forward-only: rewind to start without flipping
            if self.current_frame >= self.total_frames:
                self.current_loop += 1
                self.current_frame = self.frame_id_start
            return self.current_loop < self.num_loops

        # Shuttle: forward + backward per loop
        reached_end = False
        if self.replay_forward and self.current_frame >= self.total_frames:
            reached_end = True
        elif not self.replay_forward and self.current_frame < self.frame_id_start:
            reached_end = True

        if reached_end:
            if not self.replay_forward:
                self.current_loop += 1
            self.replay_forward = not self.replay_forward
            self.current_frame = (
                self.total_frames - 1 if not self.replay_forward
                else self.frame_id_start
            )

        return self.current_loop < self.num_loops

    def adjust_timestamp(self, timestamp: int) -> int:
        """Adjust timestamp for shuttle/repeat playback so timestamps stay monotonic."""
        # Time shift by 1 frame
        delta = int(1e9 / self.fps)

        if self.repeat_type == "none" or self.num_loops == 0:
            return timestamp

        if self.repeat_type == "repeat":
            return timestamp + self.current_loop * (self.sequence_duration + delta)

        # Shuttle: alternate direction, two passes per loop
        if self.replay_forward:
            adjusted_ts = timestamp
        else:
            adjusted_ts = 2 * self.max_ts + delta - timestamp
        adjusted_ts += 2 * self.current_loop * (self.sequence_duration + delta)
        return adjusted_ts
