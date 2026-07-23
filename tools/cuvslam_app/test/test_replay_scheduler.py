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

import pathlib
import sys
import unittest
from types import SimpleNamespace

import numpy as np

APP_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))

from dataset_reader import DatasetReader, ReplayScheduler  # noqa: E402
from edex_reader import EdexReader  # noqa: E402


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def advance(self, duration):
        self.now += duration

    def sleep(self, duration):
        self.sleeps.append(duration)
        self.advance(duration)


class RecordingProcessor:
    def __init__(self, clock):
        self.clock = clock
        self.events = []
        self.metadata = {}

    def process_images(self, frame_id, timestamps, images, masks, depths=None):
        self.events.append(("frame", frame_id, len(images), len(depths or [])))
        self.clock.advance(0.25)

    def process_imu(self, timestamp, linear_accelerations, angular_velocities):
        self.events.append(("imu", timestamp))

    def get_camera_pose(self, frame_id):
        return None

    def set_frame_metadata(self, frame_id, metadata):
        self.metadata[frame_id] = metadata


class ReplaySchedulerTest(unittest.TestCase):
    def test_waits_until_next_deadline(self):
        clock = FakeClock()
        scheduler = ReplayScheduler(10.0, True, clock=clock, sleeper=clock.sleep)

        self.assertEqual(scheduler.wait_for_frame(), (0, 0))
        self.assertEqual(scheduler.wait_for_frame(), (0, 1))
        self.assertAlmostEqual(clock.sleeps[0], 0.1)

    def test_selects_latest_due_frame(self):
        clock = FakeClock()
        scheduler = ReplayScheduler(10.0, True, clock=clock, sleeper=clock.sleep)

        self.assertEqual(scheduler.wait_for_frame(), (0, 0))
        clock.advance(0.25)

        self.assertEqual(scheduler.wait_for_frame(), (1, 2))

    def test_can_pace_without_dropping(self):
        clock = FakeClock()
        scheduler = ReplayScheduler(10.0, False, clock=clock, sleeper=clock.sleep)

        self.assertEqual(scheduler.wait_for_frame(), (0, 0))
        clock.advance(0.25)

        self.assertEqual(scheduler.wait_for_frame(), (0, 1))


class EdexReplayDroppingTest(unittest.TestCase):
    def test_drops_whole_multicamera_frame_but_keeps_imu(self):
        clock = FakeClock()
        reader = EdexReader.__new__(EdexReader)
        DatasetReader.__init__(
            reader, "", target_fps=10.0, drop_late_frames=True)
        reader.replay_scheduler = ReplayScheduler(
            10.0, True, clock=clock, sleeper=clock.sleep)
        reader.rig = SimpleNamespace(cameras=[
            SimpleNamespace(size=(2, 2)),
            SimpleNamespace(size=(2, 2)),
        ])
        reader.total_frames = 3
        reader.current_frame = 0
        reader.frame_id_start = 0
        reader.fps = 10
        reader.first_ts = 1_700_000_000_000_000_000
        reader.last_ts = reader.first_ts + 200_000_000
        reader.max_ts = reader.last_ts
        reader.sequence_duration = reader.last_ts - reader.first_ts
        reader.depth_enabled = True
        reader.depth_camera_ids = [0, 1]
        reader.rgbd_settings = SimpleNamespace(depth_scale_factor=1000.0)
        reader.edex_dir = "/unused"
        reader.tar_archives = {}
        reader.cache_uncompressed = False
        reader.frames = {}

        for frame_id in range(3):
            timestamp = reader.first_ts + frame_id * 100_000_000
            reader.frames[frame_id] = {
                "cams": [
                    {"id": 0, "filename": f"cam0/{frame_id}.png", "timestamp": timestamp},
                    {"id": 1, "filename": f"cam1/{frame_id}.png", "timestamp": timestamp},
                ],
                "depth": [
                    {"id": 0, "filename": f"depth0/{frame_id}.npy", "timestamp": timestamp},
                    {"id": 1, "filename": f"depth1/{frame_id}.npy", "timestamp": timestamp},
                ],
                "cams_min_ts": timestamp,
                "cams_max_ts": timestamp,
                "imu_data": [{
                    "timestamp": timestamp + 50_000_000,
                    "LinearAccelerationX": 1.0,
                    "LinearAccelerationY": 2.0,
                    "LinearAccelerationZ": 3.0,
                    "AngularVelocityX": 4.0,
                    "AngularVelocityY": 5.0,
                    "AngularVelocityZ": 6.0,
                }],
            }

        reader.load_image = lambda *args, **kwargs: np.zeros((2, 2), dtype=np.uint8)
        reader.load_depth_image = lambda *args, **kwargs: np.ones((2, 2), dtype=np.uint16)
        processor = RecordingProcessor(clock)

        reader.replay(processor)

        frame_events = [event for event in processor.events if event[0] == "frame"]
        imu_events = [event for event in processor.events if event[0] == "imu"]
        self.assertEqual(frame_events, [
            ("frame", 0, 2, 2),
            ("frame", 2, 2, 2),
        ])
        self.assertEqual(len(imu_events), 3)
        self.assertEqual(
            [event[0:2] for event in processor.events],
            [("frame", 0), ("imu", reader.first_ts + 50_000_000),
             ("imu", reader.first_ts + 150_000_000), ("frame", 2),
             ("imu", reader.first_ts + 250_000_000)])
        self.assertFalse(processor.metadata[0]["dropped"])
        self.assertTrue(processor.metadata[1]["dropped"])
        self.assertFalse(processor.metadata[2]["dropped"])
        self.assertEqual(reader.num_dropped_frames, 1)
        self.assertEqual(reader.max_consecutive_dropped_frames, 1)


if __name__ == "__main__":
    unittest.main()
