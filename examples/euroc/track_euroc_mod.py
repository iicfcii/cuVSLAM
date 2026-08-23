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

# Experiment variant of track_euroc.py: adds an EuRoC ground-truth overlay in rerun.
#
# EuRoC ground truth (state_groundtruth_estimate0) is T_W_B, world-from-body, where the body
# frame is the IMU frame (imu0/sensor.yaml has T_BS = identity). cuVSLAM's rig origin here is
# cam0 (dataset_utils.get_rig sets cam0.rig_from_camera = identity), so ground truth is pushed
# into cam0 with T_W_cam0 = T_W_B @ T_B_cam0, using T_B_cam0 from the *original*
# cam0/sensor.yaml (the recalibrated yamls are already cam0-relative and cannot supply it).
#
# cuVSLAM starts at identity in its own frame, so ground truth is rigidly aligned to the
# odometry frame at the first frame where both exist:
#     T_align = T_odom(t_a) @ inv(T_W_cam0(t_a))
# Drift then shows up honestly as the two curves separate.

import os

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import yaml
from scipy.spatial.transform import Rotation

import cuvslam
from dataset_utils import prepare_frame_metadata_euroc, get_rig, load_frame

# Set up dataset path
sequence_path = os.path.join(os.path.dirname(__file__), "dataset/mav0")

# Ground-truth overlay settings
GT_COLOR = [255, 170, 0]        # amber, vs the default-colored odometry trajectory
ODOM_COLOR = [0, 200, 255]      # cyan, so the two curves are distinguishable
BLACKOUT_COLOR = [255, 0, 0]    # red markers at poses reported on blackout frames
# GT is logged at 200 Hz and frames at 20 Hz, so the nearest sample is always within 2.5 ms.
GT_MATCH_TOLERANCE_NS = 5_000_000

# Blackout oscillator, same semantics as libs/camera_rig_edex/blackout_oscillator_filter.cpp:
#     index = frame_id % period; blackout if index < duration and frame_id >= period
# The first BLACKOUT_PERIOD frames are exempt so the inertial state machine can reach `Ok`
# and gravity can be estimated (PR #80 gates its blackout path on gravity being available).
# The constants below are the defaults; each can be overridden from the environment so a
# batch of runs can be scripted without editing the file:
#     BLACKOUT_DURATION=5 BLACKOUT_FILL=0 RUN_TAG=nopatch python3 track_euroc_mod.py
BLACKOUT_PERIOD = int(os.environ.get('BLACKOUT_PERIOD', 500))
BLACKOUT_DURATION = int(os.environ.get('BLACKOUT_DURATION', 21))

# What a blacked-out frame is filled with.
#   0   - all zeros, the C++ filter and issue #47 reproducer. The only case PR #80's
#         IsAllZeroHostU8 detects, so the odometry-level SOF bypass kicks in.
#   255 - saturated white. NOT detected by IsAllZeroHostU8, so no SOF bypass and the frame
#         still becomes the reference for the next frame's optical flow. The solver-side
#         integrated_from_blackout condition keys on obs_vector_ being empty rather than on
#         pixel values, and a uniform patch has zero variance so KLT's NCC gate should kill
#         every track — if so, the pose still gets integrated despite the missing bypass.
BLACKOUT_FILL = int(os.environ.get('BLACKOUT_FILL', 0))

# Camera frame of the sequence to start tracking from, 0-indexed (0 = the whole sequence).
# Everything before it is dropped from the stream — the earlier images *and* the IMU samples
# that precede them — so the tracker boots from scratch at that point instead of integrating
# IMU across a span with no images to constrain it. Useful for reaching a different part of
# the trajectory (e.g. the moving section of MH_01) without replaying the run-up:
#     START_FRAME=1000 BLACKOUT_PERIOD=500 python3 track_euroc_mod.py
# The blackout oscillator and the camera_frame column both count from the first *tracked*
# frame, so a given START_FRAME/BLACKOUT_PERIOD pair blacks out the same relative ticks.
START_FRAME = int(os.environ.get('START_FRAME', 0))

# Camera frame of the sequence to stop at, exclusive, counted in the same source-frame index
# as START_FRAME; 0 means run to the end. The pair behaves like a Python slice
# [START_FRAME, END_FRAME), so a short window around one event is
#     START_FRAME=300 END_FRAME=820 python3 track_euroc_mod.py
# which is cheap enough to pair with VERBOSITY=3 without producing a 18k-line log.
END_FRAME = int(os.environ.get('END_FRAME', 1100))
if END_FRAME and END_FRAME <= START_FRAME:
    raise SystemExit(f"END_FRAME ({END_FRAME}) must be greater than START_FRAME ({START_FRAME})")

# Per-frame diagnostics are written here, one CSV per run. The filename encodes the run
# configuration so a directory of runs is self-describing. Set RUN_DIR = None to skip saving.
# RUN_TAG is appended to the filename to distinguish runs that share a configuration but not
# a build (e.g. the same blackout settings before and after a solver change).
# Kept under dataset/ because that path is already covered by the root .gitignore, so run
# output never shows up as untracked noise in git status.
RUN_DIR = os.path.join(os.path.dirname(__file__), "dataset", "runs")
RUN_TAG = os.environ.get('RUN_TAG', '')

# Spawning the rerun viewer needs a display; set CUVSLAM_RERUN_SPAWN=0 for headless batches.
SPAWN_VIEWER = os.environ.get('CUVSLAM_RERUN_SPAWN', '1') == '1'

# Library log level (libs/common/log.h: None=0, Error=1, Warning=2, Message=3, Debug=4).
# 3 is the useful one — it turns on the per-frame TraceMessage lines from the inertial solver
# ("Frame: pnp=.. integrated=..", "[PRE-PnP]", "[INERTIAL PnP]", "[POSE OUT] .. diff=.."), about
# 5 lines per frame. TraceDebug (4) is compiled out of Release builds, so 4 adds nothing here.
# Output is printf to stdout, not sys.stdout, so redirect at the shell to capture it. When
# redirecting, libc switches the library's stdout to full buffering while Python keeps its own
# buffer on the same fd, so the two interleave wrongly; stdbuf fixes the C side:
#     VERBOSITY=3 stdbuf -oL python3 track_euroc_mod.py > trace.log 2>&1
# On a terminal libc is already line-buffered and stdbuf is unnecessary. When VERBOSITY is on,
# each track() call is preceded by an "=== frame N ... ===" banner so every [MESSAGE] line
# after it belongs to that frame.
VERBOSITY = int(os.environ.get('VERBOSITY', 3))


def color_from_id(identifier):
    """Generate pseudo-random color from integer identifier for visualization."""
    return [
        (identifier * 17) % 256,
        (identifier * 31) % 256,
        (identifier * 47) % 256
    ]


def load_body_from_cam0(euroc_path):
    """Read T_B_cam0 from the original (non-recalibrated) cam0/sensor.yaml."""
    sensor_path = os.path.join(euroc_path, 'cam0', 'sensor.yaml')
    with open(sensor_path, 'r') as stream:
        config = yaml.safe_load(stream)
    return np.asarray(config['T_BS']['data'], dtype=np.float64).reshape(4, 4)


def load_groundtruth(euroc_path):
    """Load EuRoC ground truth as (timestamps_ns, T_W_B stack)."""
    gt_path = os.path.join(euroc_path, 'state_groundtruth_estimate0', 'data.csv')
    if not os.path.exists(gt_path):
        return None, None

    raw = np.loadtxt(gt_path, delimiter=',', skiprows=1)
    timestamps = raw[:, 0].astype(np.int64)

    # Columns: p_RS_R_{xyz} then q_RS_{wxyz}; scipy wants [x, y, z, w].
    positions = raw[:, 1:4]
    quaternions_wxyz = raw[:, 4:8]
    rotations = Rotation.from_quat(quaternions_wxyz[:, [1, 2, 3, 0]]).as_matrix()

    poses = np.tile(np.eye(4), (len(timestamps), 1, 1))
    poses[:, :3, :3] = rotations
    poses[:, :3, 3] = positions
    return timestamps, poses


def groundtruth_at(timestamps, poses, timestamp_ns):
    """Nearest ground-truth pose to timestamp_ns, or None outside GT coverage."""
    index = int(np.searchsorted(timestamps, timestamp_ns))
    candidates = [i for i in (index - 1, index) if 0 <= i < len(timestamps)]
    if not candidates:
        return None
    best = min(candidates, key=lambda i: abs(int(timestamps[i]) - timestamp_ns))
    if abs(int(timestamps[best]) - timestamp_ns) > GT_MATCH_TOLERANCE_NS:
        return None
    return poses[best]


def pose_to_matrix(pose):
    """cuvslam.Pose (quaternion is [x, y, z, w]) to a 4x4 matrix."""
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(np.asarray(pose.rotation)).as_matrix()
    matrix[:3, 3] = np.asarray(pose.translation)
    return matrix


# Setup rerun visualizer
rr.init("cuVSLAM Visualizer", spawn=SPAWN_VIEWER)

# Setup coordinate basis for root, cuvslam uses right-hand system with
# X-right, Y-down, Z-forward
rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

# Setup rerun views
blueprint = rrb.Blueprint(
    rrb.TimePanel(state="collapsed"),
    rrb.Horizontal(
        column_shares=[0.5, 0.5],
        contents=[
            rrb.Vertical(contents=[
                rrb.Horizontal(contents=[
                    rrb.Spatial2DView(origin='world/camera_0'),
                    rrb.Spatial2DView(origin='world/camera_1')
                ]),
                rrb.Vertical(contents=[
                    rrb.TimeSeriesView(
                    name="IMU Acceleration",
                    origin="world/imu/accel",
                    overrides={
                        "world/imu/accel/x": rr.SeriesLines(colors=[255, 0, 0]),
                        "world/imu/accel/y": rr.SeriesLines(colors=[0, 255, 0]),
                        "world/imu/accel/z": rr.SeriesLines(colors=[0, 0, 255]),
                    },
                ),
                rrb.TimeSeriesView(
                    name="IMU Angular Velocity",
                    origin="world/imu/gyro",
                    overrides={
                        "world/imu/gyro/x": rr.SeriesLines(colors=[255, 0, 0]),
                        "world/imu/gyro/y": rr.SeriesLines(colors=[0, 255, 0]),
                        "world/imu/gyro/z": rr.SeriesLines(colors=[0, 0, 255]),
                    },
                )
                ])
            ]),
            rrb.Spatial3DView(origin='world')
        ]
    )
)
rr.send_blueprint(blueprint)

# Available tracking modes:
# 0: Multicamera - Visual tracking using stereo camera (can be extended to multiple stereo cameras)
# 1: Inertial - Visual-inertial tracking using stereo camera + IMU
# 2: RGBD - Visual tracking using monocular camera + depth (supports grayscale input)
# 3: Mono - Visual tracking using monocular camera (without scale, accurate rotation only)

euroc_tracking_mode = cuvslam.Odometry.OdometryMode.Inertial

# Set before the tracker is built so construction-time messages are covered too.
if VERBOSITY:
    cuvslam.set_verbosity(VERBOSITY)
    print(f"Library verbosity set to {VERBOSITY}")

# Configure tracker
cfg = cuvslam.Odometry.Config(
    async_sba=False,
    enable_observations_export=True,
    enable_landmarks_export=True,   # needed for the per-frame landmark count below
    enable_final_landmarks_export=True,
    rectified_stereo_camera=False,
    odometry_mode=euroc_tracking_mode
)

# Get camera rig
rig = get_rig(sequence_path)

# Initialize tracker
tracker = cuvslam.Tracker(rig, cfg)
print(f"cuVSLAM Tracker initilized with odometry mode: {cfg.odometry_mode}")

# Load ground truth for the overlay
body_from_cam0 = load_body_from_cam0(sequence_path)
gt_timestamps, gt_poses = load_groundtruth(sequence_path)
if gt_timestamps is None:
    print("Warning: no ground truth found, running without the GT overlay")
else:
    print(
        f"Ground truth: {len(gt_timestamps)} samples spanning "
        f"{(gt_timestamps[-1] - gt_timestamps[0]) / 1e9:.1f} s"
    )

# Track frames
last_camera_timestamp = None
imu_count_since_last_camera = 0
frame_id = 0
trajectory = []
frames_metadata = prepare_frame_metadata_euroc(
    sequence_path, euroc_tracking_mode
)

odom_trajectory = []
gt_trajectory = []
odom_from_gt_world = None  # T_align, set at the first frame covered by GT
frames_with_gt = 0
last_odom_with_gt = None   # odometry position at the last GT-covered frame

# Camera-frame tick counter for the blackout oscillator (incremented only on
# type=='camera' entries, not on IMU entries).
camera_tick = 0
# Position in the *source* sequence, used only to fast-forward to START_FRAME.
source_camera_tick = 0
blackout_positions = []    # poses reported on blackout frames, marked red in the 3D view
blackout_frames = 0
failures_on_blackout = 0
failures_elsewhere = 0

# One record per camera frame, tracked or not. n_obs and the gravity vector are the
# Python-visible proxies for obs_vector_ and map_.get_gravity() on the C++ side.
records = []

for frame_metadata in frames_metadata:
    timestamp = frame_metadata['timestamp']

    # Fast-forward to START_FRAME, discarding IMU samples along with the images so nothing
    # from the skipped span reaches the tracker.
    if source_camera_tick < START_FRAME:
        if frame_metadata['type'] != 'imu':
            source_camera_tick += 1
        continue

    # camera_tick counts frames already fed to the tracker, so the source index of the frame
    # about to be read is START_FRAME + camera_tick. Breaking here rather than in the camera
    # branch stops as soon as the last wanted frame has been tracked.
    if END_FRAME and START_FRAME + camera_tick >= END_FRAME:
        break

    if frame_metadata['type'] == 'imu':
        accel_data = frame_metadata['accel']
        gyro_data = frame_metadata['gyro']
        imu_measurement = cuvslam.ImuMeasurement()
        imu_measurement.timestamp_ns = int(timestamp)
        imu_measurement.linear_accelerations = np.asarray(accel_data)
        imu_measurement.angular_velocities = np.asarray(gyro_data)
        tracker.register_imu_measurement(0, imu_measurement)
        imu_count_since_last_camera += 1
        continue

    images = [load_frame(image_path) for image_path in frame_metadata['images_paths']]

    # Blackout filter — flatten the stereo pair to BLACKOUT_FILL when this camera tick
    # falls in the blackout window of the oscillator.
    is_blackout = (
        camera_tick >= BLACKOUT_PERIOD
        and (camera_tick % BLACKOUT_PERIOD) < BLACKOUT_DURATION
    )
    if is_blackout:
        images = [np.full_like(image, BLACKOUT_FILL) for image in images]
        blackout_frames += 1
    camera_tick += 1

    # Check IMU measurements before tracking
    if (cfg.odometry_mode == cuvslam.Odometry.OdometryMode.Inertial
            and last_camera_timestamp is not None):
        if imu_count_since_last_camera == 0:
            print(
                f"Warning: No IMU measurements between timestamps "
                f"{last_camera_timestamp} and {timestamp}"
            )

    # Reset counters
    last_camera_timestamp = timestamp
    imu_count_before_track = imu_count_since_last_camera
    imu_count_since_last_camera = 0

    # Banner so the library's [MESSAGE] lines can be attributed to a frame: every trace line
    # printed after this one belongs to this track() call. flush=True is required because the
    # library writes with C printf on the same fd while Python buffers separately; see the note
    # on stdbuf in the VERBOSITY comment above.
    if VERBOSITY:
        print(
            f"=== frame {camera_tick - 1} (source {START_FRAME + camera_tick - 1}) "
            f"t={int(timestamp)} imu={imu_count_before_track}"
            f"{' BLACKOUT' if is_blackout else ''} ===",
            flush=True
        )

    # Track frame
    odom_pose_estimate, _ = tracker.track(timestamp, images)

    if odom_pose_estimate.world_from_rig is None:
        if is_blackout:
            failures_on_blackout += 1
        else:
            failures_elsewhere += 1
        print(f"Warning: Failed to track frame {frame_id}{' (blackout)' if is_blackout else ''}")
        # Record the failure so gaps are visible in the CSV. There is no pose on this frame,
        # and n_obs is whatever the last successful frame left behind, so it is not recorded.
        records.append({
            'camera_frame': camera_tick - 1,
            'frame_id': frame_id,
            'timestamp_ns': int(timestamp),
            'tracked': 0,
            'is_blackout': int(is_blackout),
        })
        continue

    # Get current pose and observations for the main camera and gravity in rig frame
    odom_pose = odom_pose_estimate.world_from_rig.pose
    current_observations_main_cam = tracker.odometry.get_last_observations(0)
    trajectory.append(odom_pose.translation)
    odom_trajectory.append([timestamp] + list(odom_pose.translation) + list(odom_pose.rotation))
    if is_blackout:
        blackout_positions.append(odom_pose.translation)

    # Ground truth for this frame, expressed in the odometry frame
    gt_position = None
    if gt_timestamps is not None:
        world_from_body = groundtruth_at(gt_timestamps, gt_poses, int(timestamp))
        if world_from_body is not None:
            world_from_cam0 = world_from_body @ body_from_cam0
            if odom_from_gt_world is None:
                # Align at the first frame where both odometry and GT exist
                odom_from_gt_world = pose_to_matrix(odom_pose) @ np.linalg.inv(world_from_cam0)
            gt_position = (odom_from_gt_world @ world_from_cam0)[:3, 3]
            gt_trajectory.append(gt_position)
            last_odom_with_gt = np.asarray(odom_pose.translation)
            frames_with_gt += 1

    gravity = None
    if cfg.odometry_mode == cuvslam.Odometry.OdometryMode.Inertial:
        # Gravity estimation requires collecting sufficient number of keyframes with motion diversity
        gravity = tracker.odometry.get_last_gravity()

    record = {
        'camera_frame': camera_tick - 1,
        'frame_id': frame_id,
        'timestamp_ns': int(timestamp),
        'tracked': 1,
        'is_blackout': int(is_blackout),
        'n_obs': len(current_observations_main_cam),
        # Landmarks backing the current frame. This drops to zero when the odometry map is
        # cleared, so a reset shows up as a trough rather than having to be inferred.
        'n_landmarks': len(tracker.odometry.get_last_landmarks()),
        'has_gravity': int(gravity is not None),
    }
    for key, value in zip(('tx', 'ty', 'tz'), odom_pose.translation):
        record[key] = float(value)
    for key, value in zip(('qx', 'qy', 'qz', 'qw'), odom_pose.rotation):
        record[key] = float(value)
    if gt_position is not None:
        for key, value in zip(('gt_x', 'gt_y', 'gt_z'), gt_position):
            record[key] = float(value)
    if gravity is not None:
        # Gravity is reported in the rig frame; the quaternion above rotates it to world.
        for key, value in zip(('grav_x', 'grav_y', 'grav_z'), gravity):
            record[key] = float(value)
    records.append(record)

    # Visualize
    rr.set_time("frame", sequence=frame_id)
    rr.log("world/trajectory", rr.LineStrips3D(trajectory, colors=[ODOM_COLOR]), static=True)
    if len(gt_trajectory) > 1:
        rr.log(
            "world/trajectory_gt",
            rr.LineStrips3D(gt_trajectory, colors=[GT_COLOR]),
            static=True
        )
    if blackout_positions:
        rr.log(
            "world/blackout_poses",
            rr.Points3D(blackout_positions, colors=[BLACKOUT_COLOR], radii=0.01),
            static=True
        )
    rr.log(
        "world/camera_0",
        rr.Transform3D(
            translation=odom_pose.translation,
            quaternion=odom_pose.rotation
        ),
        rr.Arrows3D(
            vectors=np.eye(3) * 0.2,
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]  # RGB for XYZ axes
        )
    )

    points = np.array([[obs.u, obs.v] for obs in current_observations_main_cam])
    colors = np.array([color_from_id(obs.id) for obs in current_observations_main_cam])
    rr.log(
        "world/camera_0/observations",
        rr.Points2D(positions=points, colors=colors, radii=5.0),
        rr.Image(images[0]).compress(jpeg_quality=80)
    )

    rr.log(
        "world/camera_1/observations",
        rr.Points2D(positions=points, colors=colors, radii=5.0),
        rr.Image(images[1]).compress(jpeg_quality=80)
    )

    if gravity is not None:
        rr.log(
            "world/camera_0/gravity",
            rr.Arrows3D(vectors=gravity, colors=[[255, 0, 0]], radii=0.015)
        )

    if cfg.odometry_mode == cuvslam.Odometry.OdometryMode.Inertial:
        rr.log("world/imu/accel/x", rr.Scalars(accel_data[0]), static=False)
        rr.log("world/imu/accel/y", rr.Scalars(accel_data[1]), static=False)
        rr.log("world/imu/accel/z", rr.Scalars(accel_data[2]), static=False)

        rr.log("world/imu/gyro/x", rr.Scalars(gyro_data[0]), static=False)
        rr.log("world/imu/gyro/y", rr.Scalars(gyro_data[1]), static=False)
        rr.log("world/imu/gyro/z", rr.Scalars(gyro_data[2]), static=False)

    frame_id += 1

if START_FRAME:
    print(f"Skipped the first {source_camera_tick} camera frames and their IMU samples")
if END_FRAME:
    print(f"Stopped at source camera frame {END_FRAME} (exclusive)")
print(f"Tracked {frame_id} frames, {frames_with_gt} of them covered by ground truth")
print(
    f"Blackout frames: {blackout_frames} "
    f"(period {BLACKOUT_PERIOD}, duration {BLACKOUT_DURATION}, fill {BLACKOUT_FILL}); "
    f"track failures on blackout frames: {failures_on_blackout}, elsewhere: {failures_elsewhere}"
)
if frames_with_gt > 0:
    # Compare at the last frame covered by GT, not the last tracked frame:
    # EuRoC GT stops about a second before the images do.
    final_offset = np.linalg.norm(last_odom_with_gt - gt_trajectory[-1])
    print(f"Final odometry-vs-GT position difference: {final_offset:.3f} m")

if RUN_DIR is not None and records:
    os.makedirs(RUN_DIR, exist_ok=True)
    start_suffix = f"_s{START_FRAME}" if START_FRAME else ""
    end_suffix = f"_e{END_FRAME}" if END_FRAME else ""
    csv_path = os.path.join(
        RUN_DIR,
        f"euroc_inertial_p{BLACKOUT_PERIOD}_d{BLACKOUT_DURATION}_f{BLACKOUT_FILL}"
        f"{start_suffix}{end_suffix}{RUN_TAG}.csv"
    )
    columns = [
        'camera_frame', 'frame_id', 'timestamp_ns', 'tracked', 'is_blackout', 'n_obs',
        'n_landmarks', 'has_gravity', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw',
        'gt_x', 'gt_y', 'gt_z', 'grav_x', 'grav_y', 'grav_z',
    ]
    with open(csv_path, 'w') as stream:
        stream.write(f"# cuvslam_version={cuvslam.get_version()[0]}\n")
        stream.write(f"# odometry_mode={cfg.odometry_mode}\n")
        stream.write(
            f"# blackout_period={BLACKOUT_PERIOD} blackout_duration={BLACKOUT_DURATION} "
            f"blackout_fill={BLACKOUT_FILL}\n"
        )
        # camera_frame below counts from the first tracked frame; add start_frame for the
        # index into the original sequence.
        stream.write(f"# start_frame={START_FRAME} end_frame={END_FRAME}\n")
        stream.write(','.join(columns) + '\n')
        for record in records:
            stream.write(','.join(
                '' if record.get(column) is None else str(record.get(column, ''))
                for column in columns
            ) + '\n')
    print(f"Wrote {len(records)} per-frame records to {csv_path}")
