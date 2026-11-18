# lastfeet: replay a control-rs recording through cuVSLAM

`replay_rrd_compare.py` takes one lastfeet `.rrd` recording, feeds the inputs the live tracker saw (front_forward
stereo pair, the rear_backward pair, the 400 Hz IMU, calibration) to the **installed** cuVSLAM with the live
tracker's configuration, and compares the result with the trajectory the robot produced at the time.

It exists to answer, on the target hardware (boxer, Orin):

1. does a newer cuVSLAM build track the same drives as well as the build that was on the robot?
2. is `track()` fast enough, and does it ramp up over a drive (the v17.0.0 driving regression)?
3. are the two local IMU patches (see below) still needed on top of upstream `main`?

## The two local patches, as runtime switches

This branch carries two IMU changes that boxer and milo have been running. Both are environment variables, so one
build covers every variant:

| variable | default | meaning |
|---|---|---|
| `CUVSLAM_HOLD_GRAVITY` | `1` | `1`: `optimize_inertial` holds gravity at the LinearAlignment + RefineGravity estimate. `0`: stock upstream update. |
| `CUVSLAM_IMU_SBA_INIT` | `0` | `0`: skip the synchronous IMU SBA at gravity init. `1`: stock upstream (now on the GPU bundler after #190). |

The defaults reproduce the build deployed on the robots, so a plain build of this branch behaves as they do today.
`HOLD_GRAVITY=0 IMU_SBA_INIT=1` is stock upstream. With `--cuvslam-verbosity 2` (the script's default) the build
prints `IMU experiment flags: ...` once, and the label of every run records the values.

## Build and install (boxer)

Build in a separate conda env so the working `lastfeet` env stays as the fallback:

```bash
conda create -n lastfeet-v18 --clone lastfeet
conda activate lastfeet-v18
pip install -U 'rerun-sdk>=0.35' pyyaml matplotlib      # older rerun cannot read these recordings

cd ~/code/cuVSLAM && git switch fuchen/boxer-multisensor-v18
# cuNLS is required by Multisensor. TensorRT/AnyLoc (visual place recognition) is on by default on main;
# switch it off unless you want VPR, it needs a TensorRT engine build.
cmake -S . -B build-v18 -DCMAKE_BUILD_TYPE=Release -DUSE_CUNLS=ON -DUSE_TENSORRT=OFF
cmake --build build-v18 --parallel $(nproc) --target cuvslam
CUVSLAM_BUILD_DIR=$PWD/build-v18 pip install --no-cache-dir ./python
python -c "import cuvslam; print(cuvslam.get_version())"     # 18.0.0+<sha>
```

cuNLS ships no sm_87 cubin; its kernels JIT from PTX on first use (a few seconds, cached in `~/.nv/ComputeCache`),
so the first few `track()` calls of a fresh machine are slow. The script reports per-window means so that stands out.

## Run

```bash
cd ~/code/lastfeet-monorepo/logs/delivery-veho-rw-20261111
S=~/code/cuVSLAM/examples/lastfeet_rrd_replay/replay_rrd_compare.py

python $S real_nov-11-2026_12-58-41.rrd --out ~/replay_out                      # paced at the recorded rate
python $S real_nov-11-2026_12-58-41.rrd --out ~/replay_out --speed 0            # as fast as possible
python $S ... --max-seconds 30 --speed 0                                        # quick functional check

# the A/B matrix: each run writes its own .npz
CUVSLAM_HOLD_GRAVITY=0                         python $S <rrd> --out ~/replay_out
CUVSLAM_IMU_SBA_INIT=1                         python $S <rrd> --out ~/replay_out
CUVSLAM_HOLD_GRAVITY=0 CUVSLAM_IMU_SBA_INIT=1  python $S <rrd> --out ~/replay_out

# one figure / one rerun recording with all of them against the live run
python $S <rrd> --out ~/replay_out --no-replay --overlay ~/replay_out/*.npz
```

Use the raw `real_*.rrd`, not `*_policy_pinhole.rrd`. Other useful flags: `--aux left right` (try 4 pairs, live used
`rear_backward` only), `--no-slam`, `--no-imu`, `--sync` (offline modes: reproducible, not live-like),
`--no-rerun`, `--spawn`. `--help` lists the rest.

Needs: `cuvslam` (this branch), numpy, scipy, opencv, pyyaml, matplotlib, rerun-sdk >= 0.35, and a lastfeet monorepo
checkout (`--monorepo`, default `~/code/lastfeet-monorepo`) because it imports the live tracker's calibration helpers
and reads the IMU noise / correction quaternion from `configs/real_boxer.yaml` and the URDF from `robot_description/`.

## What it reproduces, and what it does not

Reproduced from `cuvslam_tracker/tracker.py`: Multisensor with the IMU in the rig and no depth, `async_sba=True`,
`use_gpu=True`, exports off, SLAM with `sync_mode=False` (`Tracker.Mode.OdometryWithSlamRealtime`), the rig anchored at
the URDF base, poses composed with the initial world pose, the same IMU feed order (all samples up to the frame time,
strictly increasing), and aux pairing on `host_sync_ts` within half the aux frame period.

Two sanity checks print at the start when the sibling `.log` exists: the initial world pose and `T_base_from_camA`
must match what the live tracker printed.

Not reproduced:

- The live tracker drops stale frames when it falls behind. The replay feeds every recorded frame. `lag_ms` shows how
  far behind the recorded timeline each frame was when `track()` started; growing lag is the live "backlog" symptom.
- The recordings have camera stalls of up to ~1 s, hitting all cameras at once (the IMU is unaffected). They are
  replayed as they happened.
- The segmenter, mapper and everything else that shared the GPU and CPU with the tracker on the robot are not running,
  so `track()` times here are a floor, not what the robot saw.
- The IMU noise values and correction quaternion come from the *current* `real_boxer.yaml`, not the one on the day.

## Outputs

For recording `X` and run label `L` (cuvslam version + the two flags), `--out` gets `X__L.{rrd,png,json,npz}`:

- `.rrd`: open with `rerun`. 3D view with the live SLAM path in white and each replay run in colour (solid = slam,
  dark = odometry, light = the pose graph after the final optimisation), loop closures, a moving pose frame per source,
  and time series for `track()` ms, lag, distance to the live pose and gravity tilt.
- `.png`: the same on one page (top-down path, `track()` time, divergence from live, gravity tilt).
- `.json`: numbers. `track_ms` mean/p50/p95/p99/max, `track_over_frame_period`, `track_ramp_last_over_first_third` (the
  regression signature: ~1.0 is flat, the v17.0.0 collapse was >30), `track_ms_per_10s_window_mean`, lag, gravity
  convergence tick and tilt, path length, end-to-start distance, loop-closure count, and position difference to the live
  slam pose per frame.
- `.npz`: the raw per-frame arrays, for `--overlay`.

## Reading the numbers

- The live and replay poses share a world frame, so `divergence_vs_live_slam_m` needs no alignment. It mixes drift
  with loop-closure jumps (either run can jump), so read it next to `end_to_start_m`.
- Every session in `delivery-veho-rw-20261111` ends where it started, so `end_to_start_m` is a cheap drift measure, but
  only if the robot really returned to the same spot.
- Gravity tilt is the angle of the estimated gravity from world vertical. A few degrees is mounting tolerance; it is
  blank until the IMU state machine converges, which needs some motion.
