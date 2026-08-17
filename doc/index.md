cuVSLAM: CUDA-Accelerated Visual Odometry and Mapping
=====================================================

This page provides documentation for cuVSLAM C++ API.

The API has three classes. `cuvslam::Odometry` tracks the rig pose from camera and IMU data, and
`cuvslam::Slam` builds a map from odometry results, closes loops, and saves and reloads maps.
`cuvslam::Tracker` combines the two behind one interface and is the recommended entry point: it runs
the per-frame sequence both classes require, returns the odometry and SLAM poses together, and keeps
its SLAM accessors safe to call when SLAM is disabled. Use `cuvslam::Odometry` and `cuvslam::Slam`
directly when you need full control over the two components; `cuvslam::Tracker::GetOdometry` and
`cuvslam::Tracker::GetSlam` reach them without giving up the tracker.

Tracking modes
--------------

The tracker supports several odometry modes — see `cuvslam::Odometry::OdometryMode` for the enum
and per-mode requirements. `Multisensor` accepts at least one RGB-D camera or one overlapping camera
pair, with an optional IMU, and requires a cuNLS-enabled build. It is configured through
`cuvslam::Odometry::MultisensorSettings`. A high-level mode chooser table lives in the top-level
README's "Tracking modes" section.

For saving maps and relocalizing with `LocalizeInMap`, see [load_map.md](load_map.md).

---

Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
