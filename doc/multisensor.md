# Multisensor odometry (C++)

Multisensor tracking is experimental: tracking may be inaccurate or fail for some sensor configurations and scenes.
It requires a build with `USE_CUNLS=ON` and currently supports pinhole cameras only.

## Valid rigs

A rig must contain:

- at least one RGB-D camera, configured in `MultisensorSettings::depth_camera_ids`; **or**
- at least one camera pair with overlapping frustums.

It may also contain additional RGB or RGB-D cameras and zero or one IMU. A single RGB-D camera with an IMU is valid.
Depth must be pixel-aligned with the RGB image of the same camera.

## Minimal RGB-D configuration

```cpp
#include <cstdint>
#include <vector>

#include <cuvslam2.h>

cuvslam::Camera camera{};
camera.size = {640, 480};
camera.principal = {320.f, 240.f};
camera.focal = {320.f, 320.f};
camera.distortion.model = cuvslam::Distortion::Model::Pinhole;

cuvslam::Rig rig{{camera}, {}};

cuvslam::Odometry::Config config;
config.odometry_mode = cuvslam::Odometry::OdometryMode::Multisensor;
config.multisensor_settings.depth_camera_ids = {0};
config.multisensor_settings.depth_scale_factor = 1000.f;  // uint16 millimeters -> meters

cuvslam::Odometry odometry(rig, config);

std::vector<uint8_t> rgb(640 * 480);
std::vector<uint16_t> depth(640 * 480);
const int64_t timestamp_ns = 1'000'000;

cuvslam::Image rgb_image{};
rgb_image.pixels = rgb.data();
rgb_image.width = 640;
rgb_image.height = 480;
rgb_image.encoding = cuvslam::Image::Encoding::MONO;
rgb_image.data_type = cuvslam::Image::DataType::UINT8;
rgb_image.timestamp_ns = timestamp_ns;
rgb_image.camera_index = 0;

cuvslam::Image depth_image{};
depth_image.pixels = depth.data();
depth_image.width = 640;
depth_image.height = 480;
depth_image.encoding = cuvslam::Image::Encoding::MONO;
depth_image.data_type = cuvslam::Image::DataType::UINT16;
depth_image.timestamp_ns = timestamp_ns;
depth_image.camera_index = 0;

const cuvslam::PoseEstimate estimate = odometry.Track({rgb_image}, {}, {depth_image});
if (!estimate.world_from_rig.has_value()) {
  // Tracking has not initialized or was lost.
}
```

The C++ API accepts `UINT16` or `FLOAT32` depth. PyCuVSLAM currently accepts `uint16` depth only. One configured depth
stream may be omitted from an individual `Track()` call after a frame drop; any supplied depth image must use a camera
index listed in `depth_camera_ids`.

## Optional IMU

Add one `ImuCalibration` to `Rig::imus` before constructing `Odometry`. Multisensor mode enables IMU fusion
automatically. Submit every IMU sample between camera frames in non-decreasing timestamp order:

```cpp
odometry.Track(previous_images, {}, previous_depths);
odometry.RegisterImuMeasurement(0, imu_sample_0);
odometry.RegisterImuMeasurement(0, imu_sample_1);
const cuvslam::PoseEstimate estimate = odometry.Track(current_images, {}, current_depths);
```

Calls to `Track()` and `RegisterImuMeasurement()` must be externally serialized. For a complete Python dataset example,
see [`examples/multisensor/`](../examples/multisensor/README.md).

---

Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
