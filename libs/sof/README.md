# Sparse Optical Flow (SOF)

Feature selection and Lucas-Kanade (LK) tracking — the front end of cuVSLAM's
odometry. This README is for maintainers who need to change anything in this
library. If you only call SOF from above, you do not need to read this.

## Implementation invariants

The LK tracker here is not a textbook Lucas-Kanade. Every constant below is
load-bearing — they were tuned together across automotive, drone, warehouse,
indoor, and street datasets. Changing any one of them in isolation will improve
one ODD and break another.

| Invariant | Value / behavior |
|---|---|
| Patch size | 7 × 7 |
| Image pyramid | Integer (`image_pyramid_u8`) |
| Gradient pyramid | Integer (`gradient_pyramid`) |
| Per-level scoring | Normalized cross-correlation (NCC) on every level |
| Level fallback | If a level fails to converge, the tracker drops to the next level rather than killing the feature outright |
| Patch padding | Zero-padded at the borders |
| Patch normalization | Mean subtracted from the patch before matching |
| Convergence check | Custom — see `lk_tracker.cpp` |

The integer pyramid + integer gradient predate the float path: they were
introduced for the original Jetson Nano port and stayed because they hold up on
all current targets. Do not convert to float "for clarity" — you will get a
different tracker.

The NCC + per-level fallback + mean subtraction combination is the part most
often missing from third-party LK implementations. If you are integrating an
external LK kernel as an alternative backend, these are the features it must
replicate to match accuracy.

## What lives where

- `lk_tracker.{h,cpp}` — the LK iteration itself.
- `klt_tracker.{h,cpp}` — KLT wrapper used by the multi-camera path.
- `image_pyramid_u8.{h,cpp}`, `gradient_pyramid.{h,cpp}` — the integer pyramids.
- `gftt.{h,cpp}` — Shi-Tomasi (good features to track) selection.
- `selector_mono.{h,cpp}`, `selector_stereo.{h,cpp}` — pick which features to
  track this frame.
- `image_manager.{h,cpp}` — owns image buffers and abstracts over the
  allocator kind (CUDA device memory vs plain host C buffers). This is the
  point of extension when adding a new memory backend — see
  [DESIGN_CONCEPTS.md](../../DESIGN_CONCEPTS.md).
- `sof_mono_cpu.cpp` / `sof_mono_gpu.cpp` and
  `sof_multicamera_cpu.cpp` / `sof_multicamera_gpu.cpp` — CPU and GPU paths.

Both CPU and GPU paths are kept bit-equivalent on purpose: regressions are
diagnosed by toggling the backend and comparing.

## Touching the LK tuning

If you genuinely need to change a constant (patch size, level count,
convergence threshold, NCC threshold, gradient threshold, …):

1. Run `tools/cuvslam_app` reporter on the full dataset list both with and
   without the change. See
   [DEVELOPMENT.md — Accuracy regression workflow](../../DEVELOPMENT.md#accuracy-regression-workflow-reporter).
2. Compare the resulting PDFs page by page. A constant that improves one
   dataset by 0.5% but breaks another by 5% is a regression, not a win.
3. Apply the drift-interpretation rule: ≤2% drift = trust the number, ≥20%
   drift = the trajectory is broken and the number is meaningless.
