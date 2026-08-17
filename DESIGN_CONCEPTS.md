# cuVSLAM Design Concepts

This document captures architectural decisions and design principles for cuVSLAM.
Follow these when making code changes or designing new features.

---

## 1. Per-frame internal overrides are stateless — no setters

**Rule:** A low-level parameter that must vary for a single frame is passed explicitly to
`Odometry::Track()` through `cuvslam::internal::Internals`. Do not mutate a long-lived
object through a setter to change one frame.

`Internals` is an unstable expert/development interface declared in
`libs/cuvslam/cuvslam2_internal.h`. It is not part of the stable user-facing API. Normal
applications should omit it and use the built-in defaults.

**Why:** Setters create implicit shared state between frames. A parameter change made
during one `Track()` call can silently bleed into the next frame if the setter mutates an
object that is reused across calls. This makes behavior hard to reason about, test, and
reproduce.

**How it works in cuVSLAM:**

`Odometry::Track()` accepts an optional pointer to `Internals`. All fields have concrete
defaults. `BuildTrackFrameSettings()` converts the selected values to
`odom::TrackPerFrameSettings`, which is threaded down the call stack without modifying
stored settings.

```cpp
// Expert/development use: override feature count for one frame only.
cuvslam::internal::Internals internals;
internals.num_desired_tracks = 200;
odometry.Track(images, {}, {}, &internals);

// The next call uses built-in defaults.
odometry.Track(images);
```

Construction-time settings stored by `Odometry::Impl` are not changed by the per-frame
override.

**What to avoid:**

```cpp
// BAD — setter mutates shared state and can bleed across frames.
odometry.SetNumDesiredTracks(200);
odometry.Track(images);
```

**Where to add a new per-frame internal parameter:**

1. Confirm that the parameter is for expert/development tuning rather than a normal user
   feature. Stable user-facing behavior belongs in `Odometry::Config` or another public API.
2. Add the field and its default to `cuvslam::internal::Internals` in
   `libs/cuvslam/cuvslam2_internal.h`.
3. Map it into `odom::TrackPerFrameSettings` in `BuildTrackFrameSettings()` in
   `libs/cuvslam/cuvslam2.cpp`.
4. Add it to the appropriate `TrackPerFrameSettings` sub-struct (`sof`, `kf`, `pnp`,
   `icp`, and so on), then thread it through the call chain without storing it.
5. Update the Python binding and YAML loader only when the parameter must be available to
   the corresponding development tools.

---

## 2. Resolve optional inputs at the API boundary

**Rule:** Optional input is appropriate at an API boundary where a caller may genuinely
omit a value. Internal APIs below that boundary should receive concrete settings whenever
possible.

**Why:** Optionals at every layer of the call stack force every internal function to check
`has_value()` before use. This is noise. Once the public API has resolved an optional to a
concrete value (using a default), the rest of the system should not need to know the value
was ever absent.

**How it works in cuVSLAM:**

`Odometry::Track()` accepts a nullable `Internals` pointer. A null pointer selects
`Internals{}`. `BuildTrackFrameSettings()` converts the result to a concrete
`TrackPerFrameSettings`; lower layers do not need to know whether the caller supplied an
override.

`Internals::kf_override_frame_selection` is an intentional tri-state exception: unset uses
automatic keyframe selection, `true` forces a keyframe, and `false` forces a non-keyframe.
Resolve such values at the first layer that has enough context, rather than propagating
optionality farther down the call stack.

```text
Internals* (null or expert/development overrides)
    └─► Internals{} when null
        └─► BuildTrackFrameSettings() produces TrackPerFrameSettings
            └─► IVisualOdometry::track(TrackPerFrameSettings&)   // no optional
                    └─► IMultiSOF::trackNextFrame(TrackPerFrameSettings&)  // no optional
                            └─► IMonoSOF::track(Settings&)  // no optional
```

**What to avoid:**

```cpp
// BAD — optional leaks into internal API
void trackNextFrame(..., std::optional<Settings> sof_settings = std::nullopt);

// BAD — internal function must check presence
if (sof_settings.has_value()) { ... }
```

**Corollary — no default arguments on internal functions:**

Internal functions should not have `= {}` default arguments. That is just a hidden optional.
Every call site should pass the struct explicitly, making the data flow visible in the code.

```cpp
// BAD — hides that data is being passed; caller can silently get wrong defaults
void track(const Settings& sof_settings = {});

// GOOD — caller always states what settings it is using
void track(const Settings& sof_settings);
```

---

## 3. Bundle related parameters into a struct rather than growing argument lists

**Rule:** When a group of parameters is always used together or represents a coherent
configuration unit, wrap them in a named struct. Do not add individual parameters to
function signatures.

**Why:** Long argument lists are fragile (easy to reorder), hard to extend, and obscure
what a function actually needs. A named struct documents intent, can be forwarded as a
single argument through multiple layers, and makes adding new fields backwards-compatible
at the struct level.

**How it works in cuVSLAM:**

- `sof::Settings` — all feature tracking parameters.
- `odom::KeyFrameSettings` — keyframe selection thresholds.
- `odom::TrackPerFrameSettings` — bundles the above two for passing through the VO layer.
- `sba::Settings`, `pnp::PNPSettings`, etc. — each subsystem owns its config struct.

When a new per-frame parameter category is needed (e.g. ICP overrides), add a new
sub-struct to `TrackPerFrameSettings` rather than adding individual fields or new function
parameters:

```cpp
struct TrackPerFrameSettings {
  sof::Settings sof;
  KeyFrameSettings kf;
  // Add new categories here, not as additional function parameters
};
```

---

## 4. Construction-time config vs internal runtime tuning

Settings fall into three categories:

| Category | Example | Where it lives | Stability |
|---|---|---|---|
| **Construction-time configuration** | GPU on/off, odometry mode, data export | `Odometry::Config`, passed to the constructor | Public API |
| **Per-frame internal tuning** | Feature count, border sizes, keyframe threshold | `internal::Internals`, passed to `Track()`, never stored | Unstable expert/development API |
| **Persistent internal tuning** | SBA window and solver parameters | `internal::InternalParameter`, passed to `ApplyPersistentInternalParameters()` | Internal use only |

If a normal user must choose a value at startup, it belongs in `Config`. If a low-level
development tool must vary a solver value per frame, it may belong in `Internals`. Values
that intentionally persist after tracker construction use
`ApplyPersistentInternalParameters()`.

Do not expose a user-facing feature through `Internals` merely because it is convenient.
Design a stable public API for that feature instead.

---

## 5. Convenience classes compose the public API without widening it

**Rule:** A convenience class built on top of other public classes uses only their public API,
exposes the components it owns instead of mirroring every one of their methods, and never moves
policy — tuning constants, per-language defaults — into the public header.

**Why:** A facade that mirrors everything has to grow every time the classes underneath it grow,
and each mirrored method is somewhere the two can disagree. Accessors cost one call and stay
correct forever. Choosing a value on the caller's behalf is harder to undo: `EnableReadingData`
takes `max_items_count` because only the caller knows how many items it needs, and a number picked
for them in `cuvslam2.h` becomes observable behavior we cannot change without changing the API.

**How it works in cuVSLAM:**

`cuvslam::Tracker` owns an `Odometry` and an optional `Slam`. It is implemented in
`libs/cuvslam/tracker.cpp`, which includes nothing but `cuvslam2.h`, so it cannot depend on
internals even by accident. It runs the per-frame sequence, and enables the exports SLAM needs on
its own copy of the caller's config rather than mutating the caller's object.

Reading SLAM data layers is deliberately *not* mirrored. The useful form of `ReadLandmarks()`
needs a preceding `EnableReadingData(layer, max_items_count)`, and a sensible `max_items_count` is
a caller's choice, not part of the API. So callers that want it go through `GetSlam()`, and the
Python binding — which does pick concrete limits for convenience — keeps those numbers in
`python/cuvslam2.cpp`.

The same split applies to defaults. PyCuVSLAM enables observation and landmark export by default
because convenience matters more there; C++ does not. Those defaults live in named constants in
the binding, shared between the `Odometry.Config` constructor and `Tracker`'s default config, so
the two cannot drift.

**What to avoid:**

```cpp
// BAD — a magic limit in the public header becomes an API contract
std::shared_ptr<const Slam::Landmarks> ReadLandmarks(Slam::DataLayer layer) {
  slam->EnableReadingData(layer, 100000);
  ...
}

// BAD — the caller's config is an input, not scratch space
Tracker::Tracker(const Rig& rig, Config& cfg) { cfg.odometry.enable_landmarks_export = true; }
```
