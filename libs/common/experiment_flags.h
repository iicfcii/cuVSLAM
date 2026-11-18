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

#pragma once

#include <cstdlib>
#include <cstring>

namespace cuvslam {

// Boolean switch read from the process environment, for A/B experiments that must not need a
// rebuild per variant. Unset or empty -> `default_value`; "0", "false", "off" or "no" -> false;
// anything else -> true. Callers read it once (function-local static or constructor) and keep the
// result, so the environment is not consulted on a hot path.
inline bool ExperimentFlag(const char* name, bool default_value) {
  const char* v = std::getenv(name);
  if (v == nullptr || *v == '\0') {
    return default_value;
  }
  return !(std::strcmp(v, "0") == 0 || std::strcmp(v, "false") == 0 || std::strcmp(v, "off") == 0 ||
           std::strcmp(v, "no") == 0);
}

}  // namespace cuvslam
