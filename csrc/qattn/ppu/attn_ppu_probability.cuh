/* Copyright (c) 2026, SageAttention PPU contributors. */
#pragma once

#include <cute/config.hpp>

namespace sageattention::ppu {

// Original integer-PV rule: s <= -1e29 produces P=0, otherwise exp2(s-max).
// For finite s <= max, this row-level origin gives the same result:
//   max > threshold: a masked s is at least one enormous FP32 ULP below max,
//                    so exp2(s-max) is exactly zero;
//   max <= threshold: every s is masked and exp2(s-0) is exactly zero.
// In particular, do not leave origin=max for an all-masked row: that gives
// exp2(-1e30 - -1e30)=1. The host seam test retains that exact negative.
CUTE_HOST_DEVICE constexpr float probability_exponent_origin(float maximum) {
  return maximum > -1.0e29f ? maximum : 0.0f;
}

}  // namespace sageattention::ppu
