/* Copyright (c) 2026, SageAttention PPU contributors. */
#pragma once

#include <cstdint>
#include <cute/config.hpp>

namespace sageattention::ppu {

// Real SDK census: this address chain shrinks for D128/noncausal. The D64
// and causal bodies already hoist more addressing; staging grows their ALU.
// Keep those original paths until a separately measured candidate wins.
template <int HeadDim, bool Causal, bool Int8PV>
inline constexpr bool use_shared_value_scale = Int8PV && HeadDim == 128 && !Causal;

// The quantizer's authority is [batch, kv_head, K64_block, channel]. Scale
// values are floats, not quantized codes or swizzled AIU payloads. One CTA
// stages each channel once; the existing issue-V publication barrier makes
// them visible to all four warps. No quantization granularity changes here.
template <int HeadDim>
struct ValueScaleStage {
  static_assert(HeadDim == 64 || HeadDim == 128);
  static constexpr int Bytes = HeadDim * sizeof(float);

  CUTE_HOST_DEVICE static constexpr int64_t head_offset(
      int batch, int kv_head, int heads, int k64_blocks) {
    return (int64_t(batch) * heads + kv_head) * k64_blocks * HeadDim;
  }

  CUTE_HOST_DEVICE static void publish(
      float *stage, float const *head, int k64_block, int thread) {
    if (thread < HeadDim)
      stage[thread] = head[int64_t(k64_block) * HeadDim + thread];
  }
};

}  // namespace sageattention::ppu
