/* Copyright (c) 2026, SageAttention PPU contributors. */
#pragma once

namespace sageattention::ppu::layout {

// Two 16x16 C fragments -> one 16x32 U8 A fragment. The real PPU0010
// MMA_Traits oracle exhausts this map independently; no floating MMA bridge.
constexpr int probability_source_lane(int lane, int byte) {
  return (lane & ~3) + byte;
}
constexpr int probability_source_value(int lane, int word) {
  return 4 * (word / 2) + (lane & 3);
}
constexpr int probability_source_half(int word) { return word & 1; }

// V's quantizer writes each K64 block transposed as [D,64]. A non-transposed
// INT8 AIU/TSM read then presents V columns as the MMA B operand's N mode.
constexpr int packed_value_offset(int channel, int k) { return channel * 64 + k; }

}  // namespace sageattention::ppu::layout
