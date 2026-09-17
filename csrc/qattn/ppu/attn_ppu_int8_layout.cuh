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

// A four-lane 4x4 byte transpose. First exchange columns within each lane
// pair, then exchange the two lane pairs. All selectors are ordinary byte
// picks (<8), never sign replication. The host oracle anchors the result to
// the real MMA ALayout, including every input bit independently.
constexpr unsigned probability_pair_selector(int lane) {
  return (lane & 1) ? 0x3715u : 0x6240u;
}
constexpr unsigned probability_quad_selector(int lane) {
  return (lane & 2) ? 0x3276u : 0x5410u;
}

// V's quantizer writes each K64 block transposed as [D,64]. A non-transposed
// INT8 AIU/TSM read then presents V columns as the MMA B operand's N mode.
constexpr int packed_value_offset(int channel, int k) { return channel * 64 + k; }

}  // namespace sageattention::ppu::layout
