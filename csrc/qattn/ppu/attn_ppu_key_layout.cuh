/* Copyright (c) 2026, SageAttention PPU contributors. */
#pragma once
#include <cute/config.hpp>

namespace sageattention::ppu::layout {

// Involution: exchange the two low two-bit row groups, keep the N16 origin.
// C's (lane%4 + 4*local_column) becomes A's (4*lane%4 + local_column).
CUTE_HOST_DEVICE constexpr int permute_key_row(int row) {
  return (row & ~15) | ((row & 3) << 2) | ((row >> 2) & 3);
}

// A partial K64 block retains the original format. Applying the permutation
// there would create holes or require reading/writing past the caller's N.
CUTE_HOST_DEVICE constexpr int key_storage_row(int row, int tokens) {
  return (row / 64 + 1) * 64 <= tokens ? permute_key_row(row) : row;
}

}  // namespace sageattention::ppu::layout
