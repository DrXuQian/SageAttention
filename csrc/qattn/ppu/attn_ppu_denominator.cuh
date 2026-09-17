/* Copyright (c) 2026, SageAttention PPU contributors. */
#pragma once

#include <cute/config.hpp>

namespace sageattention::ppu {

// PPU CLayout row peers differ only in lane%4. Keep the two-stage sum in a
// host/device seam so tests execute this exact finalization, not a copy.
// Integer PV calls it once after K. Each lane's running maximum/rescales must
// be identical across its four peers; the denominator itself stays local.
template <class PeerXor>
CUTE_HOST_DEVICE float finalize_row_denominator(float sum, PeerXor peer_xor) {
  sum += peer_xor(sum, 1);
  sum += peer_xor(sum, 2);
  return sum;
}

}  // namespace sageattention::ppu
