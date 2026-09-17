// Host-only production map + independent real MMA traits + complete bit basis.
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cute/atom/mma_traits_ppu0010.hpp>
#include "../../csrc/qattn/ppu/attn_ppu_key_layout.cuh"

namespace map = sageattention::ppu::layout;
using QK = cute::MMA_Traits<cute::PPU0010_16x16x32_S32S8S8S32_TN>;
using PV = cute::MMA_Traits<cute::PPU0010_16x16x32_S32U8S8S32_TN>;

int c_offset(int lane, int value) {
  return QK::CLayout{}(cute::make_coord(cute::make_coord(lane % 4, lane / 4),
                                       cute::make_coord(value % 4, value / 4)));
}
int a_offset(int lane, int value) {
  return PV::ALayout{}(cute::make_coord(cute::make_coord(lane % 4, lane / 4),
      cute::make_coord(value % 4, value / 4 % 2, value / 8)));
}

int main(int argc, char** argv) {
  const char* plant = argc == 2 ? argv[1] : "";
  auto is = [&](const char* x) { return std::strcmp(plant, x) == 0; };
  if (argc > 1 && !is("omit-permutation") && !is("wrong-bit") &&
      !is("permute-tail") && !is("physical-mask") && !is("missing-basis")) return 2;
  auto storage = [&](int row, int n) {
    if (is("omit-permutation")) return row;
    if (is("permute-tail")) return map::permute_key_row(row);
    int result = map::key_storage_row(row, n);
    return is("wrong-bit") ? result ^ 1 : result;
  };
  int roundtrip_bad = 0, extent_bad = 0, mask_bad = 0;
  int row_cases = 0, mask_cases = 0;
  for (int n = 1; n <= 192; ++n) {
    std::array<int, 192> seen{};
    for (int row = 0; row < n; ++row) {
      int const at = storage(row, n);
      ++row_cases;
      if (at < 0 || at >= n) { ++extent_bad; continue; }
      ++seen[at];
      roundtrip_bad += storage(at, n) != row;
      // Tail must be unchanged, not merely a roundtripping permutation.
      extent_bad += (row / 64 == n / 64 && n % 64 && at != row);
      for (int q = 0; q < n; ++q) {
        int const semantic = is("physical-mask") ? at : map::key_storage_row(at, n);
        mask_bad += (semantic <= q) != (row <= q);
        ++mask_cases;
      }
    }
    for (int row = 0; row < n; ++row) roundtrip_bad += seen[row] != 1;
  }

  // Both physical C fragments get tags in ORIGINAL logical [M,K] space.
  // The direct consumer interleaves their packed row words, with no shuffle.
  int basis_bad = 0, basis_words = 0, basis_cases = 0;
  for (int bit = 0; bit < 4096 - int(is("missing-basis")); ++bit) {
    int const wanted_coord = bit / 8, wanted_bit = bit % 8;
    for (int lane = 0; lane < 32; ++lane) for (int word = 0; word < 4; ++word) {
      uint32_t got = 0, want = 0;
      for (int byte = 0; byte < 4; ++byte) {
        int const c = c_offset(lane, 4 * (word / 2) + byte);
        int const original_key = storage(c / 16 + 16 * (word & 1), 64);
        int const actual_coord = c % 16 + 16 * original_key;
        if (actual_coord == wanted_coord) got |= 1u << (8 * byte + wanted_bit);
        if (a_offset(lane, 4 * word + byte) == wanted_coord)
          want |= 1u << (8 * byte + wanted_bit);
      }
      basis_bad += got != want;
      ++basis_words;
    }
    ++basis_cases;
  }
  std::printf("[K permutation] rows=%d roundtrip_bad=%d extent_bad=%d "
              "causal_comparisons=%d mask_bad=%d bit_basis=%d/4096 "
              "words=%d/524288 basis_bad=%d\n", row_cases, roundtrip_bad,
              extent_bad, mask_cases, mask_bad, basis_cases, basis_words, basis_bad);
  if (roundtrip_bad || extent_bad || mask_bad || basis_bad || basis_cases != 4096 ||
      basis_words != 524288 || row_cases != 18528 || mask_cases != 2377760) {
    std::puts("[K permutation] FAIL: layout/extent/mask/coverage");
    return 1;
  }
  std::puts("[K permutation] PASS: real QK-C/PV-A, involution, all K64 tails and causal order");
}
