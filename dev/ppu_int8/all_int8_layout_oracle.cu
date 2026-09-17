// Host execution only. Real actlize traits anchor the integer register map.
#include <array>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <cute/atom/mma_traits_ppu0010.hpp>
#include "../../csrc/qattn/ppu/attn_ppu_int8_layout.cuh"

namespace map = sageattention::ppu::layout;
using Traits = cute::MMA_Traits<cute::PPU0010_16x16x32_S32U8S8S32_TN>;
using QK = cute::MMA_Traits<cute::PPU0010_16x16x32_S32S8S8S32_TN>;
static_assert(std::is_same_v<typename QK::CLayout, typename Traits::CLayout>);
static_assert(std::is_same_v<typename QK::BLayout, typename Traits::BLayout>);
static_assert(64 * 255 * 127 == 2072640 && 64 * 255 * 127 < (1 << 24));

int a_linear(int lane, int value) {
  return int(typename Traits::ALayout{}(cute::make_coord(
      cute::make_coord(lane % 4, lane / 4),
      cute::make_coord(value % 4, (value / 4) % 2, value / 8))));
}
int c_linear(int lane, int value) {
  return int(typename Traits::CLayout{}(cute::make_coord(
      cute::make_coord(lane % 4, lane / 4),
      cute::make_coord(value % 4, value / 4))));
}

// ISA byte-permutation semantics, independent of either layout derivation.
uint32_t byte_perm(uint32_t a, uint32_t b, unsigned selector) {
  uint64_t const source = uint64_t(a) | (uint64_t(b) << 32);
  uint32_t result = 0;
  for (int byte = 0; byte < 4; ++byte) {
    unsigned const index = (selector >> (4 * byte)) & 15u;
    if (index >= 8) return 0xdeadbeefu;  // Sign replication is not admitted.
    result |= uint32_t((source >> (8 * index)) & 255u) << (8 * byte);
  }
  return result;
}

std::array<uint32_t, 32> transpose_bytes(
    std::array<uint32_t, 32> const &source, bool wrong_selector = false) {
  std::array<uint32_t, 32> pair{}, result{};
  for (int lane = 0; lane < 32; ++lane)
    pair[lane] = byte_perm(source[lane], source[lane ^ 1],
                          map::probability_pair_selector(lane));
  for (int lane = 0; lane < 32; ++lane)
    result[lane] = byte_perm(pair[lane], pair[lane ^ 2],
        map::probability_quad_selector(lane) ^ unsigned(wrong_selector));
  return result;
}

int main(int argc, char **argv) {
  bool const wrong_selector = argc == 2 && std::strcmp(argv[1], "--wrong-selector") == 0;
  bool const missing_basis = argc == 2 && std::strcmp(argv[1], "--missing-basis") == 0;
  if (argc > 1 && !wrong_selector && !missing_basis) return 2;
  int map_bad = 0, lane_negative = 0, half_negative = 0;
  std::array<int, 512> owners{};
  for (int lane = 0; lane < 32; ++lane)
    for (int word = 0; word < 4; ++word)
      for (int byte = 0; byte < 4; ++byte) {
        int const actual = c_linear(map::probability_source_lane(lane, byte),
                                    map::probability_source_value(lane, word))
                         + 256 * map::probability_source_half(word);
        int const wanted = a_linear(lane, 4 * word + byte);
        map_bad += actual != wanted;
        ++owners.at(actual);
        lane_negative += (c_linear(map::probability_source_lane(lane, byte) ^ 1,
                                   map::probability_source_value(lane, word))
                          + 256 * map::probability_source_half(word)) != wanted;
        half_negative += (actual % 256) != wanted;
      }
  for (int count : owners) map_bad += count != 1;

  // Emulate exactly the packed-word/shuffle operation, not the coordinate map.
  int byte_bad = 0, bit_negative = 0;
  for (int salt = 0; salt < 256; ++salt) {
    uint32_t source[2][32][2]{};
    for (int half = 0; half < 2; ++half)
      for (int lane = 0; lane < 32; ++lane)
        for (int e = 0; e < 8; ++e) {
          int const tag = c_linear(lane, e) + 256 * half;
          source[half][lane][e / 4] |= uint32_t((tag * 13 + salt) & 255) << (8 * (e & 3));
        }
    for (int word = 0; word < 4; ++word) {
      std::array<uint32_t, 32> lane_words{};
      for (int lane = 0; lane < 32; ++lane)
        lane_words[lane] = source[word & 1][lane][word / 2];
      auto const transposed = transpose_bytes(lane_words, wrong_selector);
      for (int lane = 0; lane < 32; ++lane) {
        uint32_t const got = transposed[lane];
        uint32_t want = 0;
        for (int byte = 0; byte < 4; ++byte) {
          int const tag = a_linear(lane, 4 * word + byte);
          want |= uint32_t((tag * 13 + salt) & 255) << (8 * byte);
        }
        byte_bad += got != want;
        bit_negative += (got ^ 1u) != want;
      }
    }
  }

  // Complete bit basis, not random payloads: every bit of both score-C
  // fragments independently reaches exactly its real MMA-A coordinate.
  int basis_cases = 0, basis_bad = 0, basis_words = 0;
  for (int basis = 0; basis < 4096 - int(missing_basis); ++basis) {
    int const src_word = basis / 1024;
    int const src_lane = (basis / 32) % 32;
    int const src_bit = basis % 32;
    int const coord = c_linear(src_lane, 4 * (src_word / 2) + src_bit / 8)
                    + 256 * (src_word & 1);
    for (int word = 0; word < 4; ++word) {
      std::array<uint32_t, 32> source{};
      if (word == src_word) source[src_lane] = 1u << src_bit;
      auto const got = transpose_bytes(source, wrong_selector);
      for (int lane = 0; lane < 32; ++lane) {
        uint32_t want = 0;
        for (int byte = 0; byte < 4; ++byte)
          if (a_linear(lane, 4 * word + byte) == coord)
            want |= 1u << (8 * byte + src_bit % 8);
        basis_bad += got[lane] != want;
        ++basis_words;
      }
    }
    ++basis_cases;
  }

  // Packed V is anchored to the original [K,D] values, including masked tails.
  int v_bad = 0, v_negative = 0, v_cells = 0;
  for (int dim : {64, 128})
    for (int length : {1, 31, 32, 33, 63, 64}) {
      std::array<int, 8192> packed{};
      for (int k = 0; k < 64; ++k)
        for (int d = 0; d < dim; ++d)
          packed[map::packed_value_offset(d, k)] = k < length ? 1 + k * dim + d : 0;
      for (int d = 0; d < dim; ++d)
        for (int k = 0; k < 64; ++k) {
          int const want = k < length ? 1 + k * dim + d : 0;
          v_bad += packed[d * 64 + k] != want;
          v_negative += packed[k * dim + d] != want;
          ++v_cells;
        }
    }
  // Unsigned P>=128 is a semantic requirement, not merely a pointer spelling.
  int signed_negative = 0;
  for (int p = 0; p < 256; ++p)
    signed_negative += int(int8_t(uint8_t(p))) * 127 != p * 127;

  std::printf("[all-int8 layout] map_bad=%d/512 packed_word_bad=%d/32768 "
              "V_bad=%d/%d integer_partial_bound=2072640\n",
              map_bad, byte_bad, v_bad, v_cells);
  std::printf("[all-int8 negatives] lane=%d/512 missing_half=%d/512 bit=%d/32768 "
              "V_transpose=%d/%d signed_P=%d/256\n",
              lane_negative, half_negative, bit_negative, v_negative, v_cells, signed_negative);
  std::printf("[all-int8 byte butterfly] basis=%d/4096 words=%d/524288 bad=%d "
              "shuffle_stages=2\n", basis_cases, basis_words, basis_bad);
  if (map_bad || byte_bad || v_bad || lane_negative != 512 || half_negative != 256 ||
      bit_negative != 32768 || !v_negative || signed_negative != 128 ||
      basis_cases != 4096 || basis_words != 524288 || basis_bad) {
    std::puts("[all-int8 layout] FAIL: coordinate/value/coverage contract");
    return 1;
  }
  std::puts("[all-int8 layout] PASS: real traits + packed words + V transpose/tail + five negatives");
}
