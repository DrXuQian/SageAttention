// Host execution only. Real actlize traits anchor the integer register map.
#include <array>
#include <cstdio>
#include <cstdint>
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

int main() {
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
    for (int lane = 0; lane < 32; ++lane)
      for (int word = 0; word < 4; ++word) {
        uint32_t got = 0, want = 0;
        for (int byte = 0; byte < 4; ++byte) {
          uint32_t const shuffled = source[word & 1][(lane & ~3) + byte][word / 2];
          got |= ((shuffled >> (8 * (lane & 3))) & 255u) << (8 * byte);
          int const tag = a_linear(lane, 4 * word + byte);
          want |= uint32_t((tag * 13 + salt) & 255) << (8 * byte);
        }
        byte_bad += got != want;
        bit_negative += (got ^ 1u) != want;
      }
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
  if (map_bad || byte_bad || v_bad || lane_negative != 512 || half_negative != 256 ||
      bit_negative != 32768 || !v_negative || signed_negative != 128) return 1;
  std::puts("[all-int8 layout] PASS: real traits + packed words + V transpose/tail + five negatives");
}
