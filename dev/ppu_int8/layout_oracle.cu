// Host-only proof for the physical seams of the SageAttention PPU path.
// It uses actlize's real PPU0010 MMA_Traits; no device code is executed.
#include <array>
#include <cstdio>
#include <type_traits>

#include <cute/atom/mma_traits_ppu0010.hpp>

#include "../../csrc/qattn/ppu/attn_ppu_layout.cuh"

using S8Traits = cute::MMA_Traits<
    cute::PPU0010_16x16x32_S32S8S8S32_TN>;
using F16Traits = cute::MMA_Traits<
    cute::PPU0010_16x16x16_F32F16F16F32_TN>;
using ALayout = typename F16Traits::ALayout;
using BLayout = typename F16Traits::BLayout;
using CLayout = typename S8Traits::CLayout;

static_assert(std::is_same_v<typename S8Traits::CLayout,
                             typename F16Traits::CLayout>);
static_assert(std::is_same_v<ALayout, BLayout>);

static int a_linear(int lane, int value) {
  return int(ALayout{}(cute::make_coord(
      cute::make_coord(lane % 4, lane / 4),
      cute::make_coord(value % 2, (value / 2) % 2, value / 4))));
}

static int c_linear(int lane, int value) {
  return int(CLayout{}(cute::make_coord(
      cute::make_coord(lane % 4, lane / 4),
      cute::make_coord(value % 4, value / 4))));
}

using Matrix = std::array<std::array<int, 16>, 16>;
using Physical = std::array<std::array<int, 8>, 32>;

static Matrix score_matrix() {
  Matrix result{};
  for (int m = 0; m < 16; ++m)
    for (int n = 0; n < 16; ++n) result[m][n] = 1 + 16 * m + n;
  return result;
}

static Physical score_physical(Matrix const &score) {
  Physical result{};
  for (int lane = 0; lane < 32; ++lane)
    for (int value = 0; value < 8; ++value) {
      int const linear = c_linear(lane, value);
      // CuTe's logical M mode is fastest: linear = m + 16*n.
      result[lane][value] = score[linear % 16][linear / 16];
    }
  return result;
}

static Physical bridge_physical() {
  Physical result{};
  for (int lane = 0; lane < 32; ++lane)
    for (int word = 0; word < 4; ++word) {
      uint32_t bits = sageattention::ppu::layout::pv_bridge_b_word(lane, word);
#ifdef SAGE_PPU_BREAK_BRIDGE_BIT
      if (lane == 0 && word == 0) bits ^= 0x00003c00u;
#endif
      result[lane][2 * word] = (bits & 0xffffu) == 0x3c00u;
      result[lane][2 * word + 1] = (bits >> 16) == 0x3c00u;
    }
  return result;
}

static Matrix physical_as_a(Physical const &physical) {
  Matrix result{};
  for (int lane = 0; lane < 32; ++lane)
    for (int value = 0; value < 8; ++value) {
      int const linear = a_linear(lane, value);
      result[linear % 16][linear / 16] = physical[lane][value];
    }
  return result;
}

static Matrix physical_as_b(Physical const &physical) {
  Matrix result{};
  for (int lane = 0; lane < 32; ++lane)
    for (int value = 0; value < 8; ++value) {
      int const linear = a_linear(lane, value);
      // PPU0010's TN B trait exposes (K,N), with K as the fastest mode.
      result[linear % 16][linear / 16] = physical[lane][value];
    }
  return result;
}

static Matrix multiply(Matrix const &a, Matrix const &b) {
  Matrix d{};
  for (int m = 0; m < 16; ++m)
    for (int n = 0; n < 16; ++n)
      for (int k = 0; k < 16; ++k) d[m][n] += a[m][k] * b[k][n];
  return d;
}

static int downstream_bad(Matrix const &d, Matrix const &score) {
  int bad = 0;
  for (int lane = 0; lane < 32; ++lane)
    for (int value = 0; value < 8; ++value) {
      int const c = c_linear(lane, value);
      int const a = a_linear(lane, value);
      bad += d[c % 16][c / 16] != score[a % 16][a / 16];
    }
  return bad;
}

static int row_formula_bad() {
  int bad = 0;
  for (int lane = 0; lane < 32; ++lane)
    for (int value = 0; value < 8; ++value) {
      int const linear = c_linear(lane, value);
      bad += sageattention::ppu::layout::accumulator_row(lane, value)
          != linear % 16;
      bad += sageattention::ppu::layout::accumulator_column(lane, value)
          != linear / 16;
    }
  return bad;
}

static int transposed_formula_bad() {
  int bad = 0;
  for (int lane = 0; lane < 32; ++lane)
    for (int value = 0; value < 8; ++value) {
      // This was the first host-model bug: it treats CuTe's linear result as
      // row-major and therefore swaps the two logical modes.
      int const old_row = (lane & 3) + 4 * (value & 3);
      int const old_column = lane / 4 + 8 * (value >> 2);
      int const linear = c_linear(lane, value);
      bad += old_row != linear % 16 || old_column != linear / 16;
    }
  return bad;
}

static int row_peer_bad() {
  int bad = 0;
  for (int lane_div4 = 0; lane_div4 < 8; ++lane_div4)
    for (int row_half = 0; row_half < 2; ++row_half) {
      bool seen[16] = {};
      int row = -1;
      for (int lane_mod4 = 0; lane_mod4 < 4; ++lane_mod4)
        for (int column_slot = 0; column_slot < 4; ++column_slot) {
          int const lane = 4 * lane_div4 + lane_mod4;
          int const value = 4 * row_half + column_slot;
          int const linear = c_linear(lane, value);
          if (row < 0) row = linear % 16;
          bad += row != linear % 16;
          int const column = linear / 16;
          bad += seen[column];
          seen[column] = true;
        }
      for (bool hit : seen) bad += !hit;
    }
  return bad;
}

int main() {
  Matrix const score = score_matrix();
  Physical const score_p = score_physical(score);
  Physical const bridge_p = bridge_physical();

  // Production bridge: score is A and the trait-derived permutation matrix is
  // B.  The output C registers are immediately consumed as the next MMA's A.
  int const bridge_bad = downstream_bad(
      multiply(physical_as_a(score_p), physical_as_b(bridge_p)),
      score);
  // Negative: reversing score and constant operands is not an equivalent MMA.
  int const old_order_bad = downstream_bad(
      multiply(physical_as_a(bridge_p), physical_as_b(score_p)),
      score);
  int const formula_bad = row_formula_bad();
  int const old_formula_bad = transposed_formula_bad();
  int const peers_bad = row_peer_bad();

  std::printf(
      "[PPU Sage layout] c_formula_bad=%d/512 transposed_formula_bad=%d/256 "
      "row_peer_bad=%d bridge_bad=%d/256 old_order_bad=%d/256\n",
      formula_bad, old_formula_bad, peers_bad, bridge_bad, old_order_bad);

#ifdef SAGE_PPU_BREAK_BRIDGE_BIT
  if (bridge_bad == 0) {
    std::fprintf(stderr, "bridge-bit negative did not turn red\n");
    return 2;
  }
  std::fprintf(stderr, "[PPU Sage layout negative] bridge-bit EXPECTED-RED bad=%d/256\n",
               bridge_bad);
  return 1;
#else
  if (formula_bad != 0 || old_formula_bad == 0 || peers_bad != 0 ||
      bridge_bad != 0 || old_order_bad != 240) return 1;
  std::printf("[PPU Sage layout] PASS: real traits + row peers + C-to-A bridge\n");
  return 0;
#endif
}
