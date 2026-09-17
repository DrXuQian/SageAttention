// Executes the production finalization with an explicit four-peer exchange.
// The independent anchor is a long-double nonnegative recurrence, not the
// legacy FP32 result. Neither this nor the CPU tensor oracle runs a PPU kernel.
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#include "attn_ppu_denominator.cuh"

using Row = std::array<float, 4>;

static uint32_t bits(float x) {
  uint32_t result;
  std::memcpy(&result, &x, sizeof(result));
  return result;
}

struct PeerXor {
  Row const &row;
  int lane;
  bool wrong_peer = false;
  float operator()(float, int mask) const {
    int peer = lane ^ (wrong_peer && mask == 2 ? 1 : mask);
    return mask == 1 ? row[peer] : row[peer] + row[peer ^ 1];
  }
};

static Row finalize(Row const &state, bool wrong_peer = false) {
  Row result;
  for (int lane = 0; lane < 4; ++lane)
    result[lane] = sageattention::ppu::finalize_row_denominator(
        state[lane], PeerXor{state, lane, wrong_peer});
  return result;
}

static uint32_t random_word(uint32_t &s) {
  s ^= s << 13;
  s ^= s >> 17;
  s ^= s << 5;
  return s;
}

int main(int argc, char **argv) {
  std::string plant = argc > 1 ? argv[1] : "";
  if (!plant.empty() && plant != "omit-final" && plant != "wrong-peer" &&
      plant != "double-final" && plant != "demand-raw") return 2;
  constexpr double limit = 5e-4;  // Registered before the kernel change.
  constexpr int lengths[] = {0, 1, 2, 4, 16, 64, 256, 1153};
  uint64_t traces = 0, block_updates = 0, bad = 0, unequal = 0;
  double worst_deferred = 0, worst_eager = 0;
  for (int blocks : lengths) {
    for (int pattern = 0; pattern < 5; ++pattern) {
      for (int seed = 1; seed <= 64; ++seed) {
        uint32_t rng = seed * 0x9e3779b9u + pattern;
        Row state{};
        float eager = 0, maximum = -1e30f;
        long double exact = 0;
        for (int k = 0; k < blocks; ++k) {
          Row local;
          for (int lane = 0; lane < 4; ++lane) {
            // Legal nonnegative sums of <=16 probabilities. lane 0 includes
            // a P=1 maximum; other lanes can be very small or completely masked.
            local[lane] = float(random_word(rng) % 1025) / 64;
            if (pattern == 1 && lane > 0) local[lane] = std::ldexp(local[lane], -24);
          }
          local[0] = std::max(local[0], 1.0f);
          bool masked = pattern == 4 || (pattern == 3 && k % 7 == 0);
          float tile_max = pattern < 2 ? 0.0f : float(int(random_word(rng) % 129) - 64);
          if (masked) {
            local.fill(0);
            tile_max = -1e30f;
          }
          float next_max = std::max(maximum, tile_max);
          float rescale = std::exp2(maximum - next_max);
          float block_scale = std::exp2(tile_max - next_max);
          maximum = next_max;
          float tile_sum = (local[0] + local[1]) + (local[2] + local[3]);
          eager = std::fma(eager, rescale, tile_sum * block_scale);
          for (int lane = 0; lane < 4; ++lane)
            state[lane] = std::fma(state[lane], rescale, local[lane] * block_scale);
          long double sum = 0;
          for (float x : local) sum += x;
          exact = exact * rescale + sum * block_scale;
          ++block_updates;
        }
        Row result = plant == "omit-final" ? state : finalize(state, plant == "wrong-peer");
        if (plant == "double-final") result = finalize(result);
        for (float x : result) {
          double error = exact == 0 ? std::abs(x) : double(std::abs((x-exact)/exact));
          worst_deferred = std::max(worst_deferred, error);
          bad += !std::isfinite(x) || error > limit;
          bad += bits(x) != bits(result[0]);
        }
        double eager_error = exact == 0 ? std::abs(eager) : double(std::abs((eager-exact)/exact));
        worst_eager = std::max(worst_eager, eager_error);
        bad += !std::isfinite(eager) || eager_error > limit;
        unequal += bits(eager) != bits(result[0]);
        ++traces;
      }
    }
  }

  // Keep the exact known reassociation witness alive; do not turn its
  // expected difference into a 'fix' that quietly restores the eager loop.
  Row state{}, local{16, 0x1p-20f, 0, 0};
  float eager = 0;
  for (int k = 0; k < 1153; ++k) {
    eager += (local[0] + local[1]) + (local[2] + local[3]);
    for (int lane = 0; lane < 4; ++lane) state[lane] += local[lane];
  }
  float deferred = finalize(state)[0];
  bad += bits(eager) != 0x46902000u || bits(deferred) != 0x46902001u;
  if (plant == "demand-raw") bad += bits(eager) != bits(deferred);
  bad += traces != 2560 || block_updates != 478720;
  std::printf("[deferred denominator host] traces=%llu/2560 updates=%llu/478720 "
              "bound=%.8g worst_deferred=%.9g worst_eager=%.9g raw_different=%llu "
              "rounding_witness=0x%08x/0x%08x bad=%llu plant=%s %s\n",
              (unsigned long long)traces, (unsigned long long)block_updates,
              limit, worst_deferred, worst_eager, (unsigned long long)unequal,
              bits(eager), bits(deferred), (unsigned long long)bad,
              plant.empty() ? "none" : plant.c_str(), bad ? "FAIL" : "PASS");
  return bad ? 1 : 0;
}
