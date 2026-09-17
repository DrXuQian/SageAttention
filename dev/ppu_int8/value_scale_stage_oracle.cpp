// Execute the production staging helper. The anchor is a caller-populated
// [B,H,K64,D] tensor with unique float bit patterns, not a second address model.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include "../../csrc/qattn/ppu/attn_ppu_layout.cuh"
#include "../../csrc/qattn/ppu/attn_ppu_value_scale.cuh"

namespace map = sageattention::ppu::layout;

uint32_t bits(float value) {
  uint32_t result;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}
float tag(uint32_t index) {
  uint32_t const raw = 0x3f000000u + index;
  float value;
  std::memcpy(&value, &raw, sizeof(value));
  return value;
}

struct Counts {
  int64_t blocks = 0, publications = 0, consumers = 0;
  int64_t bad = 0, expected_publications = 0, expected_consumers = 0;
};

template <int D>
void check(Counts &count, std::string const &plant) {
  using Stage = sageattention::ppu::ValueScaleStage<D>;
  static_assert(Stage::Bytes == D * sizeof(float));
  // int64 before multiplication: no code/byte or 32-bit expert-pitch seam.
  static_assert(Stage::head_offset(1000000, 9999, 10000, 1153) ==
                (10000000000LL + 9999) * 1153 * D);
  constexpr int PayloadFloats = (128 + 64 + 64) * D / sizeof(float);
  constexpr uint32_t Poison = 0xdeadbeefu;
  for (int heads : {1, 2, 7}) {
    for (int blocks : {1, 2, 4, 1153}) {
      int constexpr Batch = 3;
      std::vector<float> input(Batch * heads * blocks * D);
      uint32_t next = 1;
      for (auto &value : input) value = tag(next++);
      int64_t logical = 0;
      for (int batch = 0; batch < Batch; ++batch) {
        for (int head = 0; head < heads; ++head) {
          int pitch = blocks - int(plant == "wrong-head-pitch" && blocks > 1);
          auto const *head_data = input.data() +
              Stage::head_offset(batch, head, heads, pitch);
          // Visit every full and tail block in the caller's order. A head's
          // expected values are materialized by the caller, independently.
          for (int block = 0; block < blocks; ++block) {
            std::vector<float> shared(PayloadFloats + D + 1);
            for (auto &value : shared) std::memcpy(&value, &Poison, sizeof(value));
            int const offset = PayloadFloats - int(plant == "overlap-payload");
            float *stage = shared.data() + offset;
            std::array<int, D> owners{};
            for (int warp = 0; warp < 4; ++warp) {
              for (int lane = 0; lane < 32; ++lane) {
                int const thread = warp * 32 + lane;
                if (plant == "missing-thread" && thread == D - 1) continue;
                Stage::publish(stage, head_data, block, thread);
                if (thread < D) {
                  ++owners[thread];
                  ++count.publications;
                }
              }
            }
            // Simulated publication barrier, then the real CLayout consumer.
            for (int channel = 0; channel < D; ++channel) {
              count.bad += owners[channel] != 1;
              count.bad += bits(stage[channel]) != bits(input[logical + channel]);
            }
            for (int warp = 0; warp < 4; ++warp) {
              for (int lane = 0; lane < 32; ++lane) {
                for (int d = 0; d < D / 16; ++d) {
                  for (int cs = 0; cs < 4; ++cs) {
                    int const channel = d * 16 + map::accumulator_column(lane, cs);
                    int const read_channel = channel ^ int(plant == "wrong-column");
                    count.bad += bits(stage[read_channel]) != bits(input[logical + channel]);
                    ++count.consumers;
                  }
                }
              }
            }
            for (int i = 0; i < PayloadFloats; ++i)
              count.bad += bits(shared[i]) != Poison;
            count.bad += bits(shared[PayloadFloats + D]) != Poison;
            logical += D;
            ++count.blocks;
            count.expected_publications += D;
            count.expected_consumers += 128 * D / 4;
          }
        }
      }
      count.bad += logical != int64_t(input.size());
    }
  }
}

int main(int argc, char **argv) {
  std::string const plant = argc == 2 ? argv[1] : "";
  if (argc > 2 || (!plant.empty() && plant != "missing-thread" &&
      plant != "wrong-head-pitch" && plant != "overlap-payload" &&
      plant != "wrong-column")) return 2;
  Counts count;
  check<64>(count, plant);
  check<128>(count, plant);
  // Independent cartesian denominator, not the loop's self-reported count.
  int64_t constexpr Blocks = 2 * 3 * (1 + 2 + 7) * (1 + 2 + 4 + 1153);
  int64_t constexpr Publications = (64 + 128) * (Blocks / 2);
  int64_t constexpr Consumers = 32 * Publications;
  bool const pass = !count.bad && count.blocks == Blocks &&
      count.publications == Publications && count.consumers == Consumers &&
      count.expected_publications == Publications && count.expected_consumers == Consumers;
  std::printf("[V-scale stage] blocks=%lld/%lld publications=%lld/%lld "
              "consumers=%lld/%lld bad=%lld extra_barriers=0 "
              "shared_added=256/512 bytes %s\n",
      static_cast<long long>(count.blocks), static_cast<long long>(Blocks),
      static_cast<long long>(count.publications), static_cast<long long>(Publications),
      static_cast<long long>(count.consumers), static_cast<long long>(Consumers),
      static_cast<long long>(count.bad), pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
