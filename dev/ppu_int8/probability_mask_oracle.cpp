// The production origin function is shared. The original per-element rule
// and FP32 underflow separation independently anchor the equivalence proof.
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include "../../csrc/qattn/ppu/attn_ppu_probability.cuh"

uint32_t bits(float x) {
  uint32_t result;
  std::memcpy(&result, &x, sizeof(result));
  return result;
}

int main(int argc, char **argv) {
  bool const plant = argc == 2 && std::strcmp(argv[1], "--unmasked-origin") == 0;
  if (argc > 1 && !plant) return 2;
  float const threshold = -1.0e29f;
  float const next_valid = std::nextafter(threshold, 0.0f);
  float const gap = next_valid - threshold;
  // Binary32 exp2 underflows even with gradual underflow below -150.
  // This establishes the separation for every finite masked/valid pair,
  // not just the explicit data points below.
  if (!(threshold < -150.0f && gap > 150.0f &&
        std::exp2(-gap) == 0.0f && std::exp2(threshold) == 0.0f)) return 1;
  std::array<float, 1035> values{};
  int size = 0;
  values[size++] = -std::numeric_limits<float>::max();
  values[size++] = -1.0e30f;
  values[size++] = std::nextafter(threshold, -INFINITY);
  values[size++] = threshold;
  values[size++] = next_valid;
  values[size++] = 0.0f;
  values[size++] = -0.0f;
  for (int i = -512; i <= 512; ++i) values[size++] = float(i) / 4;
  uint64_t cases = 0, bad = 0, legacy_bad = 0;
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      float const s = values[i], maximum = std::fmax(s, values[j]);
      float const wanted = s <= threshold ? 0.0f : std::exp2(s - maximum);
      float const origin = plant ? maximum :
          sageattention::ppu::probability_exponent_origin(maximum);
      float const got = std::exp2(s - origin);
      bad += bits(got) != bits(wanted);
      legacy_bad += bits(std::exp2(s - maximum)) != bits(wanted);
      ++cases;
    }
  }
  std::printf("[P mask origin] cases=%llu/%d squared bad=%llu "
              "masked-valid-min-gap=%g all-finite-separation=PROVED "
              "unmasked-origin-negative=%llu\n",
              static_cast<unsigned long long>(cases), size,
              static_cast<unsigned long long>(bad), double(gap),
              static_cast<unsigned long long>(legacy_bad));
  if (size != 1032 || cases != 1065024 || bad || legacy_bad == 0) return 1;
  std::puts("[P mask origin] RAW-BIT/PASS; native exp2 instruction remains a build/device boundary");
}
