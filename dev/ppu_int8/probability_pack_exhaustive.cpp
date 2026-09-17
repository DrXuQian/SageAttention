// Exhaust the bounded numerical domain of the inverse-dequant packing idea.
// This is IEEE host arithmetic, not a claim to execute PPU instructions.
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

uint32_t bits(float value) {
  uint32_t result;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}
float from_bits(uint32_t value) {
  float result;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

int main(int argc, char **argv) {
  bool const missing = argc == 2 && std::strcmp(argv[1], "--missing-input") == 0;
  if (argc > 1 && !missing) return 2;
  std::fesetround(FE_TONEAREST);
  uint64_t cases = 0, bad = 0;
  uint32_t const final = 0x3f800000u - unsigned(missing);
  // Every positive binary32 bit pattern from +0 through +1, including
  // subnormals. Compile without contraction or fast-math; test the two rounds.
  for (uint32_t word = 0; word <= final; ++word) {
    float const p = from_bits(word);
    float const product = p * 255.f;
    unsigned const expected = unsigned(std::nearbyint(product));
    unsigned const extracted = bits(product + 12582912.f) & 255u;
    bad += expected != extracted;
    ++cases;
  }
  int fused_bad = 0, threshold_cases = 0;
  for (int q = 0; q < 255; ++q) {
    float const p = float((double(q) + 0.5) / 255.0);
    float const points[] = {std::nextafter(p, 0.f), p, std::nextafter(p, 1.f)};
    for (float x : points) {
      float const product = x * 255.f;
      unsigned const expected = unsigned(std::nearbyint(product));
      fused_bad += (bits(std::fma(x, 255.f, 12582912.f)) & 255u) != expected;
      ++threshold_cases;
    }
  }
  bool const negative_zero_ok = (bits((-0.f * 255.f) + 12582912.f) & 255u) == 0;
  std::printf("[P inverse pack] inputs=%llu/1065353217 bad=%llu negative_zero=%d "
              "thresholds=%d/765 fused-rounding-negative=%d\n",
              static_cast<unsigned long long>(cases),
              static_cast<unsigned long long>(bad), int(negative_zero_ok),
              threshold_cases, fused_bad);
  return cases == 1065353217 && bad == 0 && negative_zero_ok &&
      threshold_cases == 765 && fused_bad == 128 ? 0 : 1;
}
