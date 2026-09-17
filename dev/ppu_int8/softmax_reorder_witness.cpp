// Offline arithmetic-boundary witnesses, NOT an attention correctness oracle.
// Compile with -ffp-contract=off; std::fma is intentional only in the FMA arm.
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>

static uint32_t bits(float x) {
  uint32_t result;
  std::memcpy(&result, &x, sizeof(result));
  return result;
}

// One row is split over four lanes, each holding at most 16 probabilities.
// The largest probability is 1; rescale and block_rescale are both exactly 1.
// Thus this is a legal special case of both online-softmax recurrences.
static float reduce(std::array<float, 4> x) {
  return (x[0] + x[1]) + (x[2] + x[3]);
}

static std::array<float, 2> denominator(int blocks, std::array<float, 4> sum) {
  float eager = 0;
  std::array<float, 4> deferred{};
  for (int k = 0; k < blocks; ++k) {
    eager += reduce(sum);
    for (int lane = 0; lane < 4; ++lane) deferred[lane] += sum[lane];
  }
  return {eager, reduce(deferred)};
}

int main() {
  static_assert(std::numeric_limits<float>::is_iec559 && sizeof(float) == 4);
  auto single = denominator(1, {16, 0x1p-20f, 0, 0});
  auto exact = denominator(1153, {1, 2, 4, 8});
  auto rounding = denominator(1153, {16, 0x1p-20f, 0, 0});
  if (bits(single[0]) != bits(single[1]) || bits(exact[0]) != bits(exact[1]) ||
      bits(rounding[0]) == bits(rounding[1])) return 1;
  std::printf("[denominator reorder] one-block=RAW-EQUAL exact-1153=RAW-EQUAL "
              "rounding-1153 eager=%.9g/0x%08x deferred=%.9g/0x%08x "
              "RAW-EQUIVALENCE-CLAIM=EXPECTED-RED/PASS\n",
              rounding[0], bits(rounding[0]), rounding[1], bits(rounding[1]));

  // A representable positive scale and legal S32 QK score. The old path
  // materializes RN(score*scale) before subtracting the row maximum; native
  // FMA computes that subtraction with one final rounding instead of two.
  volatile float score_source = 1001.0f, scale_source = 0.1f;
  float score = score_source, scale = scale_source;
  volatile float rounded_product = score * scale;
  float origin = rounded_product;
  float separate = rounded_product - origin;
  float fused = std::fma(score, scale, -origin);
  float positive = std::fma(16.0f, 0.5f, -8.0f);
  if (bits(separate) == bits(fused) || bits(positive) != bits(0.0f)) return 2;
  std::printf("[score FMA] score=1001 scale=0.1 origin=%.9g separate=%a/0x%08x "
              "fused=%a/0x%08x exact-control=RAW-EQUAL "
              "RAW-EQUIVALENCE-CLAIM=EXPECTED-RED/PASS\n",
              origin, separate, bits(separate), fused, bits(fused));
  std::puts("[softmax witnesses] PASS: opportunities require a new numerical gate; production unchanged");
}
