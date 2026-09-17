// Local representation proof and numerical counterexamples, NOT device runs.
// Build with -ffp-contract=off; only the FMA arm uses fused arithmetic.
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>

template <class To, class From> To bit_copy(From x) {
  static_assert(sizeof(To) == sizeof(From));
  To out;
  std::memcpy(&out, &x, sizeof(out));
  return out;
}

int main(int argc, char** argv) {
  static_assert(std::numeric_limits<float>::is_iec559 && sizeof(float) == 4);
  std::string const plant = argc == 2 ? argv[1] : "";
  if (!plant.empty() && plant != "zero-seed" && plant != "wrong-bias" &&
      plant != "allow-fma") return 2;
  constexpr int bound = 128 * 128 * 128; // Includes signed input code -128.
  int const seed = plant == "zero-seed" ? 0 : 0x4b400000;
  float const bias = plant == "wrong-bias" ? 12582911.f : 12582912.f;
  unsigned bad = 0;
  for (int n = -bound; n <= bound; ++n) {
    float value = bit_copy<float>(seed + n);
    float recovered = value - bias;
    if (recovered != static_cast<float>(n)) ++bad;
  }
  std::printf("[score seed] scores=%d range=[%d,%d] seed=0x%08x "
              "mismatches=%u %s\n", 2 * bound + 1, -bound, bound, seed,
              bad, bad ? "FAIL" : "EXACT/PASS");
  if (bad) return 1;

  // All 64 keys have the same legal QK score. V is identically one, so the
  // independent mathematical answer is exactly one regardless of logit size.
  // Current P=exp2(RN(score*scale)-same)=1. A fused subtraction can make P>1;
  // U8 then saturates the numerator while the FP32 denominator still sums P.
  bool found = false;
  unsigned failures = 0, cases = 0;
  std::array<int, 5> const scores = {1001, 128 * 127 * 127,
                                   -128 * 127 * 127, bound, -bound};
  std::array<float, 5> const mantissas = {.1f, .3f, .7f, 1.1f, 1.7f};
  for (int n : scores) for (float m : mantissas) for (int e = -20; e <= 20; ++e) {
    float const scale = std::ldexp(m, e);
    volatile float rounded = static_cast<float>(n) * scale;
    float const residual = std::fma(static_cast<float>(n), scale, -rounded);
    float const p = std::exp2(residual);
    float const scaled = p * 255.f;
    float const code = std::fmin(255.f, std::fmax(0.f, std::nearbyint(scaled)));
    double const output = static_cast<double>(code) / (255.0 * p);
    bool const failed = !std::isfinite(output) || std::abs(output - 1.0) > .002 + .01;
    ++cases;
    failures += failed;
    if (failed && !found) {
      std::printf("[score FMA witness] score=%d scale=%a origin=%a "
                  "residual=%a p=%.9g code=%.9g want=1 got=%.9g "
                  "O_atol=0.002 O_rtol=0.01 EXPECTED-RED\n",
                  n, scale, static_cast<float>(rounded), residual, p, code, output);
      found = true;
    }
  }
  if (!found || plant == "allow-fma") {
    std::puts("[score FMA admission] FAIL: unconditional FMA cannot be admitted");
    return 1;
  }
  std::printf("[score FMA screen] equal-logit-cases=%u rejected=%u "
              "constant-V anchor=1 production=UNCHANGED/PASS\n", cases, failures);

  // Subtraction-before-scale is not old-path raw equivalent, even though the
  // biased integer representation above is exact. Large common offsets can
  // round two distinct products together. Keep that boundary explicit.
  float const scale = .1f;
  float const a = float(bound), b = float(bound - 1);
  volatile float a_scaled = a * scale, b_scaled = b * scale;
  float const old_delta = b_scaled - a_scaled;
  float const new_delta = (b - a) * scale;
  if (bit_copy<uint32_t>(old_delta) == bit_copy<uint32_t>(new_delta)) return 1;
  std::printf("[score difference-first] scores=%d/%d scale=%a old_delta=%a "
              "new_delta=%a RAW-EQUIVALENCE=EXPECTED-RED; "
              "representation proof is not attention admission\n",
              bound, bound - 1, scale, old_delta, new_delta);
}
