// An independent IEEE identity check, not a device correctness claim.
// Compile without contraction/fast-math; every baseline multiply is observable.
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <xmmintrin.h>

template <class T, class U> T bits(U x) {
  static_assert(sizeof(T) == sizeof(U));
  T y;
  std::memcpy(&y, &x, sizeof(y));
  return y;
}

// This is the one expression being tested in the native microprobe/full body.
static bool needs_scale(float factor, std::string const& plant) {
  if (plant == "skip-nonunit") return false;
  if (plant == "approximate-one") return std::abs(factor - 1.f) > 1.e-5f;
  return factor != 1.f;
}

int main(int argc, char** argv) {
  static_assert(std::numeric_limits<float>::is_iec559 && sizeof(float) == 4);
  std::string const plant = argc == 2 ? argv[1] : "";
  if (!plant.empty() && plant != "skip-nonunit" &&
      plant != "approximate-one" && plant != "missing-output") return 2;
  unsigned const old_csr = _mm_getcsr();
  // The real HGGC kernel resource descriptor must separately prove
  // fp_denorm_flush=0; a host result cannot assert the device mode.
  _mm_setcsr(old_csr & ~unsigned(0x8040));
  std::array<float, 8> const factors = {1.f, std::nextafter(1.f, 0.f),
      std::nextafter(1.f, 2.f), .5f, 0.f, std::ldexp(1.f, -126),
      std::ldexp(1.f, -149), 0.99999f};
  uint64_t checked = 0, identities = 0;
  uint32_t rng = 0x9e3779b9;
  for (unsigned e = 0; e < 255; ++e) {
    for (unsigned m = 0; m < 4096; ++m) {
      for (unsigned sign : {0u, 0x80000000u}) {
        // Every exponent, both signs, exact endpoints and stratified mantissas.
        uint32_t const word = sign | (e << 23) | (m == 4095 ? 0x7fffff : m << 11);
        float const x = bits<float>(word);
        for (float factor : factors) {
          volatile float product = x * factor;
          float const current = product;
          float candidate = needs_scale(factor, plant) ? current : x;
          if (plant == "missing-output" && e == 127 && m == 0) candidate = 0;
          ++checked;
          identities += factor == 1.f;
          if (bits<uint32_t>(current) != bits<uint32_t>(candidate)) {
            std::printf("[identity rescale] FAIL plant=%s x=%a factor=%a old=%a new=%a checked=%llu\n",
                        plant.c_str(), x, factor, current, candidate,
                        static_cast<unsigned long long>(checked));
            _mm_setcsr(old_csr);
            return 1;
          }
          // P/V restore FMA is unchanged. Include signed/cancelling addends.
          rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
          float const addend = float(int(rng % 2049) - 1024) / 1024.f;
          float const a = std::fma(addend, .003921568859f, current);
          float const b = std::fma(addend, .003921568859f, candidate);
          if (bits<uint32_t>(a) != bits<uint32_t>(b)) return 1;
        }
      }
    }
  }
  std::printf("[identity rescale] pairs=%llu identity_pairs=%llu FP32-multiply-and-following-FMA=RAW-EQUAL/PASS; device-mode-must-be-checked\n",
              static_cast<unsigned long long>(checked),
              static_cast<unsigned long long>(identities));
  // Uniform predicate must represent ALL eight rows, not just lane0's row.
  unsigned mixed_cases = 0;
  for (unsigned mask = 0; mask < 256; ++mask) {
    bool any = false;
    for (unsigned row = 0; row < 8; ++row) any |= bool(mask & (1u << row));
    for (unsigned row = 0; row < 8; ++row) {
      float const factor = (mask & (1u << row)) ? .5f : 1.f;
      volatile float x = float(int(row) - 4);
      float const current = x * factor;
      float const candidate = any ? x * factor : x;
      if (bits<uint32_t>(current) != bits<uint32_t>(candidate)) return 1;
      ++mixed_cases;
    }
  }
  std::printf("[identity rescale] warp-masks=256 row-cases=%u RAW-EQUAL/PASS\n", mixed_cases);
  // This is a negative boundary, not permission to assume denormal support.
  // A flush-to-zero multiply can lose a subnormal that the skipped path adds
  // back to a normal FMA. Refuse to extrapolate the identity to that mode.
  for (unsigned mode : {0u, 0x8000u, 0x40u, 0x8040u}) {
    _mm_setcsr((old_csr & ~unsigned(0x8040)) | mode);
    volatile float x = bits<float>(uint32_t(0x00400000));
    volatile float unit = 1.f;
    volatile float normal = bits<float>(uint32_t(0x00800000));
    volatile float product = x * unit;
    volatile float old_value = std::fma(unit, normal, product);
    volatile float new_value = std::fma(unit, normal, x);
    bool const equal = bits<uint32_t>(float(old_value)) == bits<uint32_t>(float(new_value));
    std::printf("[identity rescale FP mode] ftz=%u daz=%u equal=%u %s\n",
                !!(mode & 0x8000), !!(mode & 0x40), equal,
                mode == 0x8000 ? "UNSUPPORTED-MODE/EXPECTED-RED" : "CONTROL");
    if (equal == (mode == 0x8000)) return 1;
  }
  _mm_setcsr(old_csr);
}
