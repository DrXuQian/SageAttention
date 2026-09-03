// PPU-only code-generation bisection for the score-C -> PV-A bridge.
// Neither kernel is shipped or launched.  Keeping the raw, device-proven
// fattn_ppu.cu spelling beside the actlize spelling lets hgobjdump establish
// whether the abstraction preserves the exact register contract.
#include <cstdint>

#include "../../csrc/qattn/ppu/attn_ppu_ops.cuh"

namespace sageattention::ppu::probe {

// Exact historical defect: the API receives independent references, but the
// implementation walks from &c0 as though c0..c3 were one array.  The caller
// below aliases all four references to one zero object, so c[1..3] are not
// zeros.  Retain this only as a generated-ISA negative.
__device__ __forceinline__ void legacy_contiguous_reference_bridge(
    uint32_t &d0, uint32_t &d1, uint32_t &d2, uint32_t &d3,
    uint32_t const &a0, uint32_t const &a1,
    uint32_t const &a2, uint32_t const &a3,
    uint32_t const &b0, uint32_t const &b1,
    uint32_t const &b2, uint32_t const &b3,
    uint32_t const &c0, uint32_t const &, uint32_t const &, uint32_t const &) {
  float *d = reinterpret_cast<float *>(&d0);
  float const *a = reinterpret_cast<float const *>(&a0);
  float const *b = reinterpret_cast<float const *>(&b0);
  float const *c = reinterpret_cast<float const *>(&c0);
  asm volatile(
      "ppu.mma.sync.aligned.m16n16k16.row.col.f16.f16.f16.f16 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, "
      "{%12,%13,%14,%15};\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "f"(a[0]), "f"(a[1]), "f"(a[2]), "f"(a[3]),
        "f"(b[0]), "f"(b[1]), "f"(b[2]), "f"(b[3]),
        "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

__device__ __forceinline__ void raw_bridge(
    uint32_t (&dst)[4], float const (&score)[8]) {
  uint32_t a[4];
  auto *packed = reinterpret_cast<__half2 *>(a);
  packed[0] = __floats2half2_rn(score[0], score[1]);
  packed[1] = __floats2half2_rn(score[2], score[3]);
  packed[2] = __floats2half2_rn(score[4], score[5]);
  packed[3] = __floats2half2_rn(score[6], score[7]);

  int const lane = int(threadIdx.x);
  bool const high = lane >> 4;
  unsigned const inner = unsigned(lane) & 15;
  unsigned const row = inner >> 2;
  unsigned const column = inner & 3;
  unsigned identity = 0;
  if (!high && row == column) identity = 0x00003c00u;
  if (high && row == column) identity = 0x3c000000u;
  uint32_t const b[4] = {identity, 0, 0, identity};
  float const zero = 0.0f;
  auto *d = reinterpret_cast<float *>(dst);
  auto const *af = reinterpret_cast<float const *>(a);
  auto const *bf = reinterpret_cast<float const *>(b);
  asm volatile(
      "ppu.mma.sync.aligned.m16n16k16.row.col.f16.f16.f16.f16 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, "
      "{%12,%13,%14,%15};\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "f"(af[0]), "f"(af[1]), "f"(af[2]), "f"(af[3]),
        "f"(bf[0]), "f"(bf[1]), "f"(bf[2]), "f"(bf[3]),
        "f"(zero), "f"(zero), "f"(zero), "f"(zero));
}

__global__ void bridge_atom_codegen(uint32_t *dst, float const *src) {
  float score[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) score[i] = src[8 * int(threadIdx.x) + i];
  uint32_t converted[4];
  score_accumulator_to_pv_operand(converted, score);
#pragma unroll
  for (int i = 0; i < 4; ++i) dst[4 * int(threadIdx.x) + i] = converted[i];
}

__global__ void bridge_raw_codegen(uint32_t *dst, float const *src) {
  float score[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) score[i] = src[8 * int(threadIdx.x) + i];
  uint32_t converted[4];
  raw_bridge(converted, score);
#pragma unroll
  for (int i = 0; i < 4; ++i) dst[4 * int(threadIdx.x) + i] = converted[i];
}

__global__ void bridge_legacy_codegen(uint32_t *dst, float const *src) {
  float score[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) score[i] = src[8 * int(threadIdx.x) + i];
  uint32_t a[4];
  auto *packed = reinterpret_cast<__half2 *>(a);
  packed[0] = __floats2half2_rn(score[0], score[1]);
  packed[1] = __floats2half2_rn(score[2], score[3]);
  packed[2] = __floats2half2_rn(score[4], score[5]);
  packed[3] = __floats2half2_rn(score[6], score[7]);
  int const lane = int(threadIdx.x) & 31;
  uint32_t const b[4] = {
      layout::pv_bridge_b_word(lane, 0),
      layout::pv_bridge_b_word(lane, 1),
      layout::pv_bridge_b_word(lane, 2),
      layout::pv_bridge_b_word(lane, 3)};
  uint32_t const zero = 0;
  uint32_t converted[4];
  legacy_contiguous_reference_bridge(
      converted[0], converted[1], converted[2], converted[3],
      a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3],
      zero, zero, zero, zero);
#pragma unroll
  for (int i = 0; i < 4; ++i) dst[4 * int(threadIdx.x) + i] = converted[i];
}

}  // namespace sageattention::ppu::probe
