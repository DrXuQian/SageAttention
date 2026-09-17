// Compile-only positive-scale/unmasked score screen, not an attention kernel.
// Each lane processes a 16x64 QK tile with the real shipping S32 MMA atom.
// Keep maximum and probabilities observable; count the whole helper and keep
// the changed subtraction/scaling rounding explicit in the host witness.
#include "../../csrc/qattn/ppu/attn_ppu_ops.cuh"

template <bool Biased>
__device__ __forceinline__ void score_body(
    uint4 const *a, uint4 const *b, float *output, float scale) {
  int const lane = int(threadIdx.x);
  union Storage { int32_t i[4][8]; float f[4][8]; } score;
#pragma unroll
  for (int kb = 0; kb < 4; ++kb) {
#pragma unroll
    for (int e = 0; e < 8; ++e) score.i[kb][e] = Biased ? 0x4b400000 : 0;
  }
#pragma unroll
  for (int step = 0; step < 4; ++step) {
    uint4 const av = a[step * 32 + lane];
    uint32_t af[4] = {av.x, av.y, av.z, av.w};
#pragma unroll
    for (int kb = 0; kb < 4; ++kb) {
      uint4 const bv = b[(step * 4 + kb) * 32 + lane];
      uint32_t bf[4] = {bv.x, bv.y, bv.z, bv.w};
      sageattention::ppu::mma_s8s8s32(score.i[kb], af, bf);
    }
  }
  if constexpr (!Biased) {
#pragma unroll
    for (int kb = 0; kb < 4; ++kb) {
#pragma unroll
      for (int e = 0; e < 8; ++e) score.f[kb][e] = float(score.i[kb][e]) * scale;
    }
  }
#pragma unroll
  for (int row = 0; row < 2; ++row) {
    float maximum = -1.e30f;
#pragma unroll
    for (int kb = 0; kb < 4; ++kb) {
#pragma unroll
      for (int col = 0; col < 4; ++col) maximum = fmaxf(maximum, score.f[kb][row * 4 + col]);
    }
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, 1));
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, 2));
    // An online-softmax caller also needs the correctly scaled row maximum.
    // Omitting this restore would unfairly undercount the candidate.
    float scaled_maximum = maximum;
    if constexpr (Biased) scaled_maximum = (maximum - 12582912.f) * scale;
    output[32 * 32 + lane * 2 + row] = scaled_maximum;
#pragma unroll
    for (int kb = 0; kb < 4; ++kb) {
#pragma unroll
      for (int col = 0; col < 4; ++col) {
        int const e = row * 4 + col;
        float delta = score.f[kb][e] - maximum;
        if constexpr (Biased) delta *= scale;
        output[(kb * 8 + e) * 32 + lane] = exp2f(delta);
      }
    }
  }
}

extern "C" __global__ void score_current(uint4 const *a, uint4 const *b, float *out, float scale) {
  score_body<false>(a, b, out, scale);
}
extern "C" __global__ void score_biased_difference(uint4 const *a, uint4 const *b, float *out, float scale) {
  score_body<true>(a, b, out, scale);
}
