// Compile-only screen: one row owns 32 persistent FP32 output values.
// These same scaling operations precede the integer-PV restoration FMA.
#include "../../csrc/qattn/ppu/attn_ppu_ops.cuh"

template <int Mode>
__device__ __forceinline__ void rescale_body(
    float const *input, float const *factor, float *output) {
  int const lane = int(threadIdx.x);
  float values[32];
#pragma unroll
  for (int i = 0; i < 32; ++i) values[i] = input[i * 32 + lane];
  float const rescale = factor[lane / 4];
  if constexpr (Mode == 0) {
#pragma unroll
    for (int i = 0; i < 32; ++i) values[i] *= rescale;
  } else if constexpr (Mode == 1) {
    if (rescale != 1.0f) {
#pragma unroll
      for (int i = 0; i < 32; ++i) values[i] *= rescale;
    }
  } else {
    // A uniform predicate is a separate candidate, not an assumption that
    // the eight different Q rows have identical maxima/factors.
    if (__any_sync(0xffffffffu, rescale != 1.0f)) {
#pragma unroll
      for (int i = 0; i < 32; ++i) values[i] *= rescale;
    }
  }
#pragma unroll
  for (int i = 0; i < 32; ++i) output[i * 32 + lane] = values[i];
}

extern "C" __global__ void rescale_current(
    float const *input, float const *factor, float *output) {
  rescale_body<0>(input, factor, output);
}
extern "C" __global__ void rescale_conditional(
    float const *input, float const *factor, float *output) {
  rescale_body<1>(input, factor, output);
}
extern "C" __global__ void rescale_uniform(
    float const *input, float const *factor, float *output) {
  rescale_body<2>(input, factor, output);
}
