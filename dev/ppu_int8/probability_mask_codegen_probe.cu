#include "../../csrc/qattn/ppu/attn_ppu_ops.cuh"

extern "C" __global__ void probability_mask_before(
    float4 const *in, float const *maxima, float4 *out) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  float4 s = in[i];
  float m = maxima[i];
  out[i] = make_float4(s.x <= -1e29f ? 0.f : exp2f(s.x-m),
                      s.y <= -1e29f ? 0.f : exp2f(s.y-m),
                      s.z <= -1e29f ? 0.f : exp2f(s.z-m),
                      s.w <= -1e29f ? 0.f : exp2f(s.w-m));
}

extern "C" __global__ void probability_mask_after(
    float4 const *in, float const *maxima, float4 *out) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  float4 s = in[i];
  float m = sageattention::ppu::probability_exponent_origin(maxima[i]);
  out[i] = make_float4(exp2f(s.x-m), exp2f(s.y-m), exp2f(s.z-m), exp2f(s.w-m));
}
