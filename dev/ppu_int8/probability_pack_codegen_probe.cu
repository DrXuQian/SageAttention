// Counterfactuals only: strict inverse-magic packing is correct but not cheaper.
// The fused form changes threshold rounding and is NOT a production candidate.
#include "../../csrc/qattn/ppu/attn_ppu_ops.cuh"

__device__ __forceinline__ unsigned pick_low_bytes(unsigned a, unsigned b,
                                                   unsigned c, unsigned d) {
  return __byte_perm(__byte_perm(a, b, 0x40), __byte_perm(c, d, 0x40), 0x5410);
}

extern "C" __global__ void p_pack_current(float4 const *in, unsigned *out) {
  unsigned const i = blockIdx.x * blockDim.x + threadIdx.x;
  float4 p = in[i];
  float values[4] = {p.x, p.y, p.z, p.w};
  out[i] = sageattention::ppu::pack_probability_u8(values);
}

extern "C" __global__ void p_pack_magic_strict(float4 const *in, unsigned *out) {
  unsigned const i = blockIdx.x * blockDim.x + threadIdx.x;
  float4 p = in[i];
  out[i] = pick_low_bytes(
      __float_as_uint(__fadd_rn(p.x * 255.f, 12582912.f)),
      __float_as_uint(__fadd_rn(p.y * 255.f, 12582912.f)),
      __float_as_uint(__fadd_rn(p.z * 255.f, 12582912.f)),
      __float_as_uint(__fadd_rn(p.w * 255.f, 12582912.f)));
}

extern "C" __global__ void p_pack_magic_fused(float4 const *in, unsigned *out) {
  unsigned const i = blockIdx.x * blockDim.x + threadIdx.x;
  float4 p = in[i];
  out[i] = pick_low_bytes(
      __float_as_uint(__fmaf_rn(p.x, 255.f, 12582912.f)),
      __float_as_uint(__fmaf_rn(p.y, 255.f, 12582912.f)),
      __float_as_uint(__fmaf_rn(p.z, 255.f, 12582912.f)),
      __float_as_uint(__fmaf_rn(p.w, 255.f, 12582912.f)));
}
