// Isolated before/after device bodies; before is the fbf3d0f implementation.
// The optimized arms call production helpers, not copies of their algorithm.
#include "../../csrc/qattn/ppu/attn_ppu_ops.cuh"

namespace ops = sageattention::ppu;

extern "C" __global__ void requant_before(float4 const *in, uint32_t *out) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  float4 p = in[i];
  float values[4] = {p.x, p.y, p.z, p.w};
  uint32_t packed = 0;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    uint32_t code = uint32_t(max(0, min(255, __float2int_rn(values[j] * 255.f))));
    packed |= code << (8 * j);
  }
  out[i] = packed;
}

extern "C" __global__ void requant_after(float4 const *in, uint32_t *out) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  float4 p = in[i];
  float values[4] = {p.x, p.y, p.z, p.w};
  out[i] = ops::pack_probability_u8(values);
}

extern "C" __global__ void transpose_before(uint4 const *in, uint4 *out) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  int const lane = int(threadIdx.x) & 31;
  uint4 value = in[i];
  uint32_t source[4] = {value.x, value.y, value.z, value.w};
  uint32_t result[4] = {};
#pragma unroll
  for (int word = 0; word < 4; ++word) {
#pragma unroll
    for (int byte = 0; byte < 4; ++byte) {
      uint32_t peer = __shfl_sync(0xffffffffu, source[word], (lane & ~3) + byte);
      result[word] |= ((peer >> (8 * (lane & 3))) & 255u) << (8 * byte);
    }
  }
  out[i] = make_uint4(result[0], result[1], result[2], result[3]);
}

extern "C" __global__ void transpose_after(uint4 const *in, uint4 *out) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  uint4 value = in[i];
  uint32_t left[2] = {value.x, value.z};
  uint32_t right[2] = {value.y, value.w};
  uint32_t result[4];
  ops::probability_to_u8_operand(result, left, right);
  out[i] = make_uint4(result[0], result[1], result[2], result[3]);
}
