// Compile-only zero-source alternatives. No aliasing of fragment arrays with
// a single scalar reference: each output and input operand is passed explicitly.
#include "../../csrc/qattn/ppu/attn_ppu_ops.cuh"
#include <hgrt/hggc_mma.h>

template <bool UnsignedA, int Mode>
__device__ __forceinline__ void first_product(
    uint32_t (&d)[8], uint32_t const (&a)[4], uint32_t const (&b)[4]) {
  using Atom = std::conditional_t<UnsignedA,
      cute::PPU0010_16x16x32_S32U8S8S32_TN,
      cute::PPU0010_16x16x32_S32S8S8S32_TN>;
  if constexpr (Mode == 0) {
#pragma unroll
    for (int i = 0; i < 8; ++i) d[i] = 0;
    Atom::fma(d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7],
              a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3],
              d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]);
  } else if constexpr (Mode == 1) {
    uint32_t const zero = 0;
    Atom::fma(d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7],
              a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3],
              zero, zero, zero, zero, zero, zero, zero, zero);
  } else if constexpr (Mode == 2) {
    uint32_t const zero[8] = {};
    if constexpr (UnsignedA)
      awmma::mma_dense_sync<16,16,32,int,unsigned char,signed char,int>(d,a,b,zero);
    else
      awmma::mma_dense_sync<16,16,32,int,signed char,signed char,int>(d,a,b,zero);
  }
#if defined(PROBE_LITERAL_TUPLE) || defined(PROBE_LITERAL_SCALAR)
  else {
#ifdef PROBE_LITERAL_TUPLE
#define ZERO_INPUT "{0,0,0,0,0,0,0,0};\n"
#else
#define ZERO_INPUT "0;\n"
#endif
#define ZERO_MMA(TYPE) \
    asm volatile("ppu.tc01.mma.sync.aligned.m16n16k32.row.col.s32." TYPE ".s8.s32 " \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, " ZERO_INPUT \
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), \
        "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7]) \
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), \
        "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]));
    if constexpr (UnsignedA) { ZERO_MMA("u8"); }
    else { ZERO_MMA("s8"); }
#undef ZERO_MMA
#undef ZERO_INPUT
  }
#endif
}

template <bool UnsignedA, int Mode>
__device__ __forceinline__ void run(uint4 const *a, uint4 const *b, uint32_t *out) {
  int const lane = int(threadIdx.x);
  uint4 av = a[lane], bv = b[lane];
  uint32_t af[4] = {av.x,av.y,av.z,av.w}, bf[4] = {bv.x,bv.y,bv.z,bv.w};
  uint32_t d[8];
  first_product<UnsignedA, Mode>(d, af, bf);
  av = a[32+lane]; bv = b[32+lane];
  uint32_t af2[4] = {av.x,av.y,av.z,av.w}, bf2[4] = {bv.x,bv.y,bv.z,bv.w};
  using Atom = std::conditional_t<UnsignedA,
      cute::PPU0010_16x16x32_S32U8S8S32_TN,
      cute::PPU0010_16x16x32_S32S8S8S32_TN>;
  Atom::fma(d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],
            af2[0],af2[1],af2[2],af2[3],bf2[0],bf2[1],bf2[2],bf2[3],
            d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7]);
#pragma unroll
  for (int i=0; i<8; ++i) out[lane*8+i]=d[i];
}

#define PROBE(NAME, UNSIGNED, MODE) \
extern "C" __global__ void NAME(uint4 const *a,uint4 const *b,uint32_t *o) { run<UNSIGNED,MODE>(a,b,o); }
#if defined(PROBE_LITERAL_TUPLE) || defined(PROBE_LITERAL_SCALAR)
PROBE(zero_literal_s8, false, 3)
PROBE(zero_literal_u8, true, 3)
#else
PROBE(zero_inplace_s8, false, 0)
PROBE(zero_inplace_u8, true, 0)
PROBE(zero_operand_s8, false, 1)
PROBE(zero_operand_u8, true, 1)
PROBE(zero_builtin_s8, false, 2)
PROBE(zero_builtin_u8, true, 2)
#endif
#undef PROBE

#if !defined(PROBE_LITERAL_TUPLE) && !defined(PROBE_LITERAL_SCALAR)
// Zero reuse must be tested across multiple independent products. A one-MMA
// probe cannot show whether a single immutable zero group can serve them all.
template <int Mode>
__device__ __forceinline__ void many(uint4 const *a, uint4 const *b, uint32_t *out) {
  uint32_t zeros[8] = {};
  if constexpr (Mode == 2) {
#pragma unroll
    for (int i=0; i<8; ++i) asm volatile("mov.b32 %0, 0;" : "=r"(zeros[i]));
  }
  uint32_t acc[8][8];
#pragma unroll
  for (int product=0; product<8; ++product) {
    uint4 av=a[threadIdx.x], bv=b[product*32+threadIdx.x];
    uint32_t af[4]={av.x,av.y,av.z,av.w}, bf[4]={bv.x,bv.y,bv.z,bv.w};
    auto &d=acc[product];
    if constexpr (Mode == 0) first_product<false,0>(d,af,bf);
    else {
      cute::PPU0010_16x16x32_S32S8S8S32_TN::fma(
        d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],
        af[0],af[1],af[2],af[3],bf[0],bf[1],bf[2],bf[3],
        zeros[0],zeros[1],zeros[2],zeros[3],zeros[4],zeros[5],zeros[6],zeros[7]);
    }
  }
#pragma unroll
  for (int step=1; step<4; ++step) {
    uint4 av=a[step*32+threadIdx.x];
    uint32_t af[4]={av.x,av.y,av.z,av.w};
#pragma unroll
    for (int product=0; product<8; ++product) {
      uint4 bv=b[(step*8+product)*32+threadIdx.x];
      uint32_t bf[4]={bv.x,bv.y,bv.z,bv.w};
      auto &d=acc[product];
      cute::PPU0010_16x16x32_S32S8S8S32_TN::fma(
        d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],
        af[0],af[1],af[2],af[3],bf[0],bf[1],bf[2],bf[3],
        d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7]);
    }
  }
#pragma unroll
  for (int product=0; product<8; ++product) {
#pragma unroll
    for (int i=0; i<8; ++i) out[(product*32+threadIdx.x)*8+i]=acc[product][i];
  }
}
extern "C" __global__ void many_zero_inplace(uint4 const *a,uint4 const *b,uint32_t *o) { many<0>(a,b,o); }
extern "C" __global__ void many_zero_shared_const(uint4 const *a,uint4 const *b,uint32_t *o) { many<1>(a,b,o); }
extern "C" __global__ void many_zero_shared_opaque(uint4 const *a,uint4 const *b,uint32_t *o) { many<2>(a,b,o); }
#endif
