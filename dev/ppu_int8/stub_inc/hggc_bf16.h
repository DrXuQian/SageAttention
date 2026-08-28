#pragma once
#include <cstdint>
#include <cstring>

struct __ppu_bfloat16 { uint16_t __x; };
struct __ppu_bfloat16_raw {
  uint16_t x;
  __ppu_bfloat16_raw() : x(0) {}
  __ppu_bfloat16_raw(__ppu_bfloat16 v) { std::memcpy(&x, &v.__x, 2); }
};
using bfloat16 = __ppu_bfloat16;

#define SAGE_BF16_CMP(NAME, OP) \
  inline bool NAME(__ppu_bfloat16 a, __ppu_bfloat16 b) { return a.__x OP b.__x; }
SAGE_BF16_CMP(__heq, ==) SAGE_BF16_CMP(__hne, !=)
SAGE_BF16_CMP(__hlt, <)  SAGE_BF16_CMP(__hle, <=)
SAGE_BF16_CMP(__hgt, >)  SAGE_BF16_CMP(__hge, >=)
#undef SAGE_BF16_CMP

#define SAGE_BF16_ARITH(NAME) \
  inline __ppu_bfloat16 NAME(__ppu_bfloat16 a, __ppu_bfloat16) { return a; }
SAGE_BF16_ARITH(__hadd) SAGE_BF16_ARITH(__hsub)
SAGE_BF16_ARITH(__hmul) SAGE_BF16_ARITH(__hdiv)
#undef SAGE_BF16_ARITH
inline __ppu_bfloat16 __hneg(__ppu_bfloat16 a) {
  a.__x ^= 0x8000u;
  return a;
}
struct __ppu_bfloat162 { __ppu_bfloat16 x, y; };
static_assert(sizeof(__ppu_bfloat162) == 4);
