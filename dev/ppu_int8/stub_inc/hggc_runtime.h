#pragma once
#include <cstdio>

// Host-oracle-only declarations.  This file is never on the PPU build include
// path and deliberately does not model any device behavior.
enum class hggcError_t : int { success = 0 };
inline constexpr hggcError_t hggcSuccess = hggcError_t::success;
inline bool operator==(hggcError_t a, int b) { return int(a) == b; }
inline bool operator!=(hggcError_t a, int b) { return int(a) != b; }
inline char const *hggcGetErrorName(hggcError_t) { return "host-oracle"; }
inline char const *hggcGetErrorString(hggcError_t) { return "host-oracle"; }
