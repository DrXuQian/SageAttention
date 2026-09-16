// SPDX-License-Identifier: Apache-2.0
// Bind only the native HGGC C ABI. Do not deep-bind the Torch/pybind extension:
// its ATen and C++ objects must keep using the process's existing Torch runtime.
#include <hggc_runtime.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>

#ifndef SAGE_PPU_RUNTIME_SONAME
#error "The build must supply the SDK runtime's measured ELF SONAME"
#endif

namespace {
[[noreturn]] void runtime_failure(char const* operation, char const* error) {
  std::fprintf(stderr, "SageAttention native PPU runtime %s: %s: %s\n",
               SAGE_PPU_RUNTIME_SONAME, operation, error ? error : "unknown error");
  std::abort();
}

void* runtime_handle() {
  // Keep this handle for the process lifetime: fatbinary destructors also call
  // the runtime. A handle-scoped lookup bypasses globally loaded legacy shims.
  static void* const handle = [] {
    void* value = dlopen(SAGE_PPU_RUNTIME_SONAME,
                        RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND);
    if (!value) runtime_failure("dlopen", dlerror());
    return value;
  }();
  return handle;
}

template <class Function>
Function runtime_function(char const* name) {
  dlerror();
  void* address = dlsym(runtime_handle(), name);
  char const* error = dlerror();
  if (error || !address) runtime_failure(name, error);
  return reinterpret_cast<Function>(address);
}
}  // namespace

// This declaration list also drives --wrap flags in tools/ppu_native_link.py.
// Hidden wrappers cannot be interposed by another extension. Their types are
// checked against the actual SDK declarations rather than handwritten ABI casts.
#define SAGE_RUNTIME_FORWARD(NAME, RETURN, PARAMETERS, ARGUMENTS)          \
  extern "C" __attribute__((visibility("hidden"))) RETURN                \
  __wrap_##NAME PARAMETERS {                                             \
    using Entry = decltype(static_cast<RETURN (*) PARAMETERS>(&NAME));    \
    static auto const entry = runtime_function<Entry>(#NAME);             \
    return entry ARGUMENTS;                                              \
  }

SAGE_RUNTIME_FORWARD(__hggcPushCallConfiguration, unsigned,
    (dim3 grid, dim3 block, size_t shared, void* stream),
    (grid, block, shared, stream))
SAGE_RUNTIME_FORWARD(__hggcPopCallConfiguration, hggcError_t,
    (dim3* grid, dim3* block, size_t* shared, void* stream),
    (grid, block, shared, stream))
SAGE_RUNTIME_FORWARD(__hggcRegisterFatBinary, void**,
    (void* binary), (binary))
SAGE_RUNTIME_FORWARD(__hggcUnregisterFatBinary, void,
    (void** handle), (handle))
SAGE_RUNTIME_FORWARD(__hggcRegisterFunction, void,
    (void** handle, char const* host, char* device, char const* name,
     int limit, uint3* tid, uint3* bid, dim3* block, dim3* grid, int* warp),
    (handle, host, device, name, limit, tid, bid, block, grid, warp))
SAGE_RUNTIME_FORWARD(__hggcRegisterVar, void,
    (void** handle, char* host, char* device, char const* name,
     int external, size_t bytes, int constant, int global),
    (handle, host, device, name, external, bytes, constant, global))
SAGE_RUNTIME_FORWARD(hggcFuncSetAttribute, hggcError_t,
    (void const* function, hggcFuncAttribute attribute, int value),
    (function, attribute, value))
SAGE_RUNTIME_FORWARD(hggcGetErrorString, char const*,
    (hggcError_t error), (error))
SAGE_RUNTIME_FORWARD(hggcGetLastError, hggcError_t, (), ())
SAGE_RUNTIME_FORWARD(hggcLaunchKernel, hggcError_t,
    (void const* function, dim3 grid, dim3 block, void** arguments,
     size_t shared, hggcStream_t stream),
    (function, grid, block, arguments, shared, stream))

#undef SAGE_RUNTIME_FORWARD
