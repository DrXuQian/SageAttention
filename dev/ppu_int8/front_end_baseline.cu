// Baseline for NVIDIA-nvcc parsing of the PPU-only actlize headers.  The PPU
// shipping build uses hgcc; this TU lets the local gate distinguish fixed
// cross-toolchain header diagnostics from errors added by the Sage kernel.
#include <torch/extension.h>
#include "../../csrc/qattn/ppu/attn_ppu_ops.cuh"

int main() { return 0; }
