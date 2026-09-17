/* Copyright (c) 2026, SageAttention PPU contributors. */
#pragma once

#include <torch/extension.h>

torch::Tensor qk_int8_sv_f16_accum_f32_attn_ppu(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor output,
    torch::Tensor query_scale,
    torch::Tensor key_scale,
    int tensor_layout,
    int is_causal,
    int qk_quant_gran,
    float softmax_scale,
    int return_lse);

void quant_per_warp_int8_ppu(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor scale,
    int block_size,
    int warp_block_size,
    int tensor_layout);

void quant_per_block_int8_ppu(
    torch::Tensor input,
    torch::Tensor mean,
    torch::Tensor output,
    torch::Tensor scale,
    int block_size,
    int tensor_layout);
