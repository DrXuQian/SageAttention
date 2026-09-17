/*
 * Copyright (c) 2024 by SageAttention team.
 * Copyright (c) 2026, SageAttention PPU contributors.
 *
 * Licensed under the Apache License, Version 2.0.
 */

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <type_traits>

#include <torch/extension.h>

#include <hggc_bf16.h>
#include <hggc_fp16.h>
#include <hggc_runtime.h>

#include "attn_ppu_ops.cuh"

namespace sageattention::ppu {
constexpr int kBlockQ = 128;
constexpr int kBlockKV = 64;
constexpr int kWarpQ = 32;
constexpr int kWarps = kBlockQ / kWarpQ;
constexpr float kLog2E = 1.4426950408889634f;

namespace detail {

template <int HeadDim, bool Causal, bool ReturnLse, typename Output, bool Int8PV>
__launch_bounds__(128, 1)
__global__ void qk_int8_pv_kernel(
    int8_t const *__restrict__ query,
    int8_t const *__restrict__ key,
    std::conditional_t<Int8PV, int8_t, cutlass::half_t> const *__restrict__ value,
    Output *__restrict__ output,
    float *__restrict__ lse,
    float const *__restrict__ query_scale,
    float const *__restrict__ key_scale,
    float const *__restrict__ value_scale,
    int qo_len, int kv_len, int num_qo_heads, int num_kv_heads,
    int stride_bz_q, int stride_seq_q, int stride_h_q,
    int stride_bz_k, int stride_seq_k, int stride_h_k,
    int stride_bz_v, int stride_seq_v, int stride_h_v,
    int stride_bz_o, int stride_seq_o, int stride_h_o,
    float softmax_scale) {
  static_assert(HeadDim == 64 || HeadDim == 128,
                "PPU SageAttention supports head dimensions 64 and 128");
  static_assert(kBlockQ % kWarpQ == 0);
  constexpr int kQBlocksPerWarp = kWarpQ / 16;
  constexpr int kKVBlocks = kBlockKV / 16;
  constexpr int kQKSteps = HeadDim / 32;
  constexpr int kVBlocks = HeadDim / 16;
  constexpr int kHeadSlices = HeadDim / 64;
  using Value = std::conditional_t<Int8PV, int8_t, cutlass::half_t>;

  int const lane = int(threadIdx.x);
  int const warp = int(threadIdx.y);

  int batch;
  int head;
  int q_tile;
  if constexpr (Causal) {
    int const heads_and_batches = int(gridDim.y * gridDim.z);
    int const linear =
        (int(blockIdx.z) * int(gridDim.y) + int(blockIdx.y)) * int(gridDim.x)
        + int(blockIdx.x);
    int const z = linear % heads_and_batches;
    batch = z / int(gridDim.y);
    head = z % int(gridDim.y);
    q_tile = int(gridDim.x) - 1 - linear / heads_and_batches;
  } else {
    batch = int(blockIdx.z);
    head = int(blockIdx.y);
    q_tile = int(gridDim.x) - 1 - int(blockIdx.x);
  }

  int const q0 = q_tile * kBlockQ;
  int const groups = num_qo_heads / num_kv_heads;
  int const kv_head = head / groups;

  auto const *q_base = query + int64_t(batch) * stride_bz_q
      + int64_t(head) * stride_h_q;
  auto const *k_base = key + int64_t(batch) * stride_bz_k
      + int64_t(kv_head) * stride_h_k;
  auto const *v_base = value + int64_t(batch) * stride_bz_v
      + int64_t(kv_head) * stride_h_v;

  extern __shared__ __align__(128) unsigned char smem_raw[];
  auto *smem_q = reinterpret_cast<int8_t *>(smem_raw);
  auto *smem_k = smem_q + kBlockQ * HeadDim;
  auto *smem_v = reinterpret_cast<Value *>(
      smem_k + kBlockKV * HeadDim);
  auto *smem_v_scale = reinterpret_cast<float *>(smem_v + kBlockKV * HeadDim);
  float const *v_scale_head = nullptr;
  if constexpr (use_shared_value_scale<HeadDim, Causal, Int8PV>) {
    v_scale_head = value_scale + ValueScaleStage<HeadDim>::head_offset(
        batch, kv_head, num_kv_heads, (kv_len + kBlockKV - 1) / kBlockKV);
  }

  auto issue_k = [&](int kv_block) {
    if (kv_block * kBlockKV < kv_len && lane == 0 && warp == 0) {
#pragma unroll
      for (int cube = 0; cube < kHeadSlices; ++cube) {
        aiu_load_swizzled_64<int8_t, kBlockKV>(
            smem_k + cube * kBlockKV * 64, k_base,
            kv_len, stride_seq_k, kv_block * kBlockKV, cube * 64);
      }
    }
    commit_async_group();
  };

  auto issue_v = [&](int kv_block) {
    if constexpr (use_shared_value_scale<HeadDim, Causal, Int8PV>) {
      ValueScaleStage<HeadDim>::publish(
          smem_v_scale, v_scale_head, kv_block, warp * 32 + lane);
    }
    if (lane == 0 && warp == 0) {
#pragma unroll
      for (int cube = 0; cube < kHeadSlices; ++cube) {
        if constexpr (Int8PV) {
          // V was quantized/transposed into [B,H,K64-block,D,64]. This is
          // the same non-transposed INT8 AIU/TSM path already used for K.
          aiu_load_swizzled_64<int8_t, 64>(
              smem_v + cube * 64 * 64,
              v_base + int64_t(kv_block) * HeadDim * 64,
              HeadDim, 64, cube * 64, 0);
        } else {
          aiu_load_swizzled_64<cutlass::half_t, kBlockKV>(
              smem_v + cube * kBlockKV * 64, v_base,
              kv_len, stride_seq_v, kv_block * kBlockKV, cube * 64);
        }
      }
    }
    commit_async_group();
  };

  if (lane == 0 && warp == 0) {
#pragma unroll
    for (int cube = 0; cube < kHeadSlices; ++cube) {
      aiu_load_swizzled_64<int8_t, kBlockQ>(
          smem_q + cube * kBlockQ * 64, q_base,
          qo_len, stride_seq_q, q0, cube * 64);
    }
  }
  commit_async_group();
  wait_async_group<0>();
  __syncthreads();

  float out[kVBlocks][kQBlocksPerWarp][8];
#pragma unroll
  for (int d = 0; d < kVBlocks; ++d) {
#pragma unroll
    for (int qb = 0; qb < kQBlocksPerWarp; ++qb) {
#pragma unroll
      for (int i = 0; i < 8; ++i) out[d][qb][i] = 0.0f;
    }
  }

  // PPU0010 CLayout is M-fastest: each lane owns two rows and four columns
  // per row.  The four peers of a row differ in lane%4.
  float row_max[kQBlocksPerWarp][2];
  float row_sum[kQBlocksPerWarp][2];
#pragma unroll
  for (int qb = 0; qb < kQBlocksPerWarp; ++qb) {
#pragma unroll
    for (int row_slot = 0; row_slot < 2; ++row_slot) {
      row_max[qb][row_slot] = -1.0e30f;
      row_sum[qb][row_slot] = 0.0f;
    }
  }

  int kv_limit = kv_len;
  if constexpr (Causal) {
    int const q_last = min(q0 + kBlockQ - 1, qo_len - 1);
    kv_limit = min(kv_len, q_last + (kv_len - qo_len) + 1);
    kv_limit = max(kv_limit, 0);
  }
  int const num_kv_blocks = (kv_limit + kBlockKV - 1) / kBlockKV;
  int const q_scale_blocks = int(gridDim.x) * kWarps;
  int const k_scale_blocks = (kv_len + kBlockKV - 1) / kBlockKV;
  float const q_scale = query_scale[
      (int64_t(batch) * num_qo_heads + head) * q_scale_blocks
      + q_tile * kWarps + warp];

  issue_k(0);
  for (int kv = 0; kv < num_kv_blocks; ++kv) {
    __syncthreads();
    issue_v(kv);
    wait_async_group<1>();
    __syncthreads();

    union ScoreStorage {
      int32_t integer[kQBlocksPerWarp][kKVBlocks][8];
      float fp32[kQBlocksPerWarp][kKVBlocks][8];
    } score;
    uint32_t p_codes[Int8PV ? kQBlocksPerWarp : 1][Int8PV ? kKVBlocks : 1][2];
    float p_scale[Int8PV ? kQBlocksPerWarp : 1][2];
#pragma unroll
    for (int qb = 0; qb < kQBlocksPerWarp; ++qb) {
#pragma unroll
      for (int kb = 0; kb < kKVBlocks; ++kb) {
#pragma unroll
        for (int i = 0; i < 8; ++i) score.integer[qb][kb][i] = 0;
      }
    }

#pragma unroll
    for (int step = 0; step < kQKSteps; ++step) {
      uint32_t q_fragment[kQBlocksPerWarp][4];
#pragma unroll
      for (int qb = 0; qb < kQBlocksPerWarp; ++qb) {
        load_swizzled<int8_t, kBlockQ, kHeadSlices>(
            q_fragment[qb], smem_q, warp * kWarpQ + qb * 16,
            (step & 1) * 32, step >> 1);
      }
#pragma unroll
      for (int kb = 0; kb < kKVBlocks; ++kb) {
        uint32_t k_fragment[4];
        load_swizzled<int8_t, kBlockKV, kHeadSlices>(
            k_fragment, smem_k, kb * 16,
            (step & 1) * 32, step >> 1);
#pragma unroll
        for (int qb = 0; qb < kQBlocksPerWarp; ++qb) {
          mma_s8s8s32(score.integer[qb][kb], q_fragment[qb], k_fragment);
        }
      }
    }

    __syncthreads();
    issue_k(kv + 1);

    float const k_scale = key_scale[
        (int64_t(batch) * num_kv_heads + kv_head) * k_scale_blocks + kv];
    float const score_scale = q_scale * k_scale * softmax_scale * kLog2E;
    bool const edge = (kv + 1) * kBlockKV > kv_len;
    bool const causal_edge = Causal &&
        ((kv + 1) * kBlockKV - 1 > q0 + (kv_len - qo_len));

#pragma unroll
    for (int qb = 0; qb < kQBlocksPerWarp; ++qb) {
#pragma unroll
      for (int row_slot = 0; row_slot < 2; ++row_slot) {
        int const q_in_tile = warp * kWarpQ + qb * 16
            + layout::accumulator_row(lane, row_slot * 4);
        int const q_position = q0 + q_in_tile + (kv_len - qo_len);
        float tile_max = -1.0e30f;
#pragma unroll
        for (int kb = 0; kb < kKVBlocks; ++kb) {
          float *s = score.fp32[qb][kb];
#pragma unroll
          for (int column_slot = 0; column_slot < 4; ++column_slot) {
            int const e = row_slot * 4 + column_slot;
            int32_t const quantized = score.integer[qb][kb][e];
            int const kv_col = kv * kBlockKV + kb * 16
                + layout::accumulator_column(lane, e);
            float value_f = float(quantized) * score_scale;
            if ((edge && kv_col >= kv_len) ||
                (causal_edge && kv_col > q_position)) {
              value_f = -1.0e30f;
            }
            s[e] = value_f;
          }
          int const base = row_slot * 4;
          tile_max = fmaxf(tile_max, fmaxf(fmaxf(s[base], s[base + 1]),
                                           fmaxf(s[base + 2], s[base + 3])));
        }
        tile_max = fmaxf(tile_max,
            __shfl_xor_sync(0xffffffffu, tile_max, 1));
        tile_max = fmaxf(tile_max,
            __shfl_xor_sync(0xffffffffu, tile_max, 2));

        float const new_max = fmaxf(tile_max, row_max[qb][row_slot]);
        float const rescale = exp2f(row_max[qb][row_slot] - new_max);
        float block_rescale = 1.0f;
        if constexpr (Int8PV) {
          block_rescale = exp2f(tile_max - new_max);
          p_scale[qb][row_slot] = block_rescale / 255.0f;
        }
        row_max[qb][row_slot] = new_max;
#pragma unroll
        for (int d = 0; d < kVBlocks; ++d) {
#pragma unroll
          for (int e = row_slot * 4; e < row_slot * 4 + 4; ++e) {
            out[d][qb][e] *= rescale;
          }
        }

        float tile_sum = 0.0f;
        // Only integer PV uses the finite mask sentinel; classify it once
        // per row, keeping all-masked rows zero without per-P predicates.
        float const probability_origin = Int8PV
            ? probability_exponent_origin(tile_max) : 0.0f;
#pragma unroll
        for (int kb = 0; kb < kKVBlocks; ++kb) {
          float *s = score.fp32[qb][kb];
          int const base = row_slot * 4;
          if constexpr (Int8PV) {
            float probability[4];
#pragma unroll
            for (int cs = 0; cs < 4; ++cs) {
              float const p = exp2f(s[base + cs] - probability_origin);
              tile_sum += p;
              probability[cs] = p;
            }
            p_codes[qb][kb][row_slot] = pack_probability_u8(probability);
          } else {
            s[base] = exp2f(s[base] - new_max);
            s[base + 1] = exp2f(s[base + 1] - new_max);
            s[base + 2] = exp2f(s[base + 2] - new_max);
            s[base + 3] = exp2f(s[base + 3] - new_max);
            tile_sum += s[base] + s[base + 1] + s[base + 2] + s[base + 3];
          }
        }
        tile_sum += __shfl_xor_sync(0xffffffffu, tile_sum, 1);
        tile_sum += __shfl_xor_sync(0xffffffffu, tile_sum, 2);
        row_sum[qb][row_slot] =
            row_sum[qb][row_slot] * rescale + tile_sum * block_rescale;
      }
    }

    wait_async_group<1>();
    __syncthreads();
    if constexpr (Int8PV) {
      uint32_t probability[kQBlocksPerWarp][2][4];
#pragma unroll
      for (int qb = 0; qb < kQBlocksPerWarp; ++qb) {
#pragma unroll
        for (int step = 0; step < 2; ++step) {
          probability_to_u8_operand(probability[qb][step],
                                    p_codes[qb][2 * step], p_codes[qb][2 * step + 1]);
        }
      }
#pragma unroll
      for (int d = 0; d < kVBlocks; ++d) {
        float v_scales[4];
#pragma unroll
        for (int cs = 0; cs < 4; ++cs) {
          int const channel = d * 16 + layout::accumulator_column(lane, cs);
          if constexpr (use_shared_value_scale<HeadDim, Causal, Int8PV>) {
            v_scales[cs] = smem_v_scale[channel];
          } else {
            v_scales[cs] = value_scale[
                ((int64_t(batch) * num_kv_heads + kv_head) * k_scale_blocks + kv)
                * HeadDim + channel];
          }
        }
        int32_t partial[kQBlocksPerWarp][8] = {};
#pragma unroll
        for (int step = 0; step < 2; ++step) {
          uint32_t v_fragment[4];
          load_swizzled<int8_t, 64, kHeadSlices>(
              v_fragment, smem_v, (d * 16) % 64, step * 32, d / 4);
#pragma unroll
          for (int qb = 0; qb < kQBlocksPerWarp; ++qb) {
            mma_u8s8s32(partial[qb], probability[qb][step], v_fragment);
          }
        }
#pragma unroll
        for (int qb = 0; qb < kQBlocksPerWarp; ++qb) {
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            out[d][qb][e] += float(partial[qb][e]) *
                (p_scale[qb][layout::accumulator_row_slot(e)] * v_scales[e & 3]);
          }
        }
      }
    } else {
#pragma unroll
      for (int kb = 0; kb < kKVBlocks; ++kb) {
        uint32_t probability[kQBlocksPerWarp][4];
#pragma unroll
        for (int qb = 0; qb < kQBlocksPerWarp; ++qb) {
          score_accumulator_to_pv_operand(
              probability[qb], score.fp32[qb][kb]);
        }
#pragma unroll
        for (int d = 0; d < kVBlocks; ++d) {
          int const column = d * 16;
          uint32_t v_fragment[4];
          load_swizzled_transposed<kBlockKV, kHeadSlices>(
              v_fragment, smem_v, kb * 16, column % 64, column / 64);
#pragma unroll
          for (int qb = 0; qb < kQBlocksPerWarp; ++qb) {
            mma_f16f16f32(out[d][qb], probability[qb], v_fragment);
          }
        }
      }
    }
  }

#pragma unroll
  for (int qb = 0; qb < kQBlocksPerWarp; ++qb) {
#pragma unroll
    for (int d = 0; d < kVBlocks; ++d) {
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        int const q_in_tile = warp * kWarpQ + qb * 16
            + layout::accumulator_row(lane, e);
        int const q_row = q0 + q_in_tile;
        if (q_row < qo_len) {
          int const column =
              d * 16 + layout::accumulator_column(lane, e);
          float const normalized = out[d][qb][e]
              / row_sum[qb][layout::accumulator_row_slot(e)];
          output[int64_t(batch) * stride_bz_o + int64_t(q_row) * stride_seq_o
                 + int64_t(head) * stride_h_o + column] =
              convert_output<Output>(normalized);
        }
      }
    }
  }

  if constexpr (ReturnLse) {
#pragma unroll
    for (int qb = 0; qb < kQBlocksPerWarp; ++qb) {
#pragma unroll
      for (int row_slot = 0; row_slot < 2; ++row_slot) {
        int const q_in_tile = warp * kWarpQ + qb * 16
            + layout::accumulator_row(lane, row_slot * 4);
        int const q_row = q0 + q_in_tile;
        if ((lane & 3) == 0 && q_row < qo_len) {
          lse[(int64_t(batch) * num_qo_heads + head) * qo_len + q_row] =
              row_max[qb][row_slot] + log2f(row_sum[qb][row_slot]);
        }
      }
    }
  }
}

template <int HeadDim, bool Causal, bool ReturnLse, typename Output, bool Int8PV>
void launch_ppu_attention(
    torch::Tensor const &query, torch::Tensor const &key,
    torch::Tensor const &value, torch::Tensor const &output,
    torch::Tensor const &lse, torch::Tensor const &query_scale,
    torch::Tensor const &key_scale, torch::Tensor const &value_scale, int qo_len, int kv_len,
    int num_qo_heads, int num_kv_heads,
    int stride_bz_q, int stride_seq_q, int stride_h_q,
    int stride_bz_k, int stride_seq_k, int stride_h_k,
    int stride_bz_v, int stride_seq_v, int stride_h_v,
    int stride_bz_o, int stride_seq_o, int stride_h_o,
    float softmax_scale) {
  size_t const smem_bytes =
      kBlockQ * HeadDim * sizeof(int8_t)
      + kBlockKV * HeadDim * sizeof(int8_t)
      + kBlockKV * HeadDim * (Int8PV ? sizeof(int8_t) : sizeof(cutlass::half_t))
      + (use_shared_value_scale<HeadDim, Causal, Int8PV> ? ValueScaleStage<HeadDim>::Bytes : 0);
  auto kernel = &qk_int8_pv_kernel<HeadDim, Causal, ReturnLse, Output, Int8PV>;
  // The opt-in is a per-function property.  Reissuing it for every attention
  // op adds host/runtime work but cannot change the already-loaded kernel.
  static hggcError_t const attribute_status = hggcFuncSetAttribute(
      reinterpret_cast<void const *>(kernel),
      hggcFuncAttributeMaxDynamicSharedMemorySize, int(smem_bytes));
  TORCH_CHECK(attribute_status == hggcSuccess,
              "PPU SageAttention dynamic-smem opt-in failed: ",
              hggcGetErrorString(attribute_status));
  dim3 const grid((qo_len + kBlockQ - 1) / kBlockQ,
                  num_qo_heads, query.size(0));
  dim3 const block(32, kWarps);
  kernel<<<grid, block, smem_bytes>>>(
      query.data_ptr<int8_t>(), key.data_ptr<int8_t>(),
      reinterpret_cast<std::conditional_t<Int8PV, int8_t, cutlass::half_t> const *>(value.data_ptr()),
      reinterpret_cast<Output *>(output.data_ptr()),
      ReturnLse ? lse.data_ptr<float>() : nullptr,
      query_scale.data_ptr<float>(), key_scale.data_ptr<float>(),
      Int8PV ? value_scale.data_ptr<float>() : nullptr,
      qo_len, kv_len, num_qo_heads, num_kv_heads,
      stride_bz_q, stride_seq_q, stride_h_q,
      stride_bz_k, stride_seq_k, stride_h_k,
      stride_bz_v, stride_seq_v, stride_h_v,
      stride_bz_o, stride_seq_o, stride_h_o, softmax_scale);
  auto const error = hggcGetLastError();
  TORCH_CHECK(error == hggcSuccess,
              "PPU SageAttention launch failed: ", hggcGetErrorString(error));
}

template <int HeadDim, typename Output, bool Int8PV>
void dispatch_flags(
    bool causal, bool return_lse,
    torch::Tensor const &query, torch::Tensor const &key,
    torch::Tensor const &value, torch::Tensor const &output,
    torch::Tensor const &lse, torch::Tensor const &query_scale,
    torch::Tensor const &key_scale, torch::Tensor const &value_scale, int qo_len, int kv_len,
    int num_qo_heads, int num_kv_heads,
    int stride_bz_q, int stride_seq_q, int stride_h_q,
    int stride_bz_k, int stride_seq_k, int stride_h_k,
    int stride_bz_v, int stride_seq_v, int stride_h_v,
    int stride_bz_o, int stride_seq_o, int stride_h_o,
    float softmax_scale) {
#define SAGEATTN_PPU_LAUNCH(CAUSAL, LSE)                                      \
  launch_ppu_attention<HeadDim, CAUSAL, LSE, Output, Int8PV>(                 \
      query, key, value, output, lse, query_scale, key_scale, value_scale,    \
      qo_len, kv_len, num_qo_heads, num_kv_heads,                            \
      stride_bz_q, stride_seq_q, stride_h_q,                                 \
      stride_bz_k, stride_seq_k, stride_h_k,                                 \
      stride_bz_v, stride_seq_v, stride_h_v,                                 \
      stride_bz_o, stride_seq_o, stride_h_o, softmax_scale)
  if (causal) {
    if (return_lse) SAGEATTN_PPU_LAUNCH(true, true);
    else SAGEATTN_PPU_LAUNCH(true, false);
  } else {
    if (return_lse) SAGEATTN_PPU_LAUNCH(false, true);
    else SAGEATTN_PPU_LAUNCH(false, false);
  }
#undef SAGEATTN_PPU_LAUNCH
}

}  // namespace detail
}  // namespace sageattention::ppu

template <bool Int8PV>
torch::Tensor attention_forward(
    torch::Tensor query, torch::Tensor key, torch::Tensor value,
    torch::Tensor output, torch::Tensor query_scale,
    torch::Tensor key_scale, torch::Tensor value_scale, int tensor_layout, int is_causal,
    int qk_quant_gran, float softmax_scale, int return_lse) {
  TORCH_CHECK(query.is_cuda() && key.is_cuda() && value.is_cuda() &&
                  output.is_cuda() && query_scale.is_cuda() && key_scale.is_cuda(),
              "PPU SageAttention tensors must be device tensors");
  TORCH_CHECK(query.get_device() == key.get_device() &&
                  query.get_device() == value.get_device() &&
                  query.get_device() == output.get_device() &&
                  query.get_device() == query_scale.get_device() &&
                  query.get_device() == key_scale.get_device(),
              "PPU SageAttention tensors must share one device");
  TORCH_CHECK(query.scalar_type() == torch::kInt8 &&
                  key.scalar_type() == torch::kInt8,
              "PPU SageAttention Q and K must be int8");
  TORCH_CHECK(value.scalar_type() == (Int8PV ? torch::kInt8 : torch::kHalf),
              "PPU SageAttention V dtype does not match the selected PV implementation");
  TORCH_CHECK(output.scalar_type() == torch::kHalf ||
                  output.scalar_type() == torch::kBFloat16,
              "PPU SageAttention output must be fp16 or bf16");
  TORCH_CHECK(query_scale.scalar_type() == torch::kFloat32 &&
                  key_scale.scalar_type() == torch::kFloat32,
              "PPU SageAttention scales must be fp32");
  TORCH_CHECK(query.dim() == 4 && key.dim() == 4 && value.dim() == (Int8PV ? 5 : 4) &&
                  output.dim() == 4 && query_scale.dim() == 3 &&
                  key_scale.dim() == 3,
              "PPU SageAttention requires rank-4 Q/K/O, rank-3 Q/K scales; V is rank-5 for INT8 PV, rank-4 for FP16 PV");
  TORCH_CHECK(query.is_contiguous() && key.is_contiguous() &&
                  query_scale.is_contiguous() && key_scale.is_contiguous(),
              "PPU SageAttention requires contiguous Q/K and scale tensors");
  TORCH_CHECK((Int8PV ? value.is_contiguous() : value.stride(3) == 1) && output.stride(3) == 1,
              "PPU SageAttention requires contiguous head dimension for V/O");
  TORCH_CHECK(qk_quant_gran == 2,
              "PPU SageAttention first shipping path supports per-warp Q and per-block K quantization only");
  TORCH_CHECK(tensor_layout == 0 || tensor_layout == 1,
              "tensor_layout must be 0 (NHD) or 1 (HND)");
  TORCH_CHECK(output.sizes() == query.sizes(),
              "PPU SageAttention output shape must equal query shape");

  int const batch = int(query.size(0));
  int const head_dim = int(query.size(3));
  int qo_len, kv_len, num_qo_heads, num_kv_heads;
  int stride_seq_q, stride_h_q, stride_seq_k, stride_h_k;
  int stride_seq_v, stride_h_v, stride_seq_o, stride_h_o;
  if (tensor_layout == 0) {
    qo_len = int(query.size(1));
    kv_len = int(key.size(1));
    num_qo_heads = int(query.size(2));
    num_kv_heads = int(key.size(2));
    stride_seq_q = int(query.stride(1));
    stride_h_q = int(query.stride(2));
    stride_seq_k = int(key.stride(1));
    stride_h_k = int(key.stride(2));
    stride_seq_v = int(value.stride(1));
    stride_h_v = int(value.stride(2));
    stride_seq_o = int(output.stride(1));
    stride_h_o = int(output.stride(2));
  } else {
    qo_len = int(query.size(2));
    kv_len = int(key.size(2));
    num_qo_heads = int(query.size(1));
    num_kv_heads = int(key.size(1));
    stride_seq_q = int(query.stride(2));
    stride_h_q = int(query.stride(1));
    stride_seq_k = int(key.stride(2));
    stride_h_k = int(key.stride(1));
    stride_seq_v = int(value.stride(2));
    stride_h_v = int(value.stride(1));
    stride_seq_o = int(output.stride(2));
    stride_h_o = int(output.stride(1));
  }

  TORCH_CHECK(head_dim == 64 || head_dim == 128,
              "PPU SageAttention supports head_dim 64 or 128, got ", head_dim);
  TORCH_CHECK(qo_len > 0 && kv_len > 0 && num_qo_heads > 0 &&
                  num_kv_heads > 0,
              "PPU SageAttention dimensions must be nonzero");
  TORCH_CHECK(!is_causal || qo_len == kv_len,
              "the first PPU causal path requires qo_len == kv_len");
  TORCH_CHECK(num_qo_heads % num_kv_heads == 0,
              "query heads must be divisible by key/value heads");
  TORCH_CHECK(key.size(0) == batch && value.size(0) == batch &&
                  output.size(0) == batch,
              "Q/K/V/O batch dimensions must match");
  if constexpr (Int8PV) {
    stride_seq_v = 0;
    stride_h_v = int(value.stride(1));
    int const blocks = (kv_len + 63) / 64;
    TORCH_CHECK(value.size(1) == num_kv_heads && value.size(2) == blocks &&
                    value.size(3) == head_dim && value.size(4) == 64,
                "INT8 V must have packed shape [B,Hkv,ceil(K/64),D,64]");
    TORCH_CHECK(value_scale.is_cuda() && value_scale.get_device() == query.get_device() &&
                    value_scale.scalar_type() == torch::kFloat32 && value_scale.is_contiguous() &&
                    value_scale.dim() == 4 && value_scale.size(0) == batch &&
                    value_scale.size(1) == num_kv_heads && value_scale.size(2) == blocks &&
                    value_scale.size(3) == head_dim,
                "INT8 V scales must have shape [B,Hkv,ceil(K/64),D]");
    TORCH_CHECK(key.size(3) == head_dim, "K head dimension mismatch");
  } else if (tensor_layout == 0) {
    TORCH_CHECK(int(value.size(1)) == kv_len &&
                    int(value.size(2)) == num_kv_heads &&
                    int(key.size(3)) == head_dim &&
                    int(value.size(3)) == head_dim,
                "NHD K/V shapes do not match the query contract");
  } else {
    TORCH_CHECK(int(value.size(1)) == num_kv_heads &&
                    int(value.size(2)) == kv_len &&
                    int(key.size(3)) == head_dim &&
                    int(value.size(3)) == head_dim,
                "HND K/V shapes do not match the query contract");
  }
  TORCH_CHECK(int(query_scale.size(0)) == batch &&
                  int(query_scale.size(1)) == num_qo_heads &&
                  int(query_scale.size(2)) ==
                      ((qo_len + sageattention::ppu::kBlockQ - 1) /
                       sageattention::ppu::kBlockQ) * sageattention::ppu::kWarps,
              "query_scale shape does not match the PPU per-warp contract");
  TORCH_CHECK(int(key_scale.size(0)) == batch &&
                  int(key_scale.size(1)) == num_kv_heads &&
                  int(key_scale.size(2)) ==
                      (kv_len + sageattention::ppu::kBlockKV - 1) /
                       sageattention::ppu::kBlockKV,
              "key_scale shape does not match the PPU per-block contract");

  torch::Tensor lse = return_lse
      ? torch::empty({batch, num_qo_heads, qo_len},
                     query.options().dtype(torch::kFloat32))
      : torch::empty({0}, query.options().dtype(torch::kFloat32));

#define SAGEATTN_PPU_DISPATCH(HEAD_DIM, OUTPUT_TYPE)                           \
  sageattention::ppu::detail::dispatch_flags<HEAD_DIM, OUTPUT_TYPE, Int8PV>(  \
      is_causal != 0, return_lse != 0,                                       \
      query, key, value, output, lse, query_scale, key_scale, value_scale,    \
      qo_len, kv_len, num_qo_heads, num_kv_heads,                            \
      int(query.stride(0)), stride_seq_q, stride_h_q,                         \
      int(key.stride(0)), stride_seq_k, stride_h_k,                           \
      int(value.stride(0)), stride_seq_v, stride_h_v,                         \
      int(output.stride(0)), stride_seq_o, stride_h_o, softmax_scale)
  if (output.scalar_type() == torch::kHalf) {
    if (head_dim == 64) SAGEATTN_PPU_DISPATCH(64, cutlass::half_t);
    else SAGEATTN_PPU_DISPATCH(128, cutlass::half_t);
  } else {
    if (head_dim == 64) SAGEATTN_PPU_DISPATCH(64, cutlass::bfloat16_t);
    else SAGEATTN_PPU_DISPATCH(128, cutlass::bfloat16_t);
  }
#undef SAGEATTN_PPU_DISPATCH
  return lse;
}

torch::Tensor qk_int8_sv_f16_accum_f32_attn_ppu(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o,
    torch::Tensor qs, torch::Tensor ks, int layout, int causal, int gran,
    float scale, int lse) {
  return attention_forward<false>(q, k, v, o, qs, ks, torch::Tensor{},
                                  layout, causal, gran, scale, lse);
}

torch::Tensor qk_int8_sv_int8_accum_f32_attn_ppu(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o,
    torch::Tensor qs, torch::Tensor ks, torch::Tensor vs, int layout, int causal,
    int gran, float scale, int lse) {
  return attention_forward<true>(q, k, v, o, qs, ks, vs,
                                 layout, causal, gran, scale, lse);
}
