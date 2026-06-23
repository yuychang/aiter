// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "../../opus_moe_common.cuh"
#include "opus/opus.hpp"

template<int BlockM = opus_moe::kStage2A8W4DecodeDefaultBlockM,
         int BlockN = opus_moe::kStage2A8W4DecodeDefaultBlockN,
         int SortBlockM = BlockM,
         bool DirectAtomicOut = true,
         bool PaceRouteBlocksToPow2 = false>
struct OpusMoeStage2A8W4DecodeShape
{
    static constexpr bool IS_BM16_BN128 =
        BlockM == opus_moe::kStage2A8W4DecodeBlockM16 && BlockN == opus_moe::kStage2A8W4DecodeBlockN128;
    static constexpr bool IS_BM32_BN256 =
        BlockM == opus_moe::kStage2A8W4DecodeBlockM32 && BlockN == opus_moe::kStage2A8W4DecodeBlockN256;
    static constexpr bool IS_BM64_BN256 =
        BlockM == opus_moe::kStage2A8W4DecodeBlockM64 && BlockN == opus_moe::kStage2A8W4DecodeBlockN256;
    static constexpr int BLOCK_SIZE =
        IS_BM16_BN128 ? 2 * opus_moe::kStage2A8W4DecodeSmallCtaThreads
                      : opus_moe::kStage2A8W4DecodeDefaultCtaThreads;
    static constexpr int MIN_BLOCKS_PER_CU = IS_BM16_BN128 ? 4 : 2;
    static constexpr int B_M = BlockM, B_N = BlockN;
    static constexpr int B_K_LOGICAL = opus_moe::kStage2A8W4DecodeBKLogical;
    static constexpr int K_STEP_PACKED = B_K_LOGICAL / opus_moe::kStage2A8W4DecodeFp4ValuesPerByte;

    using D_A = uint8_t;

    static constexpr int BYTES_PER_VEC = opus_moe::kStage2A8W4DecodeVectorBytes;
    static constexpr int VEC_A = BYTES_PER_VEC / sizeof(D_A), B_BYTES_PER_VEC = BYTES_PER_VEC;

    static constexpr int CACHECTL_A = 0, CACHECTL_B = 0, CACHECTL_W_SCALE = 0;

    static constexpr int MMA_M = opus_moe::kStage2A8W4DecodeMfmaM;
    static constexpr int MMA_N = opus_moe::kStage2A8W4DecodeMfmaN;
    static constexpr int MMA_K = opus_moe::kStage2A8W4DecodeMfmaK;
    static constexpr int THREADS_K = opus::get_warp_size() / MMA_M;
    static constexpr int T_M = IS_BM32_BN256 ? 2 : 1;
    static constexpr int T_N = (BLOCK_SIZE / opus::get_warp_size()) / T_M;

    static constexpr int DECODE_TOPK = opus_moe::kStage2A8W4DecodeTopK;
    static constexpr int DECODE_H = opus_moe::kStage2A8W4DecodeModelDim;
    static constexpr int DECODE_LOGICAL_INTER_DIM = opus_moe::kStage2A8W4DecodeLogicalInterDim;
    static constexpr int DECODE_INTER_DIM_PAD = opus_moe::kStage2A8W4DecodeInterDimPad;
    static constexpr int DECODE_EFFECTIVE_INTER_DIM = DECODE_LOGICAL_INTER_DIM - DECODE_INTER_DIM_PAD;
    static constexpr int DECODE_NUM_EXPERTS = opus_moe::kStage2A8W4DecodeExperts;
    static constexpr int DECODE_COL_TILES = DECODE_H / B_N;
    static constexpr int SORT_BLOCK_M = SortBlockM;
    static constexpr int ROUTE_M_STRIDE = B_M;
    static constexpr bool DIRECT_ATOMIC_OUT = DirectAtomicOut;
    static constexpr bool DECODE_PACE_ROUTE_BLOCKS_TO_POW2 = PaceRouteBlocksToPow2;
    static constexpr int K_TILES = DECODE_EFFECTIVE_INTER_DIM / K_STEP_PACKED;
    static constexpr int A_LDS_STAGES = K_TILES, A_LDS_STAGE_ELEMS = B_M * K_STEP_PACKED;
    static constexpr int SCALE_GROUP_LOGICAL_K = opus_moe::kStage2A8W4DecodeScaleGroupLogicalK;
    static constexpr int DECODE_SCALE_GROUPS = DECODE_LOGICAL_INTER_DIM / SCALE_GROUP_LOGICAL_K;
    static constexpr int SCALE_GROUPS_PER_ROW_PACK =
        DECODE_SCALE_GROUPS / opus_moe::kStage2A8W4DecodeScaleGroupsPerRowPack;
    static constexpr int SCALE_WORDS_PER_GROUP_PACK = opus_moe::kStage2A8W4DecodeScaleWordsPerGroupPack;
    static constexpr int SCALE_WORDS_PER_ROW_PACK = SCALE_GROUPS_PER_ROW_PACK * SCALE_WORDS_PER_GROUP_PACK;
    static constexpr int SCALE_ROWS_PER_ROW_PACK = 2 * MMA_M;
    static constexpr int B_PAYLOAD_COL_STRIDE_BYTES = B_K_LOGICAL;
    static constexpr int B_PAYLOAD_K_STRIDE_BYTES = BYTES_PER_VEC / opus_moe::kStage2A8W4DecodeFp4ValuesPerByte;
    static constexpr int B_THREADGROUP_STRIDE_BYTES = THREADS_K * B_PAYLOAD_COL_STRIDE_BYTES;

    static constexpr int M_MFMA_PER_WAVE = B_M / (T_M * MMA_M);
    static constexpr int N_MFMA_PER_WAVE = B_N / (T_N * MMA_N);
    static constexpr int HALF_N_MFMA_PER_WAVE = N_MFMA_PER_WAVE / 2;
    static constexpr int A_LDS_BUFFER_LOAD_INSTS = IS_BM32_BN256 ? 2 : 2 * M_MFMA_PER_WAVE;
    static constexpr int C_LDS_N = B_N, VEC_C = opus_moe::kStage2A8W4DecodeCVec;
    static constexpr int ELEM_PER_ATOMIC = opus_moe::kStage2A8W4DecodeCValuesPerAtomic;

    using D_ACC = opus::fp32_t;
    using D_MFMA_A = opus::fp8_t;
    using D_MFMA_B = opus::fp4_t;

    static_assert(BYTES_PER_VEC == opus_moe::kStage2A8W4DecodeVectorBytes);
    static_assert(VEC_A * static_cast<int>(sizeof(D_A)) == BYTES_PER_VEC);
    static_assert(BLOCK_SIZE / opus::get_warp_size() == T_M * T_N);
    static_assert(THREADS_K == K_STEP_PACKED / (2 * VEC_A));
    static_assert(THREADS_K == K_STEP_PACKED / (2 * B_BYTES_PER_VEC));
    static_assert((B_PAYLOAD_COL_STRIDE_BYTES & (B_PAYLOAD_COL_STRIDE_BYTES - 1)) == 0);
    static_assert((B_THREADGROUP_STRIDE_BYTES & (B_THREADGROUP_STRIDE_BYTES - 1)) == 0);
    static_assert(B_M % (T_M * MMA_M) == 0);
    static_assert(B_N % (T_N * MMA_N) == 0);
    static_assert(DECODE_H % B_N == 0);
    static_assert(SORT_BLOCK_M > 0);
    static_assert(SORT_BLOCK_M % B_M == 0);
    static_assert(DECODE_EFFECTIVE_INTER_DIM == K_TILES * K_STEP_PACKED);
    static_assert(K_TILES == 3);
    static_assert(DECODE_LOGICAL_INTER_DIM % SCALE_GROUP_LOGICAL_K == 0);
    static_assert(DECODE_SCALE_GROUPS % opus_moe::kStage2A8W4DecodeScaleGroupsPerRowPack == 0);
};
