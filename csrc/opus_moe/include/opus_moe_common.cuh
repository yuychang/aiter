// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include <cstdint>
#include <hip/hip_bfloat16.h>

namespace opus_moe
{

constexpr int kStage2KidAuto = -1;
constexpr int kStage2KidBf16GemmStyle256x256x64TokenSlotRouteOutNoOobNFast = 1;

constexpr bool stage2_kid_is_valid(int kid)
{
    return kid == kStage2KidBf16GemmStyle256x256x64TokenSlotRouteOutNoOobNFast;
}

constexpr const char* stage2_kid_name(int kid)
{
    switch(kid)
    {
    case kStage2KidBf16GemmStyle256x256x64TokenSlotRouteOutNoOobNFast:
        return "bf16_gemmstyle256x256x64_token_slot_route_out_no_oob_nfast";
    default:
        return "unknown";
    }
}

} // namespace opus_moe

struct opus_moe_stage2_kargs
{
    const hip_bfloat16* __restrict__ inter_states;
    const hip_bfloat16* __restrict__ w2;
    const int32_t* __restrict__ sorted_token_ids;
    const float* __restrict__ sorted_weights;
    const int32_t* __restrict__ sorted_expert_ids;
    const int32_t* __restrict__ num_valid_ids;
    hip_bfloat16* __restrict__ out_bf16;
    hip_bfloat16* __restrict__ route_out_bf16;

    int token_num;
    int topk;
    int num_experts;
    int model_dim;
    int inter_dim;
    int block_m;

    int64_t stride_a_t;
    int64_t stride_a_k;
    int64_t stride_w_e;
    int64_t stride_w_h;
    int64_t stride_o_t;
    int64_t stride_route_o_t;
};

static __device__ __forceinline__ int opus_moe_token_id(int32_t packed)
{
    return packed & 0x00ffffff;
}

static __device__ __forceinline__ int opus_moe_topk_slot(int32_t packed)
{
    return static_cast<uint32_t>(packed) >> 24;
}
