// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include <cstdint>
#include <hip/hip_bfloat16.h>

namespace opus_moe
{

constexpr int kStage2KidAuto = -1;
constexpr int kStage2KidBf16GemmStyle256x256x64TokenSlotRouteOutNoOobNFast = 1;
constexpr int kStage2KidDsv4A8W4DecodeBm32Dynamic = 2010;
constexpr int kStage2KidDsv4A8W4DecodeBm32DynamicPaced = 2011;
constexpr int kStage2KidDsv4A8W4DecodeBm64Dynamic = 2020;
constexpr int kStage2KidDsv4A8W4DecodeBm16Bn128Dynamic = 2030;
constexpr int kStage2KidDsv4A8W4P4RouteOut64x256x256Sbm128 = 2040;

constexpr int kStage2A8W4DecodeTopK = 6;
constexpr int kStage2A8W4DecodeModelDim = 7168;
constexpr int kStage2A8W4DecodeLogicalInterDim = 512;
constexpr int kStage2A8W4DecodeInterDimPad = 128;
constexpr int kStage2A8W4DecodeExperts = 384;
constexpr int kStage2A8W4P4SortBlockM = 128;
constexpr int kStage2A8W4DecodeBlockM16 = 16;
constexpr int kStage2A8W4DecodeBlockM32 = 32;
constexpr int kStage2A8W4DecodeBlockM64 = 64;
constexpr int kStage2A8W4DecodeBlockN128 = 128;
constexpr int kStage2A8W4DecodeBlockN256 = 256;
constexpr int kStage2A8W4DecodeDefaultBlockM = kStage2A8W4DecodeBlockM32;
constexpr int kStage2A8W4DecodeDefaultBlockN = kStage2A8W4DecodeBlockN256;
constexpr int kStage2A8W4DecodeDefaultCtaThreads = 256;
constexpr int kStage2A8W4DecodeSmallCtaThreads = 64;
constexpr int kStage2A8W4DecodeBKLogical = 256;
constexpr int kStage2A8W4DecodeMfmaM = 16;
constexpr int kStage2A8W4DecodeMfmaN = 16;
constexpr int kStage2A8W4DecodeMfmaK = 128;
constexpr int kStage2A8W4DecodeFp4ValuesPerByte = 2;
constexpr int kStage2A8W4DecodeVectorBytes = 16;
constexpr int kStage2MXFP4ScaleGroupLogicalK = 32;
constexpr int kStage2A8W4DecodeScaleGroupLogicalK = kStage2MXFP4ScaleGroupLogicalK;
constexpr int kStage2A8W4DecodeScaleGroupsPerRowPack = 8;
constexpr int kStage2A8W4DecodeScaleWordsPerGroupPack = 64;
constexpr int kStage2A8W4DecodeCVec = 4;
constexpr int kStage2A8W4DecodeCValuesPerAtomic = 2;
// BM32 dynamic decode uses the non-paced implementation through the largest
// measured low-grid bucket; larger BM32 buckets select the pow2-paced kid.
constexpr int kStage2A8W4DecodeBm32MaxUnpacedRouteBlocks = 576;

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

constexpr bool stage2_a8w4_kid_is_valid(int kid)
{
    return kid == kStage2KidDsv4A8W4DecodeBm32Dynamic ||
           kid == kStage2KidDsv4A8W4DecodeBm32DynamicPaced ||
           kid == kStage2KidDsv4A8W4DecodeBm64Dynamic ||
           kid == kStage2KidDsv4A8W4DecodeBm16Bn128Dynamic ||
           kid == kStage2KidDsv4A8W4P4RouteOut64x256x256Sbm128;
}

constexpr bool stage2_a8w4_kid_uses_route_out(int kid)
{
    return kid == kStage2KidDsv4A8W4P4RouteOut64x256x256Sbm128;
}

constexpr bool stage2_a8w4_bm32_uses_paced_route_blocks(int sorted_blocks)
{
    return sorted_blocks > kStage2A8W4DecodeBm32MaxUnpacedRouteBlocks;
}

constexpr const char* stage2_a8w4_kid_name(int kid)
{
    switch(kid)
    {
    case kStage2KidDsv4A8W4DecodeBm32Dynamic:
        return "dsv4_a8w4_decode_bm32_dynamic";
    case kStage2KidDsv4A8W4DecodeBm32DynamicPaced:
        return "dsv4_a8w4_decode_bm32_dynamic_paced";
    case kStage2KidDsv4A8W4DecodeBm64Dynamic:
        return "dsv4_a8w4_decode_bm64_dynamic";
    case kStage2KidDsv4A8W4DecodeBm16Bn128Dynamic:
        return "dsv4_a8w4_decode_bm16_bn128_dynamic";
    case kStage2KidDsv4A8W4P4RouteOut64x256x256Sbm128:
        return "dsv4_a8w4_p4_route_out64x256x256_sbm128";
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

struct opus_moe_stage2_a8w4_kargs
{
    const uint8_t* __restrict__ inter_states_fp8;
    const uint8_t* __restrict__ w2_fp4;
    const uint8_t* __restrict__ a2_scale_e8m0;
    const uint8_t* __restrict__ w2_scale_e8m0;
    const int32_t* __restrict__ sorted_token_ids;
    const float* __restrict__ sorted_weights;
    const int32_t* __restrict__ sorted_expert_ids;
    const int32_t* __restrict__ num_valid_ids;
    hip_bfloat16* __restrict__ out_bf16;

    int64_t stride_a_t;
    int64_t stride_a_k;
    int64_t stride_w_e;
    int64_t stride_a_scale_route;
    int64_t stride_w_scale_row;

    int token_num;
    int sorted_blocks;
};

static __device__ __forceinline__ int opus_moe_token_id(int32_t packed)
{
    return packed & 0x00ffffff;
}

static __device__ __forceinline__ int opus_moe_topk_slot(int32_t packed)
{
    return static_cast<uint32_t>(packed) >> 24;
}
