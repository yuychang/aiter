// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "../opus_moe_arch_gfx950.cuh"

using OpusMoeStage2A8W4DecodeBm32Dynamic =
    OpusMoeStage2A8W4DecodeShape<>;
using OpusMoeStage2A8W4DecodeBm32DynamicPaced =
    OpusMoeStage2A8W4DecodeShape<opus_moe::kStage2A8W4DecodeBlockM32,
                                 opus_moe::kStage2A8W4DecodeBlockN256,
                                 opus_moe::kStage2A8W4DecodeBlockM32,
                                 true,
                                 true>;
using OpusMoeStage2A8W4DecodeBm64Dynamic =
    OpusMoeStage2A8W4DecodeShape<opus_moe::kStage2A8W4DecodeBlockM64,
                                 opus_moe::kStage2A8W4DecodeBlockN256>;
using OpusMoeStage2A8W4DecodeBm16Bn128Dynamic =
    OpusMoeStage2A8W4DecodeShape<opus_moe::kStage2A8W4DecodeBlockM16,
                                 opus_moe::kStage2A8W4DecodeBlockN128,
                                 2 * opus_moe::kStage2A8W4DecodeBlockM16>;
using OpusMoeStage2A8W4P4RouteOut =
    OpusMoeStage2A8W4DecodeShape<opus_moe::kStage2A8W4DecodeBlockM64,
                                 opus_moe::kStage2A8W4DecodeBlockN256,
                                 opus_moe::kStage2A8W4P4SortBlockM,
                                 false>;

inline void opus_moe_stage2_a8w4_decode_dispatch_gfx950(
    int kid,
    const opus_moe_stage2_a8w4_kargs& kargs,
    hipStream_t stream)
{
    switch(kid)
    {
    case opus_moe::kStage2KidDsv4A8W4DecodeBm32Dynamic:
        return opus_moe_stage2_a8w4_decode_launch_gfx950<
            OpusMoeStage2A8W4DecodeBm32Dynamic>(kargs, stream);
    case opus_moe::kStage2KidDsv4A8W4DecodeBm32DynamicPaced:
        return opus_moe_stage2_a8w4_decode_launch_gfx950<
            OpusMoeStage2A8W4DecodeBm32DynamicPaced>(kargs, stream);
    case opus_moe::kStage2KidDsv4A8W4DecodeBm64Dynamic:
        return opus_moe_stage2_a8w4_decode_launch_gfx950<
            OpusMoeStage2A8W4DecodeBm64Dynamic>(kargs, stream);
    case opus_moe::kStage2KidDsv4A8W4DecodeBm16Bn128Dynamic:
        return opus_moe_stage2_a8w4_decode_launch_gfx950<
            OpusMoeStage2A8W4DecodeBm16Bn128Dynamic>(kargs, stream);
    case opus_moe::kStage2KidDsv4A8W4P4RouteOut64x256x256Sbm128:
        return opus_moe_stage2_a8w4_decode_launch_gfx950<
            OpusMoeStage2A8W4P4RouteOut>(kargs, stream);
    default: AITER_CHECK(false, "unreachable A8W4 kernel dispatch");
    }
}
