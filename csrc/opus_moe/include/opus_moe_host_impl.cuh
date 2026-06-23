// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "opus_moe.h"

#include "aiter_hip_common.h"
#include "aiter_stream.h"

#include "opus_moe_arch.cuh"
#include "gfx950/opus_moe_arch_gfx950.cuh"
#include "gfx950/a8w4/opus_moe_stage2_a8w4_decode_dispatch_gfx950.cuh"
#include "opus_moe_common.cuh"

#include <cstdint>
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

namespace {
OpusMoeStage2Bf16Kernel opus_moe_stage2_bf16_tune_dispatch(int id)
{
    switch(opus_get_gfx_arch())
    {
    case OpusGfxArch::Gfx950:
        return opus_moe_stage2_bf16_tune_dispatch_gfx950(id);
    default:
    {
        const auto& info = opus_get_arch_info();
        AITER_CHECK(false,
                    "opus_moe: BF16 stage2 dispatch is only implemented for gfx950; "
                    "current device ",
                    info.dev,
                    " has gcnArchName='",
                    info.name,
                    "'.");
    }
    }
}

void check_contiguous_last_dim(const aiter_tensor_t& t, const char* name)
{
    AITER_CHECK(t.dim() > 0, name, " must have at least one dimension");
    AITER_CHECK(t.stride(-1) == 1, name, " last dimension must be contiguous");
}

void check_tensor(const aiter_tensor_t& t,
                  const char* name,
                  int expected_dim,
                  const char* expected_shape,
                  AiterDtype expected_dtype,
                  const char* expected_dtype_name)
{
    AITER_CHECK(t.dim() == expected_dim,
                name,
                " must be ",
                expected_dim,
                "-D ",
                expected_shape,
                ", got ndim=",
                t.dim());
    AITER_CHECK(t.dtype() == expected_dtype,
                name,
                " must be ",
                expected_dtype_name,
                ", got ",
                AiterDtype_to_str(t.dtype()));
    check_contiguous_last_dim(t, name);
}

void check_i32_metadata(const aiter_tensor_t& t, const char* name, bool non_empty)
{
    AITER_CHECK(t.dim() == 1, name, " must be 1-D, got ndim=", t.dim());
    AITER_CHECK(t.dtype() == AITER_DTYPE_i32,
                name,
                " must be int32, got ",
                AiterDtype_to_str(t.dtype()));
    AITER_CHECK(t.is_contiguous(), name, " must be contiguous");
    if(non_empty)
        AITER_CHECK(t.size(0) > 0, name, " must be non-empty");
}

void check_sorted_weights(const std::optional<aiter_tensor_t>& sorted_weights)
{
    if(!sorted_weights.has_value())
        return;
    AITER_CHECK(sorted_weights->dtype() == AITER_DTYPE_fp32,
                "sorted_weights must be fp32 when provided, got ",
                AiterDtype_to_str(sorted_weights->dtype()));
    AITER_CHECK(sorted_weights->is_contiguous(), "sorted_weights must be contiguous");
}

int select_bf16_kernel_id(int requested_kernel_id)
{
    const int selected_kernel_id =
        requested_kernel_id == opus_moe::kStage2KidAuto
            ? opus_moe::kStage2KidBf16GemmStyle256x256x64TokenSlotRouteOutNoOobNFast
            : requested_kernel_id;

    AITER_CHECK(opus_moe::stage2_kid_is_valid(selected_kernel_id),
                "opus_moe_stage2_route_reduce_fwd got unsupported kernel_id=",
                selected_kernel_id,
                " (",
                opus_moe::stage2_kid_name(selected_kernel_id),
                ")");
    return selected_kernel_id;
}

int select_a8w4_kernel_id(int requested_kernel_id,
                          int sorted_blocks,
                          int block_m)
{
    int selected_kernel_id = requested_kernel_id;
    if(selected_kernel_id == opus_moe::kStage2KidAuto)
    {
        if(block_m == opus_moe::kStage2A8W4DecodeBlockM32)
            selected_kernel_id =
                opus_moe::stage2_a8w4_bm32_uses_paced_route_blocks(sorted_blocks)
                    ? opus_moe::kStage2KidDsv4A8W4DecodeBm32DynamicPaced
                    : opus_moe::kStage2KidDsv4A8W4DecodeBm32Dynamic;
        else if(block_m == opus_moe::kStage2A8W4DecodeBlockM64)
            selected_kernel_id = opus_moe::kStage2KidDsv4A8W4DecodeBm64Dynamic;
        else if(block_m == opus_moe::kStage2A8W4DecodeBlockM16)
            selected_kernel_id = opus_moe::kStage2KidDsv4A8W4DecodeBm16Bn128Dynamic;
    }
    AITER_CHECK(opus_moe::stage2_a8w4_kid_is_valid(selected_kernel_id),
                "opus_moe_stage2_a8w4_decode_fwd got unsupported kernel_id=",
                selected_kernel_id,
                " (",
                opus_moe::stage2_a8w4_kid_name(selected_kernel_id),
                ")");

    if(selected_kernel_id == opus_moe::kStage2KidDsv4A8W4DecodeBm32Dynamic)
    {
        AITER_CHECK(block_m == opus_moe::kStage2A8W4DecodeBlockM32,
                    "kernel_id=",
                    selected_kernel_id,
                    " (",
                    opus_moe::stage2_a8w4_kid_name(selected_kernel_id),
                    ") requires block_m=",
                    opus_moe::kStage2A8W4DecodeBlockM32,
                    ", got ",
                    block_m);
    }
    else if(selected_kernel_id == opus_moe::kStage2KidDsv4A8W4DecodeBm32DynamicPaced)
    {
        AITER_CHECK(block_m == opus_moe::kStage2A8W4DecodeBlockM32,
                    "kernel_id=",
                    selected_kernel_id,
                    " (",
                    opus_moe::stage2_a8w4_kid_name(selected_kernel_id),
                    ") requires block_m=",
                    opus_moe::kStage2A8W4DecodeBlockM32,
                    ", got ",
                    block_m);
    }
    else if(selected_kernel_id == opus_moe::kStage2KidDsv4A8W4DecodeBm64Dynamic)
    {
        AITER_CHECK(block_m == opus_moe::kStage2A8W4DecodeBlockM64,
                    "kernel_id=",
                    selected_kernel_id,
                    " (",
                    opus_moe::stage2_a8w4_kid_name(selected_kernel_id),
                    ") requires block_m=",
                    opus_moe::kStage2A8W4DecodeBlockM64,
                    ", got ",
                    block_m);
    }
    else if(selected_kernel_id == opus_moe::kStage2KidDsv4A8W4DecodeBm16Bn128Dynamic)
    {
        AITER_CHECK(block_m == opus_moe::kStage2A8W4DecodeBlockM16,
                    "kernel_id=",
                    selected_kernel_id,
                    " (",
                    opus_moe::stage2_a8w4_kid_name(selected_kernel_id),
                    ") requires block_m=",
                    opus_moe::kStage2A8W4DecodeBlockM16,
                    ", got ",
                    block_m);
    }
    else if(selected_kernel_id ==
            opus_moe::kStage2KidDsv4A8W4P4RouteOut64x256x256Sbm128)
    {
        AITER_CHECK(block_m == opus_moe::kStage2A8W4DecodeBlockM64,
                    "kernel_id=",
                    selected_kernel_id,
                    " (",
                    opus_moe::stage2_a8w4_kid_name(selected_kernel_id),
                    ") requires block_m=",
                    opus_moe::kStage2A8W4DecodeBlockM64,
                    ", got ",
                    block_m);
    }

    return selected_kernel_id;
}

} // namespace

void opus_moe_stage2_route_reduce_fwd(aiter_tensor_t& inter_states,
                                      aiter_tensor_t& w2,
                                      aiter_tensor_t& sorted_token_ids,
                                      std::optional<aiter_tensor_t> sorted_weights,
                                      aiter_tensor_t& sorted_expert_ids,
                                      aiter_tensor_t& num_valid_ids,
                                      aiter_tensor_t& route_out,
                                      aiter_tensor_t& out,
                                      int block_m,
                                      int kernel_id)
{
    check_tensor(
        inter_states, "inter_states", 3, "[token, topk, inter_dim]", AITER_DTYPE_bf16, "bf16");
    check_tensor(w2, "w2", 3, "[expert, model_dim, inter_dim]", AITER_DTYPE_bf16, "bf16");
    check_tensor(out, "out", 2, "[output_rows, model_dim]", AITER_DTYPE_bf16, "bf16");
    check_tensor(route_out, "route_out", 2, "[route, model_dim]", AITER_DTYPE_bf16, "bf16");
    check_i32_metadata(sorted_token_ids, "sorted_token_ids", false);
    check_i32_metadata(sorted_expert_ids, "sorted_expert_ids", true);
    check_i32_metadata(num_valid_ids, "num_valid_ids", true);
    check_sorted_weights(sorted_weights);

    const int token_num = static_cast<int>(inter_states.size(0));
    const int actual_topk = static_cast<int>(inter_states.size(1));
    const int inter_dim = static_cast<int>(inter_states.size(2));
    const int num_experts = static_cast<int>(w2.size(0));
    const int model_dim = static_cast<int>(w2.size(1));
    const int route_rows = token_num * actual_topk;

    AITER_CHECK(w2.size(2) == inter_dim,
                "w2 inter_dim mismatch, got w2.size(2)=",
                w2.size(2),
                " inter_states.size(2)=",
                inter_dim);
    AITER_CHECK(out.size(0) == token_num && out.size(1) == model_dim,
                "out shape must be [token_num, model_dim]");
    AITER_CHECK(route_out.size(0) >= route_rows && route_out.size(1) == model_dim,
                "route_out shape must be at least [token_num * topk, model_dim]");
    AITER_CHECK(block_m > 0, "block_m must be positive");

    const int selected_kernel_id = select_bf16_kernel_id(kernel_id);
    if(token_num == 0 || model_dim == 0 || inter_dim == 0)
        return;

    opus_moe_stage2_kargs kargs{};
    kargs.inter_states = reinterpret_cast<const hip_bfloat16*>(inter_states.data_ptr());
    kargs.w2 = reinterpret_cast<const hip_bfloat16*>(w2.data_ptr());
    kargs.sorted_token_ids = reinterpret_cast<const int32_t*>(sorted_token_ids.data_ptr());
    kargs.sorted_weights = sorted_weights.has_value()
                               ? reinterpret_cast<const float*>(sorted_weights->data_ptr())
                               : nullptr;
    kargs.sorted_expert_ids =
        reinterpret_cast<const int32_t*>(sorted_expert_ids.data_ptr());
    kargs.num_valid_ids = reinterpret_cast<const int32_t*>(num_valid_ids.data_ptr());
    kargs.out_bf16 = reinterpret_cast<hip_bfloat16*>(out.data_ptr());
    kargs.route_out_bf16 = reinterpret_cast<hip_bfloat16*>(route_out.data_ptr());
    kargs.token_num = token_num;
    kargs.topk = actual_topk;
    kargs.num_experts = num_experts;
    kargs.model_dim = model_dim;
    kargs.inter_dim = inter_dim;
    kargs.block_m = block_m;
    kargs.stride_a_t = inter_states.stride(0);
    kargs.stride_a_k = inter_states.stride(1);
    kargs.stride_w_e = w2.stride(0);
    kargs.stride_w_h = w2.stride(1);
    kargs.stride_o_t = out.stride(0);
    kargs.stride_route_o_t = route_out.stride(0);

    HipDeviceGuard guard(inter_states.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    const int sorted_blocks = static_cast<int>(sorted_expert_ids.size(0));
    auto launcher = opus_moe_stage2_bf16_tune_dispatch(selected_kernel_id);
    launcher(kargs, sorted_blocks, stream);
    HIP_CALL_LAUNCH(hipGetLastError());

    opus_moe_stage2_reduce_token_slot_route_output_launch_gfx950(
        kargs, stream, kOpusMoeStage2RouteOutputReduceBf16BlockN);
    HIP_CALL_LAUNCH(hipGetLastError());
}

void opus_moe_stage2_a8w4_decode_fwd(
    aiter_tensor_t& inter_states,
    aiter_tensor_t& w2,
    aiter_tensor_t& a2_scale,
    aiter_tensor_t& w2_scale,
    aiter_tensor_t& sorted_token_ids,
    std::optional<aiter_tensor_t> sorted_weights,
    aiter_tensor_t& sorted_expert_ids,
    aiter_tensor_t& num_valid_ids,
    aiter_tensor_t& out,
    int block_m,
    int kernel_id,
    int inter_dim_pad)
{
    check_tensor(inter_states,
                 "inter_states",
                 3,
                 "[token, topk, packed_inter_dim]",
                 AITER_DTYPE_fp8,
                 "fp8");
    check_tensor(
        w2, "w2", 3, "[expert, model_dim, packed_inter_dim]", AITER_DTYPE_fp4x2, "fp4x2");
    check_tensor(
        a2_scale, "a2_scale", 2, "[route, scale_cols]", AITER_DTYPE_fp8_e8m0, "fp8_e8m0");
    check_tensor(w2_scale,
                 "w2_scale",
                 2,
                 "[expert * model_dim, scale_cols]",
                 AITER_DTYPE_fp8_e8m0,
                 "fp8_e8m0");
    check_tensor(out, "out", 2, "[token, model_dim]", AITER_DTYPE_bf16, "bf16");
    check_i32_metadata(sorted_token_ids, "sorted_token_ids", false);
    check_i32_metadata(sorted_expert_ids, "sorted_expert_ids", true);
    check_i32_metadata(num_valid_ids, "num_valid_ids", true);
    check_sorted_weights(sorted_weights);

    const int token_num = static_cast<int>(inter_states.size(0));
    const int actual_topk = static_cast<int>(inter_states.size(1));
    const int logical_inter_dim = static_cast<int>(inter_states.size(2));
    const int effective_inter_dim = logical_inter_dim - inter_dim_pad;
    const int num_experts = static_cast<int>(w2.size(0));
    const int model_dim = static_cast<int>(w2.size(1));
    const int packed_inter_dim = static_cast<int>(w2.size(2));
    const int sorted_blocks = static_cast<int>(sorted_expert_ids.size(0));
    const int scale_cols =
        logical_inter_dim / opus_moe::kStage2A8W4DecodeScaleGroupLogicalK;

    AITER_CHECK(block_m == opus_moe::kStage2A8W4DecodeBlockM16 ||
                    block_m == opus_moe::kStage2A8W4DecodeBlockM32 ||
                    block_m == opus_moe::kStage2A8W4DecodeBlockM64,
                "Opus A8W4 stage2 currently supports block_m=",
                opus_moe::kStage2A8W4DecodeBlockM16,
                ", ",
                opus_moe::kStage2A8W4DecodeBlockM32,
                ", or ",
                opus_moe::kStage2A8W4DecodeBlockM64,
                ", got ",
                block_m);
    AITER_CHECK(actual_topk == opus_moe::kStage2A8W4DecodeTopK,
                "Opus A8W4 stage2 currently supports topk=",
                opus_moe::kStage2A8W4DecodeTopK,
                ", got ",
                actual_topk);
    AITER_CHECK(model_dim == opus_moe::kStage2A8W4DecodeModelDim,
                "Opus A8W4 stage2 expects model_dim=",
                opus_moe::kStage2A8W4DecodeModelDim,
                ", got ",
                model_dim);
    AITER_CHECK(num_experts == opus_moe::kStage2A8W4DecodeExperts,
                "Opus A8W4 stage2 expects experts=",
                opus_moe::kStage2A8W4DecodeExperts,
                ", got ",
                num_experts);
    AITER_CHECK(logical_inter_dim == opus_moe::kStage2A8W4DecodeLogicalInterDim &&
                    inter_dim_pad == opus_moe::kStage2A8W4DecodeInterDimPad &&
                    effective_inter_dim ==
                        opus_moe::kStage2A8W4DecodeLogicalInterDim -
                            opus_moe::kStage2A8W4DecodeInterDimPad,
                "Opus A8W4 stage2 currently expects logical/effective inter_dim "
                "logical=",
                opus_moe::kStage2A8W4DecodeLogicalInterDim,
                " effective=",
                opus_moe::kStage2A8W4DecodeLogicalInterDim -
                    opus_moe::kStage2A8W4DecodeInterDimPad,
                " inter_dim_pad=",
                opus_moe::kStage2A8W4DecodeInterDimPad);
    AITER_CHECK(packed_inter_dim ==
                    logical_inter_dim / opus_moe::kStage2A8W4DecodeFp4ValuesPerByte,
                "w2 packed inter_dim mismatch, expected ",
                logical_inter_dim / opus_moe::kStage2A8W4DecodeFp4ValuesPerByte,
                ", got ",
                packed_inter_dim);
    AITER_CHECK(a2_scale.size(0) >= sorted_token_ids.size(0) &&
                    a2_scale.size(1) >= scale_cols,
                "a2_scale shape must cover sorted route rows and logical_inter_dim / ",
                opus_moe::kStage2A8W4DecodeScaleGroupLogicalK);
    AITER_CHECK(w2_scale.size(0) >= num_experts * model_dim &&
                    w2_scale.size(1) >= scale_cols,
                "w2_scale shape must be at least [expert * model_dim, logical_inter_dim / ",
                opus_moe::kStage2A8W4DecodeScaleGroupLogicalK,
                "]");

    const int selected_kernel_id =
        select_a8w4_kernel_id(kernel_id, sorted_blocks, block_m);
    const bool route_out_mode =
        opus_moe::stage2_a8w4_kid_uses_route_out(selected_kernel_id);
    const int expected_output_rows = route_out_mode ? token_num * actual_topk : token_num;
    AITER_CHECK(out.size(0) == expected_output_rows && out.size(1) == model_dim,
                "out shape must be [",
                expected_output_rows,
                ", ",
                model_dim,
                "] for kernel_id=",
                selected_kernel_id,
                " (",
                opus_moe::stage2_a8w4_kid_name(selected_kernel_id),
                ")");
    if(token_num == 0 || model_dim == 0 || logical_inter_dim == 0)
        return;

    opus_moe_stage2_a8w4_kargs kargs{};
    kargs.inter_states_fp8 = reinterpret_cast<const uint8_t*>(inter_states.data_ptr());
    kargs.w2_fp4 = reinterpret_cast<const uint8_t*>(w2.data_ptr());
    kargs.a2_scale_e8m0 = reinterpret_cast<const uint8_t*>(a2_scale.data_ptr());
    kargs.w2_scale_e8m0 = reinterpret_cast<const uint8_t*>(w2_scale.data_ptr());
    kargs.sorted_token_ids = reinterpret_cast<const int32_t*>(sorted_token_ids.data_ptr());
    kargs.sorted_weights = sorted_weights.has_value()
                               ? reinterpret_cast<const float*>(sorted_weights->data_ptr())
                               : nullptr;
    kargs.sorted_expert_ids =
        reinterpret_cast<const int32_t*>(sorted_expert_ids.data_ptr());
    kargs.num_valid_ids = reinterpret_cast<const int32_t*>(num_valid_ids.data_ptr());
    kargs.out_bf16 = reinterpret_cast<hip_bfloat16*>(out.data_ptr());
    kargs.stride_a_t = inter_states.stride(0);
    kargs.stride_a_k = inter_states.stride(1);
    kargs.stride_w_e = w2.stride(0);
    kargs.stride_a_scale_route = a2_scale.stride(0);
    kargs.stride_w_scale_row = w2_scale.stride(0);
    kargs.token_num = token_num;
    kargs.sorted_blocks = sorted_blocks;

    HipDeviceGuard guard(inter_states.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    opus_moe_stage2_a8w4_decode_dispatch_gfx950(selected_kernel_id, kargs, stream);
    HIP_CALL_LAUNCH(hipGetLastError());
}

void opus_moe_stage2_reduce_token_slot_route_output_fwd(aiter_tensor_t& route_out,
                                                        aiter_tensor_t& out,
                                                        int topk,
                                                        int block_n)
{
    check_tensor(route_out,
                 "route_out",
                 3,
                 "[token, topk, model_dim]",
                 AITER_DTYPE_bf16,
                 "bf16");
    check_tensor(out, "out", 2, "[token, model_dim]", AITER_DTYPE_bf16, "bf16");

    const int token_num = static_cast<int>(route_out.size(0));
    const int actual_topk = static_cast<int>(route_out.size(1));
    const int model_dim = static_cast<int>(route_out.size(2));
    AITER_CHECK(topk == actual_topk,
                "route_out topk mismatch, got route_out.size(1)=",
                actual_topk,
                " topk=",
                topk);
    AITER_CHECK(out.size(0) == token_num && out.size(1) == model_dim,
                "out shape must be [route_out.size(0), route_out.size(2)]");
    AITER_CHECK(route_out.stride(0) == route_out.stride(1) * actual_topk,
                "route_out must be contiguous over [token, topk] rows");
    if(token_num == 0 || topk == 0 || model_dim == 0)
        return;

    opus_moe_stage2_kargs kargs{};
    kargs.route_out_bf16 = reinterpret_cast<hip_bfloat16*>(route_out.data_ptr());
    kargs.out_bf16 = reinterpret_cast<hip_bfloat16*>(out.data_ptr());
    kargs.token_num = token_num;
    kargs.topk = actual_topk;
    kargs.model_dim = model_dim;
    kargs.stride_route_o_t = route_out.stride(1);
    kargs.stride_o_t = out.stride(0);

    HipDeviceGuard guard(route_out.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();
    opus_moe_stage2_reduce_token_slot_route_output_launch_gfx950(
        kargs, stream, block_n);
    HIP_CALL_LAUNCH(hipGetLastError());
}
