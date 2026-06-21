// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

#include "opus_moe.h"

#include "aiter_hip_common.h"
#include "aiter_stream.h"

#include "opus_moe_arch.cuh"
#include "gfx950/opus_moe_arch_gfx950.cuh"
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
    AITER_CHECK(inter_states.dim() == 3,
                "inter_states must be 3-D [token, topk, inter_dim], got ndim=",
                inter_states.dim());
    AITER_CHECK(w2.dim() == 3, "w2 must be 3-D [expert, model_dim, inter_dim]");
    AITER_CHECK(out.dim() == 2, "out must be 2-D [token, model_dim]");
    AITER_CHECK(route_out.dim() == 2, "route_out must be 2-D [route, model_dim]");
    AITER_CHECK(sorted_token_ids.dim() == 1, "sorted_token_ids must be 1-D");
    AITER_CHECK(sorted_expert_ids.dim() == 1, "sorted_expert_ids must be 1-D");
    AITER_CHECK(num_valid_ids.dim() == 1 && num_valid_ids.size(0) >= 1,
                "num_valid_ids must be 1-D with at least one element");

    AITER_CHECK(inter_states.dtype() == AITER_DTYPE_bf16,
                "inter_states must be bf16 for opus_moe_stage2_route_reduce_fwd");
    AITER_CHECK(w2.dtype() == AITER_DTYPE_bf16,
                "w2 must be bf16 for opus_moe_stage2_route_reduce_fwd");
    AITER_CHECK(out.dtype() == AITER_DTYPE_bf16,
                "out must be bf16 for opus_moe_stage2_route_reduce_fwd");
    AITER_CHECK(route_out.dtype() == AITER_DTYPE_bf16,
                "route_out must be bf16 for opus_moe_stage2_route_reduce_fwd");
    AITER_CHECK(sorted_token_ids.dtype() == AITER_DTYPE_i32,
                "sorted_token_ids must be int32");
    AITER_CHECK(sorted_expert_ids.dtype() == AITER_DTYPE_i32,
                "sorted_expert_ids must be int32");
    AITER_CHECK(num_valid_ids.dtype() == AITER_DTYPE_i32,
                "num_valid_ids must be int32");
    if(sorted_weights.has_value())
    {
        AITER_CHECK(sorted_weights->dtype() == AITER_DTYPE_fp32,
                    "sorted_weights must be fp32 when provided");
        AITER_CHECK(sorted_weights->is_contiguous(), "sorted_weights must be contiguous");
    }

    check_contiguous_last_dim(inter_states, "inter_states");
    check_contiguous_last_dim(w2, "w2");
    check_contiguous_last_dim(out, "out");
    check_contiguous_last_dim(route_out, "route_out");
    AITER_CHECK(sorted_token_ids.is_contiguous(), "sorted_token_ids must be contiguous");
    AITER_CHECK(sorted_expert_ids.is_contiguous(), "sorted_expert_ids must be contiguous");
    AITER_CHECK(num_valid_ids.is_contiguous(), "num_valid_ids must be contiguous");

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
    AITER_CHECK(sorted_expert_ids.size(0) > 0, "sorted_expert_ids must be non-empty");

    const int selected_kernel_id =
        kernel_id == opus_moe::kStage2KidAuto
            ? opus_moe::kStage2KidBf16GemmStyle256x256x64TokenSlotRouteOutNoOobNFast
            : kernel_id;
    AITER_CHECK(opus_moe::stage2_kid_is_valid(selected_kernel_id),
                "opus_moe_stage2_route_reduce_fwd got unsupported kernel_id=",
                selected_kernel_id,
                " (",
                opus_moe::stage2_kid_name(selected_kernel_id),
                ")");
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

    opus_moe_stage2_reduce_token_slot_route_output_launch_gfx950(kargs, stream);
    HIP_CALL_LAUNCH(hipGetLastError());
}
