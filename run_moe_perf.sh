export ROCM_PATH=/home/jli10004/workspace/rocm-toolkit-flydsl
export HSA_ENABLE_SDMA=0

AITER_GROUPED_DEBUG=1 AITER_LOG_MORE=1 AITER_FORCE_A8W4=1 AITER_USE_GROUPED_GEMM=1 AITER_FORCE_GFX1250=1 python op_tests/test_flydsl_grouped_gemm_gfx1250.py   --scenario bench   --data-format a8w4   --layout gguu   --experts 256   --tokens 4096   --topk 8   --model-dim 7168   --inter-dim 2048   --act silu --no-bias

