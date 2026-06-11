#!/bin/bash
# Run fused MOE route+quant+scatter tests on gfx1250
#
# Usage:
#   ./run_moe_test.sh              # run all cases
#   ./run_moe_test.sh -k "fp4"     # filter by pytest -k expression
#   ./run_moe_test.sh -x           # stop on first failure
#   ./run_moe_test.sh -k "256-4-1-8-fp4-4"  # single case

export ROCM_PATH=/home/jli10004/workspace/rocm-toolkit-flydsl
export AITER_USE_SYSTEM_TRITON=1

exec python -m pytest op_tests/test_moe_fused_route_quant_scatter.py -v "$@"
