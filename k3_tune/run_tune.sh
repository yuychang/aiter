#!/usr/bin/env bash
# Tune the Kimi-K3 prefill GEMMs on all 8 GPUs and print a before/after table.
#
#   k3_tune/run_tune.sh                 # everything, 8 GPUs
#   k3_tune/run_tune.sh --dry-run       # print the commands, touch nothing
#   k3_tune/run_tune.sh --stages precompile,tune_a8w8   # subset (comma list)
#   k3_tune/run_tune.sh --fresh         # drop previous tuned CSVs first
#   k3_tune/run_tune.sh --force         # skip the "GPUs are idle" check
#
# Stages (in order): prep precompile bench_before tune_a8w8 tune_bf16
#                    bench_after summary
#
# The tuners are imported from the serving tree (kimi_k3/aiter, via env.sh);
# nothing is written there. Results go to this worktree:
#   aiter/configs/model_configs/a8w8_bpreshuffle_tuned_gemm_kimik3_prefill.csv
#   aiter/configs/model_configs/kimik3_prefill_bf16_tuned_gemm.csv
# Scratch (JIT, FlyDSL cache, logs, summary.md) goes to k3_tune/work/.
#
# Serving picks the rows up with no env var: with AITER_CONFIG_GEMM_* unset,
# aiter globs model_configs/*a8w8_bpreshuffle_tuned_gemm* and
# model_configs/*bf16_tuned_gemm* and merges them into the default tables.
# Commit the two tuned CSVs to the aiter branch the server runs.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WT="$(cd "${HERE}/.." && pwd)"
MC="${WT}/aiter/configs/model_configs"
WORK="${K3_TUNE_WORK:-${HERE}/work}"
ENV_SH="${K3_ENV_SH:-/dockerx/sglang-quark-opts/sglang-gpu-perf/kimi_k3/env.sh}"
NGPU="${K3_TUNE_NGPU:-8}"

A8_UNTUNED="${MC}/a8w8_bpreshuffle_untuned_gemm_kimik3_prefill.csv"
A8_TUNED="${MC}/a8w8_bpreshuffle_tuned_gemm_kimik3_prefill.csv"
BF_UNTUNED="${MC}/kimik3_prefill_bf16_untuned_gemm.csv"
BF_TUNED="${MC}/kimik3_prefill_bf16_tuned_gemm.csv"

ALL_STAGES="prep,precompile,bench_before,tune_a8w8,tune_bf16,bench_after,summary"
STAGES="${ALL_STAGES}"
DRY=0 FORCE=0 FRESH=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY=1 ;;
        --force) FORCE=1 ;;
        --fresh) FRESH=1 ;;
        --stages) STAGES="$2"; shift ;;
        -h | --help) sed -n '2,/^set -uo/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
    shift
done
want() { [[ ",${STAGES}," == *",$1,"* ]]; }

run() {
    echo "+ $*"
    [[ ${DRY} == 1 ]] || "$@"
}

mkdir -p "${WORK}/logs"
[[ ${DRY} == 1 ]] || : >"${WORK}/status.txt"

if [[ ${DRY} == 0 ]]; then
    set +u
    # shellcheck disable=SC1090
    source "${ENV_SH}" 2>"${WORK}/logs/env.log" || { echo "sourcing ${ENV_SH} failed; see ${WORK}/logs/env.log" >&2; exit 1; }
    set -u
else
    AITER_ROOT="${AITER_ROOT:-<serving aiter>}"
    AITER_JIT_DIR="${AITER_JIT_DIR:-<serving jit dir>}"
    echo "# dry run: env.sh not sourced"
fi

SERVE_JIT="${AITER_JIT_DIR}"
export TMPDIR="${WORK}/tmp"
export AITER_JIT_DIR="${WORK}/jit"
export FLYDSL_RUNTIME_CACHE_DIR="${WORK}/flydsl_cache"
export PYTHONDONTWRITEBYTECODE=1
unset AITER_REBUILD AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE AITER_CONFIG_GEMM_BF16
for v in $(compgen -e | grep '^AITER_CONFIG_' || true); do unset "$v"; done

A8_TUNE_PY="${AITER_ROOT}/csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_tune.py"
BF_TUNE_PY="${AITER_ROOT}/csrc/gemm_a16w16/gemm_a16w16_tune.py"
ALL_GPUS="$(seq -s, 0 $((NGPU - 1)))"

stage_note() { echo; echo "=== [$(date +%H:%M:%S)] $1 ==="; }
record() { [[ ${DRY} == 1 ]] || echo "$1 rc=$2 t=$(date +%s)" >>"${WORK}/status.txt"; }

gpus_idle() {
    local busy
    busy="$(rocm-smi --showmemuse 2>/dev/null | sed -n 's/.*VRAM%): *\([0-9]*\).*/\1/p' | awk '$1 > 5' | wc -l)"
    [[ "${busy}" == 0 ]]
}

# ---------------------------------------------------------------- prep
if want prep; then
    stage_note prep
    if [[ ${DRY} == 0 ]]; then
        if [[ ${FRESH} == 1 ]]; then rm -f "${A8_TUNED}" "${BF_TUNED}" "${WORK}"/*.csv; fi
        mkdir -p "${TMPDIR}" "${AITER_JIT_DIR}" "${FLYDSL_RUNTIME_CACHE_DIR}"
        # Seed with the serving build so only the *_tune modules compile.
        if [[ -d "${SERVE_JIT}" ]]; then
            cp -an "${SERVE_JIT}"/*.so "${AITER_JIT_DIR}/" 2>/dev/null || true
            mkdir -p "${AITER_JIT_DIR}/build"
        fi
        for f in "${A8_UNTUNED}" "${BF_UNTUNED}" "${A8_TUNE_PY}" "${BF_TUNE_PY}"; do
            [[ -f "${f}" ]] || { echo "missing ${f}" >&2; exit 1; }
        done
        (cd "${HERE}" && python gen_shapes.py --check) || { echo "untuned CSVs are stale vs gen_shapes.py" >&2; exit 1; }
        if [[ ${FORCE} == 0 ]] && ! gpus_idle; then
            echo "GPUs are not idle (VRAM > 5%); refusing to start. Use --force to override." >&2
            rocm-smi --showmemuse 2>/dev/null | grep -i vram >&2
            exit 3
        fi
        python -c "import torch; print('torch', torch.__version__, 'devices', torch.cuda.device_count())" || exit 1
    else
        echo "+ (prep) seed ${WORK}/jit from the serving JIT dir; gen_shapes.py --check; GPU idle check"
    fi
fi

# ---------------------------------------------------------------- precompile (CPU)
if want precompile; then
    stage_note "precompile FlyDSL + CK tune modules (CPU only)"
    PRE="${WORK}/precompile"
    run mkdir -p "${PRE}"
    run bash -c "cd '${HERE}' && python gen_precompile_csv.py a8w8 '${A8_UNTUNED}' '${PRE}/a8w8_flydsl.csv'" >"${WORK}/logs/precompile_gen.log" 2>&1
    record precompile_gen $?
    # The CK / cktile tune modules otherwise build inside an mp worker, under
    # the per-task watchdog.
    if [[ ${DRY} == 1 ]]; then
        echo "+ python ${HERE}/prebuild_tune_modules.py module_gemm_a8w8_bpreshuffle_tune module_gemm_a8w8_bpreshuffle_cktile_tune  (background)"
        echo "+ python -m aiter.aot.flydsl.gemm --csv ${PRE}/a8w8_flydsl.csv"
    else
        python "${HERE}/prebuild_tune_modules.py" module_gemm_a8w8_bpreshuffle_tune module_gemm_a8w8_bpreshuffle_cktile_tune \
            >"${WORK}/logs/prebuild_ck.log" 2>&1 &
        CK_PID=$!
        python -m aiter.aot.flydsl.gemm --csv "${PRE}/a8w8_flydsl.csv" >"${WORK}/logs/precompile_flydsl_a8w8.log" 2>&1
        record precompile_flydsl_a8w8 $?
        wait "${CK_PID}"
        rc=$?
        record prebuild_ck $rc
        if [[ ${rc} != 0 ]]; then
            echo "WARNING: CK tune-module prebuild failed (see ${WORK}/logs/prebuild_ck.log); tuning flydsl only" >&2
            A8_LIBTYPES="flydsl"
        fi
    fi
fi
A8_LIBTYPES="${A8_LIBTYPES:-flydsl,ck,cktile}"

# ---------------------------------------------------------------- bench_before
# Production op with the shipped configs (what serving runs now). One GPU per
# tuner, run side by side.
if want bench_before; then
    stage_note "bench_before: production op, default configs"
    if [[ ${DRY} == 1 ]]; then
        echo "+ HIP_VISIBLE_DEVICES=0 python ${A8_TUNE_PY} -i ${A8_UNTUNED} -o ${WORK}/dummy_a8.csv --run_config > ${WORK}/logs/before_a8w8.log"
        echo "+ HIP_VISIBLE_DEVICES=1 python ${BF_TUNE_PY} -i ${BF_UNTUNED} -o ${WORK}/dummy_bf.csv --run_config > ${WORK}/logs/before_bf16.log"
    else
        HIP_VISIBLE_DEVICES=0 python "${A8_TUNE_PY}" -i "${A8_UNTUNED}" -o "${WORK}/dummy_a8.csv" --run_config \
            >"${WORK}/logs/before_a8w8.log" 2>&1 &
        p1=$!
        HIP_VISIBLE_DEVICES=1 python "${BF_TUNE_PY}" -i "${BF_UNTUNED}" -o "${WORK}/dummy_bf.csv" --run_config \
            >"${WORK}/logs/before_bf16.log" 2>&1 &
        p2=$!
        wait "${p1}"; record before_a8w8 $?
        wait "${p2}"; record before_bf16 $?
    fi
fi

# ---------------------------------------------------------------- tune
if want tune_a8w8; then
    stage_note "tune a8w8 bpreshuffle (${A8_LIBTYPES}) on GPUs ${ALL_GPUS}"
    if [[ ${DRY} == 1 ]]; then
        echo "+ HIP_VISIBLE_DEVICES=${ALL_GPUS} python ${A8_TUNE_PY} -i ${A8_UNTUNED} -o ${A8_TUNED} -o2 ${WORK}/profile_a8w8.csv --libtype ${A8_LIBTYPES} --mp ${NGPU} --batch 6 --timeout 1800 --errRatio 0.05"
    else
        HIP_VISIBLE_DEVICES="${ALL_GPUS}" python "${A8_TUNE_PY}" \
            -i "${A8_UNTUNED}" -o "${A8_TUNED}" -o2 "${WORK}/profile_a8w8.csv" \
            --libtype "${A8_LIBTYPES}" --mp "${NGPU}" --batch 6 --timeout 1800 \
            >"${WORK}/logs/tune_a8w8.log" 2>&1
        record tune_a8w8 $?
        tail -n 5 "${WORK}/logs/tune_a8w8.log"
    fi
fi

if want tune_bf16; then
    stage_note "tune bf16 (all backends) on GPUs ${ALL_GPUS}"
    if [[ ${DRY} == 0 ]]; then
        # Needs a device to enumerate FlyDSL hgemm candidates.
        HIP_VISIBLE_DEVICES=0 python "${HERE}/gen_precompile_csv.py" bf16 "${BF_UNTUNED}" "${WORK}/precompile/bf16_flydsl.csv" \
            >"${WORK}/logs/precompile_gen_bf16.log" 2>&1 &&
            python -m aiter.aot.flydsl.gemm --csv "${WORK}/precompile/bf16_flydsl.csv" \
                >"${WORK}/logs/precompile_flydsl_bf16.log" 2>&1
        record precompile_flydsl_bf16 $?
    else
        echo "+ (bf16 FlyDSL AOT precompile via gen_precompile_csv.py bf16)"
    fi
    if [[ ${DRY} == 1 ]]; then
        echo "+ HIP_VISIBLE_DEVICES=${ALL_GPUS} python ${BF_TUNE_PY} -i ${BF_UNTUNED} -o ${BF_TUNED} -o2 ${WORK}/profile_bf16.csv --libtype all --mp ${NGPU}"
    else
        HIP_VISIBLE_DEVICES="${ALL_GPUS}" python "${BF_TUNE_PY}" \
            -i "${BF_UNTUNED}" -o "${BF_TUNED}" -o2 "${WORK}/profile_bf16.csv" \
            --libtype all --mp "${NGPU}" \
            >"${WORK}/logs/tune_bf16.log" 2>&1
        record tune_bf16 $?
        tail -n 5 "${WORK}/logs/tune_bf16.log"
    fi
fi

# ---------------------------------------------------------------- bench_after
if want bench_after; then
    stage_note "bench_after: production op with the tuned CSVs"
    if [[ ${DRY} == 1 ]]; then
        echo "+ HIP_VISIBLE_DEVICES=0 python ${A8_TUNE_PY} -i ${A8_UNTUNED} -o ${A8_TUNED} --run_config ${A8_TUNED} > ${WORK}/logs/after_a8w8.log"
        echo "+ HIP_VISIBLE_DEVICES=1 python ${BF_TUNE_PY} -i ${BF_UNTUNED} -o ${BF_TUNED} --run_config ${BF_TUNED} > ${WORK}/logs/after_bf16.log"
    else
        pids=()
        if [[ -f "${A8_TUNED}" ]]; then
            HIP_VISIBLE_DEVICES=0 python "${A8_TUNE_PY}" -i "${A8_UNTUNED}" -o "${A8_TUNED}" --run_config "${A8_TUNED}" \
                >"${WORK}/logs/after_a8w8.log" 2>&1 &
            pids+=($!)
        fi
        if [[ -f "${BF_TUNED}" ]]; then
            HIP_VISIBLE_DEVICES=1 python "${BF_TUNE_PY}" -i "${BF_UNTUNED}" -o "${BF_TUNED}" --run_config "${BF_TUNED}" \
                >"${WORK}/logs/after_bf16.log" 2>&1 &
            pids+=($!)
        fi
        for p in "${pids[@]}"; do wait "${p}"; record bench_after $?; done
    fi
fi

# ---------------------------------------------------------------- summary
if want summary; then
    stage_note "summary"
    if [[ ${DRY} == 1 ]]; then
        echo "+ python ${HERE}/summarize.py --kind a8w8|bf16 ... --md ${WORK}/summary.md"
    else
        : >"${WORK}/summary.md"
        (
            cd "${HERE}" || exit 1
            python summarize.py --kind a8w8 --untuned "${A8_UNTUNED}" --tuned "${A8_TUNED}" \
                --before "${WORK}/logs/before_a8w8.log" --after "${WORK}/logs/after_a8w8.log" --md "${WORK}/summary.md"
            echo
            python summarize.py --kind bf16 --untuned "${BF_UNTUNED}" --tuned "${BF_TUNED}" \
                --before "${WORK}/logs/before_bf16.log" --after "${WORK}/logs/after_bf16.log" --md "${WORK}/summary.md"
        )
        echo
        echo "summary: ${WORK}/summary.md"
        echo "tuned CSVs:"
        ls -l "${A8_TUNED}" "${BF_TUNED}" 2>/dev/null
        echo "stage status: ${WORK}/status.txt (rc != 0 means look in ${WORK}/logs)"
        echo "Next: git add the two tuned CSVs, commit on the aiter branch the server runs."
    fi
fi
