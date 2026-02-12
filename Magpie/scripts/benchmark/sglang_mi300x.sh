#!/usr/bin/env bash
###############################################################################
# Magpie Generic SGLang Benchmark Script for MI300X
###############################################################################

source "$(dirname "$0")/benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

hf download "$MODEL" 2>/dev/null || true

# MI300X specific: Check MEC firmware version for RCCL memory reclaim
version=$(rocm-smi --showfw 2>/dev/null | grep MEC | head -n 1 | awk '{print $NF}')
if [[ "$version" == "" || $version -lt 177 ]]; then
  export HSA_NO_SCRATCH_RECLAIM=1
fi

# SGLang optimizations for MI300X
export SGLANG_USE_AITER=1
export SGLANG_AITER_MLA_PERSIST=1

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

set -x
python3 -m sglang.launch_server \
  --model-path=$MODEL \
  --host=0.0.0.0 \
  --port=$PORT \
  --trust-remote-code \
  --tensor-parallel-size=$TP \
  --mem-fraction-static=0.8 \
  --disable-radix-cache > $SERVER_LOG 2>&1 &

SERVER_PID=$!

# Wait for server to be ready
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

run_benchmark_serving \
    --model "$MODEL" \
    --port "$PORT" \
    --backend vllm \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts $(( $CONC * 10 )) \
    --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /workspace/

# After throughput, run evaluation only if RUN_EVAL is true
if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT" --concurrent-requests $CONC
    append_lm_eval_summary
fi
set +x




