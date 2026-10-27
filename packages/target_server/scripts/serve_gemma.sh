#!/usr/bin/env bash
# Serve Gemma 4 31B-it as the target, with the refusal-free ladder loaded.
#
# The trainer and the eval reach this over HTTP, so it runs in its own shell
# and its own environment -- nothing here is importable by them.
#
#   ./scripts/serve_gemma.sh                    # no reasoning (the paper default)
#   REASONING=on ./scripts/serve_gemma.sh       # enable_thinking targets
#   TP=1 CUDA_VISIBLE_DEVICES=0 ./scripts/serve_gemma.sh    # single GPU
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-google/gemma-4-31B-it}"
PORT="${PORT:-8000}"
LADDER_DIR="${LADDER_DIR:-scaled_target_loras}"
ADAPTER_REPO="${ADAPTER_REPO:-noahrossi/gemma-4-31b-it-refusal-free-v2}"

TP="${TP:-2}"
DP="${DP:-1}"

# Reasoning off by default, matching the paper's non-reasoning runs. With
# reasoning on, a response carries its chain of thought and needs a much
# larger budget -- hence the token defaults below.
REASONING="${REASONING:-off}"

# max_model_len has to cover prompt + target_max_tokens, because vLLM rejects
# a request that exceeds it rather than truncating. The investigator prompt
# renders to ~600 tokens, so these leave room above the preset budgets
# (target_max_tokens 7000 without reasoning, 14000 with).
case "$REASONING" in
  on|1|true|True) MAX_MODEL_LEN="${MAX_MODEL_LEN:-15000}" ;;
  *)              MAX_MODEL_LEN="${MAX_MODEL_LEN:-9000}" ;;
esac

MAX_SEQS="${MAX_SEQS:-512}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-16384}"
# NCCL all-reduce pins several GiB of buffers for the captured CUDA graphs
# after the profiler has already sized the KV cache, so leave more headroom
# than the vLLM default.
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
# A ceiling on adapter rank, not an adapter's rank. vLLM sizes its adapter
# buffers from it, so raising it costs memory on every slot. These adapters
# are rank 1.
MAX_LORA_RANK="${MAX_LORA_RANK:-1}"

INFO='\033[1;34m==>\033[0m'; OK='\033[1;32m/\033[0m'; NC='\033[0m'

if ! compgen -G "${LADDER_DIR}/alpha_*" > /dev/null; then
    echo -e "${INFO} No ladder in ${LADDER_DIR}; building it from ${ADAPTER_REPO}"
    uv run build-ladder --source-repo "$ADAPTER_REPO" --output-dir "$LADDER_DIR"
fi

LORA_ARGS=()
for d in "${LADDER_DIR}"/alpha_*; do
    [ -d "$d" ] && LORA_ARGS+=("$(basename "$d")=$(realpath "$d")")
done
if [ ${#LORA_ARGS[@]} -eq 0 ]; then
    echo "No adapters found in ${LADDER_DIR}" >&2; exit 1
fi
echo -e "${OK} ${#LORA_ARGS[@]} adapters: $(printf '%s ' "${LORA_ARGS[@]%%=*}")"

echo -e "${INFO} Caching ${MODEL} ..."
HF_XET_HIGH_PERFORMANCE=1 uv run --with hf_transfer hf download "$MODEL" >/dev/null
echo -e "${OK} Model ready."

# Long runs fragment the allocator until a full-size prefill chunk can no
# longer find a contiguous block; expandable segments let PyTorch grow a
# segment in place instead of failing on reserved-but-unallocated memory.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLLM_USE_DEEP_GEMM=0

# Custom all-reduce IPC-registers the CUDA-graph buffers and rejects the
# VMM-backed pointers that expandable_segments hands it, so disable it
# whenever more than one GPU is in play.
EXTRA=()
[ $((TP * DP)) -gt 1 ] && EXTRA+=(--disable-custom-all-reduce)

echo -e "${INFO} vLLM on :${PORT}  TP=${TP} DP=${DP}  max_model_len=${MAX_MODEL_LEN}  reasoning=${REASONING}"
exec uv run --group vllm vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --data-parallel-size "$DP" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_SEQS" \
    --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --generation-config vllm \
    --reasoning-parser gemma4 \
    --limit-mm-per-prompt '{"image": 0, "audio": 0, "video": 0}' \
    --enable-lora \
    --max-loras $(( ${#LORA_ARGS[@]} + 1 )) \
    --max-lora-rank "$MAX_LORA_RANK" \
    --trust-remote-code \
    "${EXTRA[@]+"${EXTRA[@]}"}" \
    --lora-modules "${LORA_ARGS[@]}"
