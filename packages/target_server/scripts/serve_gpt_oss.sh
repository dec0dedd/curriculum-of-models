#!/usr/bin/env bash
# Serve gpt-oss-120b as the target, with the refusal-free ladder loaded, under
# the same adapter names as the Gemma script.
#
# gpt-oss always reasons, so there is no reasoning switch: vLLM's harmony
# parser returns the final channel as the response, and PRBO splits the
# reasoning off itself.
#
#   ./scripts/serve_gpt_oss.sh
#   TP=1 CUDA_VISIBLE_DEVICES=0 ./scripts/serve_gpt_oss.sh
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-openai/gpt-oss-120b}"
PORT="${PORT:-8000}"
LADDER_DIR="${LADDER_DIR:-scaled_gpt_oss_loras}"
ADAPTER_REPO="${ADAPTER_REPO:-noahrossi/gpt-oss-120b-refusal-r1-vllm}"

TP="${TP:-2}"
DP="${DP:-1}"

# Covers prompt + target_max_tokens: a 512-token investigator prompt renders
# to ~580 tokens under the harmony template, and the preset budget is 7000.
# vLLM rejects a request over this rather than truncating it, so do not lower
# it without lowering target_max_tokens to match.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_SEQS="${MAX_SEQS:-512}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-16384}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
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
# original/ and metal/ are alternate checkpoints as large as the one vLLM loads.
HF_XET_HIGH_PERFORMANCE=1 uv run --with hf_transfer hf download "$MODEL" \
    --exclude "original/*" --exclude "metal/*" >/dev/null
echo -e "${OK} Model ready."

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLLM_USE_DEEP_GEMM=0

# Custom all-reduce rejects the VMM-backed pointers that expandable_segments
# hands it (and dies with "invalid argument" where GPUs cannot share memory
# over CUDA IPC), so disable it whenever more than one GPU is in play.
EXTRA=()
[ $((TP * DP)) -gt 1 ] && EXTRA+=(--disable-custom-all-reduce)

echo -e "${INFO} vLLM on :${PORT}  TP=${TP} DP=${DP}  max_model_len=${MAX_MODEL_LEN}"
echo -e "    This needs the vLLM fork pinned in pyproject.toml: stock 0.29"
echo -e "    misapplies MoE LoRA across GPUs and silently serves base-model refusals."
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
    --reasoning-parser openai_gptoss \
    --enable-lora \
    --max-loras $(( ${#LORA_ARGS[@]} + 1 )) \
    --max-lora-rank "$MAX_LORA_RANK" \
    "${EXTRA[@]+"${EXTRA[@]}"}" \
    --lora-modules "${LORA_ARGS[@]}"
