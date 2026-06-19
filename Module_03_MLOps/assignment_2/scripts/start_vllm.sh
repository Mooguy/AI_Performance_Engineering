# #!/usr/bin/env bash
# #
# # Start vLLM with your chosen configuration.
# # Reference: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html

# set -euo pipefail

# MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"

# exec uv run python -m vllm.entrypoints.openai.api_server \
#     --model "$MODEL" \
#     --host 0.0.0.0 \
#     --port 8000
#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export NCCL_NTHREADS=8
export UV_THREADPOOL_SIZE=8

MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
# MODEL="Qwen/Qwen3-0.6B"

exec uv run python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.68 \
    --max-num-seqs 48 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --disable-log-requests