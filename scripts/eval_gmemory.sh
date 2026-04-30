#!/bin/bash

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/data1/JustinLu090/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"

# options: autogen, macnet, camel
MAS="${MAS:-autogen}"

# options: kodcode, triviaqa, popqa, pddl
DATASET="${DATASET:-kodcode}"

MAS_LLM="${MAS_LLM:-Qwen/Qwen3-4B-Instruct-2507}"

# gmemory baseline uses text memory only and does not require the latent weaver.
USE_WEAVER=False
MAS_RAG=gmemory

# Expected baseline database layout:
# results/LatentMem-Qwen3-4B-Trajectory/gmemory/rag_0
DATABASE_DIR="${DATABASE_DIR:-results/LatentMem-Qwen3-4B-Trajectory/gmemory/rag_0}"

python main.py \
    --cfg-path "configs/latentmem/${DATASET}.yaml" \
    --options \
    model.mas.structure "${MAS}" \
    model.mas.llm_name_or_path "${MAS_LLM}" \
    model.memory.llm_name_or_path "${MAS_LLM}" \
    model.memory.rag.mode "${MAS_RAG}" \
    model.memory.rag.database_dir "${DATABASE_DIR}" \
    model.memory.use_weaver "${USE_WEAVER}" \
    model.load_model_path null \
    run.mode eval
