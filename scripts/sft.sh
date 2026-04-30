#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/data1/JustinLu090/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"

MAS=autogen  # autogen, macnet
MAS_RAG=latentmem
MAS_LLM=Qwen/Qwen3-4B-Instruct-2507  # Qwen/Qwen3-4B-Instruct-2507, meta-llama/Llama-3.1-8B-Instruct
LLM_SUFFIX="${MAS_LLM##*/}"

DATASET=kodcode
DATA_PATH="<COLLECTED_DATA_PATH>/data.json"

accelerate launch \
    --config_file=configs/zero2.yaml \
    main.py \
    --cfg-path configs/latentmem/${DATASET}.yaml \
    --options \
    model.mas.structure ${MAS} \
    model.mas.llm_name_or_path ${MAS_LLM} \
    model.memory.llm_name_or_path ${MAS_LLM} \
    model.memory.rag.mode ${MAS_RAG} \
    model.memory.use_weaver True \
    model.memory.weaver.latents_len 8 \
    dataset.bootstrapped_data_path ${DATA_PATH} \
    run.mode sft \
    run.sft.per_device_train_batch_size 2 \
    run.sft.per_device_eval_batch_size 2 \
    run.sft.gradient_accumulation_steps 1 \

    
