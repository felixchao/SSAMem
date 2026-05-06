# SSAMem Training Guide

This guide reflects the current SSAMem workflow for PopQA-style trajectory
memory:

```text
task QA rows
-> collect text MAS trajectories
-> build SSA latent dataset
-> train composer/projector (train-ssa)
-> train retrieval key encoders (train-retrieval)
-> build experience bank / pointer table
-> run SEARCH-only MAS evaluation
```

The key design choice is that SSAMem stores past MAS trajectories as memory.
The bank does not store plain text QA context as its primary memory object.

## 1. What Each Stage Trains

There are two trainable components:

```text
A. Composer / projector
   input: trajectory latent
   output: injectable latent prompt tensor

B. Retrieval encoders
   input: query text + trajectory latent
   output: aligned key vectors for SEARCH
```

This separation matters:

- `train-ssa` learns how to inject latent memory into the LLM.
- `train-retrieval` learns how to find the right latent memory.

## 2. Required Data Flow

SSAMem currently uses this data flow:

```text
PopQA rows
-> collect-text-mas-trajectories
-> build-ssa-data
-> manifest.jsonl + latents/*.pt
-> train-ssa
-> train-retrieval
-> build-experience-bank / eval-mas-memory-agent-loop
```

`train-ssa` and `train-retrieval` both consume the SSA manifest and latent
shards produced by `build-ssa-data`.

## 3. Prepare Task Input

Prepare a JSONL file with at least:

```json
{
  "task_prompt": "What is George Rankin's occupation?",
  "target_text": "politician",
  "context_text": "Known answer aliases: [\"politician\"]"
}
```

Example:

```text
data/trajectories/popqa_input.jsonl
```

## 4. Collect Text MAS Trajectories

This is the teacher-side trajectory collection step.

```bash
python -m ssamem collect-text-mas-trajectories \
  --input data/trajectories/popqa_input.jsonl \
  --output data/trajectories/popqa_text_mas.jsonl \
  --runtime-mode hf \
  --model-name-or-path Qwen/Qwen3-4B-Instruct-2507 \
  --trust-remote-code \
  --device cuda \
  --torch-dtype float16 \
  --load-in-4bit \
  --limit 2500 \
  --task-domain popqa \
  --mas-style camel \
  --max-new-tokens 192
```

Output:

```text
data/trajectories/popqa_text_mas.jsonl
```

## 5. Build SSA Dataset

Convert raw trajectories into an SSA manifest plus latent tensors.

```bash
python -m ssamem build-ssa-data \
  --input data/trajectories/popqa_text_mas.jsonl \
  --output data/trajectory_context_popqa2500_qwen3_4b_768/all \
  --runtime-mode hf \
  --model-name-or-path Qwen/Qwen3-4B-Instruct-2507 \
  --trust-remote-code \
  --device cuda \
  --torch-dtype float16 \
  --load-in-4bit \
  --latent-max-tokens 768
```

Main outputs:

```text
data/trajectory_context_popqa2500_qwen3_4b_768/all/manifest.jsonl
data/trajectory_context_popqa2500_qwen3_4b_768/all/latents/*.pt
```

Optional fold split example:

```text
data/trajectory_context_popqa2500_qwen3_4b_768/folds/fold_00/train.jsonl
data/trajectory_context_popqa2500_qwen3_4b_768/folds/fold_00/val.jsonl
```

## 6. Train the Composer / Projector

Recommended config:

```text
configs/ssamem_phaseA_trajectory_context_popqa2500_768_b4_answerstrong.yaml
```

Current recommended command:

```bash
python -m ssamem train-ssa \
  --config configs/ssamem_phaseA_trajectory_context_popqa2500_768_b4_answerstrong.yaml \
  --device cuda \
  --run-name phaseA-popqa-trajectory
```

Important config fields:

```yaml
data:
  manifest: data/trajectory_context_popqa2500_qwen3_4b_768/all/manifest.jsonl
  valid_manifest: data/trajectory_context_popqa2500_qwen3_4b_768/folds/fold_00/val.jsonl

training:
  base_model: Qwen/Qwen3-4B-Instruct-2507
  learning_rate: 0.00004
  epochs: 3
  batch_size: 4
  alignment:
    answer_loss_weight: 0.1
    latent_max_tokens: 768
```

Checkpoint example:

```text
outputs/phaseA_popqa2500_768_b4_answerstrong/fold_00/distiller.pt
```

## 7. Train the Retrieval Key Encoders

The retrieval model learns:

- `QueryLatentEncoder`: task prompt -> query key
- `MemoryKeyEncoder`: trajectory latent -> memory key

Recommended training command:

```bash
python -m ssamem train-retrieval \
  --config configs/ssamem_phaseA_trajectory_context_popqa2500_768_b4_answerstrong.yaml \
  --manifest data/trajectory_context_popqa2500_qwen3_4b_768/folds/fold_00/train.jsonl \
  --valid-manifest data/trajectory_context_popqa2500_qwen3_4b_768/folds/fold_00/val.jsonl \
  --runtime-mode hf \
  --model-name-or-path Qwen/Qwen3-4B-Instruct-2507 \
  --trust-remote-code \
  --device cuda \
  --retrieval-key-dim 256 \
  --query-max-tokens 96 \
  --query-field task_prompt \
  --batch-size 8 \
  --eval-batch-size 16 \
  --epochs 1 \
  --lr 1e-4 \
  --output-path outputs/retrieval_alignment/popqa_query_memory_key.pt \
  --history-path outputs/retrieval_alignment/popqa_query_memory_key.history.json
```

To continue from a previous retrieval checkpoint:

```bash
python -m ssamem train-retrieval \
  --config configs/ssamem_phaseA_trajectory_context_popqa2500_768_b4_answerstrong.yaml \
  --manifest data/trajectory_context_popqa2500_qwen3_4b_768/folds/fold_00/train.jsonl \
  --valid-manifest data/trajectory_context_popqa2500_qwen3_4b_768/folds/fold_00/val.jsonl \
  --runtime-mode hf \
  --model-name-or-path Qwen/Qwen3-4B-Instruct-2507 \
  --trust-remote-code \
  --device cuda \
  --checkpoint outputs/retrieval_alignment/popqa_query_memory_key.pt \
  --batch-size 8 \
  --eval-batch-size 16 \
  --epochs 1 \
  --lr 1e-4 \
  --output-path outputs/retrieval_alignment/popqa_query_memory_key_continue.pt \
  --history-path outputs/retrieval_alignment/popqa_query_memory_key_continue.history.json
```

Evaluate retrieval alone:

```bash
python -m ssamem eval-retrieval \
  --config configs/ssamem_phaseA_trajectory_context_popqa2500_768_b4_answerstrong.yaml \
  --manifest data/trajectory_context_popqa2500_qwen3_4b_768/folds/fold_00/val.jsonl \
  --candidate-manifest data/trajectory_context_popqa2500_qwen3_4b_768/all/manifest.jsonl \
  --checkpoint outputs/retrieval_alignment/popqa_query_memory_key_continue.pt \
  --batch-size 16 \
  --output-path outputs/retrieval_alignment/popqa_query_memory_key_continue_eval.json
```

## 8. Build the Experience Bank

Use the trained composer and optional trained retrieval checkpoint to build the
runtime memory bank.

Recommended command:

```bash
python -m ssamem build-experience-bank \
  --config configs/ssamem_phaseA_trajectory_context_popqa2500_768_b4_answerstrong.yaml \
  --manifest data/trajectory_context_popqa2500_qwen3_4b_768/all/manifest.jsonl \
  --checkpoint outputs/phaseA_popqa2500_768_b4_answerstrong/fold_00/distiller.pt \
  --retrieval-checkpoint outputs/retrieval_alignment/popqa_query_memory_key_continue.pt \
  --device cuda \
  --limit 1952 \
  --cluster-method offline-kmeans \
  --kmeans-clusters 100 \
  --kmeans-max-iter 50 \
  --output-path outputs/experience_bank/popqa_bank1952_kmeans100.json
```

What this step does:

- projects each trajectory latent into an injectable latent tensor
- computes a retrieval key vector for SEARCH
- groups memories into pointer clusters
- assigns exact addresses such as `<PTR_0x017>:0007`

## 9. Evaluate the Full SSAMem Loop

Current recommended end-to-end evaluation uses `SEARCH-only`, because it gives
cleaner diagnostics than mixing SEARCH and GET.

```bash
python -m ssamem eval-mas-memory-agent-loop \
  --config configs/ssamem_phaseA_trajectory_context_popqa2500_768_b4_answerstrong.yaml \
  --bank-manifest data/trajectory_context_popqa2500_qwen3_4b_768/all/manifest.jsonl \
  --eval-manifest data/trajectory_context_popqa2500_qwen3_4b_768/folds/fold_00/val.jsonl \
  --checkpoint outputs/phaseA_popqa2500_768_b4_answerstrong/fold_00/distiller.pt \
  --retrieval-checkpoint outputs/retrieval_alignment/popqa_query_memory_key_continue.pt \
  --device cuda \
  --bank-limit 2500 \
  --limit 50 \
  --top-k 3 \
  --cluster-method offline-kmeans \
  --kmeans-clusters 100 \
  --kmeans-max-iter 50 \
  --memory-request-policy search-only \
  --max-new-tokens 64 \
  --output-path outputs/mas_latent_eval/memory_agent_loop_fold00_bank1952_kmeans100_trained_retrieval_continue_search_only_limit50.json
```

This command evaluates the full pipeline:

```text
query
-> MAS role asks MemoryAgent to SEARCH
-> trained retriever finds top-k latent memories
-> composer/projector injects mounted latent tensors
-> MAS role answers with latent memory available
```

## 10. How to Interpret the Main Metrics

Two metrics should always be separated:

- `retrieval_target_hit`
  - whether the correct memory target appeared in retrieved top-k memory
- `agent_memory_loop target_hit`
  - whether the final MAS answer was correct

Interpretation:

```text
high retrieval_target_hit + low answer hit
=> retrieval is working, but memory utilization / composer / answer policy is weak

low retrieval_target_hit + low answer hit
=> retrieval is still the main bottleneck
```

## 11. Current Recommended Best Practice

For the current codebase, the most stable evaluation recipe is:

1. Train `train-ssa` on trajectory-derived SSA data.
2. Train `train-retrieval` on the same trajectory-derived manifest.
3. Build/evaluate with `--retrieval-checkpoint`.
4. Use `--memory-request-policy search-only` for clean end-to-end analysis.
5. Diagnose retrieval and answer quality separately.
