# Pointer-Driven SSAMem

`ssamem` is a standalone pipeline for pointer-driven latent communication in
multi-agent systems. It does not depend on the original `latentmem` package.

## What Is Included

- `cli/`
  - CLI argument parser for `python -m ssamem`.
- `commands/`
  - Thin command handlers grouped by responsibility: data preparation,
    training, evaluation, MAS memory evaluation, and demos.
- `workflows/`
  - Higher-level runnable flows used by CLI commands, including demo runs,
    text-MAS trajectory collection, experience-bank preload, and MAS memory
    evaluation.
- `training/`
  - SSA trace/data conversion, manifest utilities, distillation, probing, and
    LMPO-style preference-tuning utilities.
- `data_models.py`
  - `LatentTensor`, `PageTable`, `AgentMessage`, and kernel-side result types.
- `userspace.py`
  - `UserSpaceMAS`, the public user-space entry that directly hosts MAS role
    execution and internally performs soft-prompt injection with
    `LlamaForCausalLM`.
- `kernelspace.py`
  - `MemoryAgent`, `IPCBus`, and `OSKernel` for pointer resolution, MIPS
    retrieval, and episodic consolidation.
- `trainingspace.py` and `ssa_data.py`
  - Backward-compatible import shims. New code should prefer
    `ssamem.training.space` and `ssamem.training.data`.
- `pipeline.py`
  - A top-level orchestrator that connects kernel-space and user-space.
- `mas.py`
  - MAS role specifications, topology registry, and prompt binding logic for
    `camel`, `autogen`, and `debate/macnet`.
- `mas_prompts/`
  - Task-domain prompt libraries migrated from `latentmem/mas_core` for
    `alfworld`, `triviaqa`, `popqa`, `pddl`, and `kodcode`.
- `tokenizer.py`
  - A tiny offline tokenizer for fully local demos.
- `main.py`
  - Minimal CLI entrypoint, dependency guard, and command dispatcher.

## Quick Start

Run an offline end-to-end user-space demo with a tiny randomly initialized
Llama:

```bash
cd LatentMem
python3 -m ssamem demo
```

Run a minimal SSA alignment demo:

```bash
cd LatentMem
python3 -m ssamem train-ssa --steps 3
```

Build SSA training data from MAS traces:

```bash
python3 -m ssamem prepare-ssa-traces --source synthetic-api --limit 200 --output data/traces.synthetic.jsonl
python3 -m ssamem build-ssa-data --input traces.jsonl --output data/ssa
```

Convert a supported HuggingFace dataset into SSA traces when `datasets` can
access the source:

```bash
python3 -m ssamem prepare-ssa-traces --source hf --dataset-name kkkc6696/APIBench --limit 500 --output data/traces.apibench.jsonl
python3 -m ssamem prepare-ssa-traces --source hf --dataset-name sentence-transformers/codesearchnet --subset pair --limit 500 --output data/traces.codesearchnet.jsonl
python3 -m ssamem prepare-ssa-traces --source hf --dataset-name codeparrot/apps --limit 200 --output data/traces.apps.jsonl
```

Run config-driven SSA hidden-state alignment:

```bash
python3 -m ssamem train-ssa --config configs/ssamem_ssa.yaml
```

Collect pointer DPO preferences and train a pointer router:

```bash
python3 -m ssamem collect-dpo-prefs --output data/pointer_prefs.jsonl
python3 -m ssamem collect-dpo-prefs --input data/rollouts.jsonl --output data/pointer_prefs.jsonl
python3 -m ssamem train-pointer-dpo --config configs/ssamem_dpo.yaml
```

Rollout JSONL rows may either include direct `chosen`/`rejected` pointers or a
`candidates` list with `{ "pointer": "<PTR_0x042>", "reward": 1.0 }` records.

Evaluate training artifacts:

```bash
python3 -m ssamem eval-training --manifest data/ssa/manifest.jsonl --preferences data/pointer_prefs.jsonl
```

Run a MAS-style demo:

```bash
cd LatentMem
python3 -m ssamem demo-mas --mas-style camel
```

Run a domain-specific MAS demo with migrated prompt assets:

```bash
cd LatentMem
python3 -m ssamem demo-mas --mas-style autogen --task-domain triviaqa
```

Run a minimal TriviaQA benchmark:

```bash
cd LatentMem
python3 -m ssamem evaluate-triviaqa --limit 20
```

## Runtime Modes

- `tiny-random`
  - Fully offline. Builds a small random `LlamaForCausalLM` inside the
    user-space runtime plus a local tokenizer. Useful for smoke tests.
- `hf`
  - Loads a HuggingFace model and tokenizer into the internal user-space
    runtime from `model_name_or_path`.

Core configuration is split into:

- `RuntimeConfig`
  - Controls the internal execution runtime used by `UserSpaceMAS`
- `KernelConfig`
  - Controls memory, retrieval, and persistence
- `MASConfig`
  - Controls the role topology and optional task-domain prompt pack exposed by
    user-space

## Persistence

If `KernelConfig.storage_root` is set, memory state is persisted under that
directory:

- `page_table.json`
  - Stores pointer-to-storage mappings and the next pointer index.
- `latents/<storage_id>.pt`
  - Stores each `LatentTensor` payload, key vector, utility score, pointer, and
    metadata.

The pipeline also exposes:

- `save_memory_store()`
- `load_memory_store()`

When `autosave=True`, each newly registered or consolidated latent page is
written to disk immediately. When `autoload=True`, persisted memory is restored
when the pipeline starts.

## MAS Orchestration

`ssamem` now includes a mainstream role-based MAS layer inspired by the
structures used in the original `latentmem` project:

- `camel`
  - `user_proxy -> actor -> critic -> summarizer`
- `autogen`
  - `assistant -> user_proxy`
- `debate` / `macnet`
  - two actor-critic branches followed by a summarizer

Each role turn still uses the same latent pointer resolution and soft-prompt
mounting path, but that execution detail is hidden inside `UserSpaceMAS`.

## Example Flow

1. Register latent memory pages in the kernel.
2. Send an IPC message containing `<PTR_...>`.
3. Kernel resolves explicit pointers and optionally prefetches more pages with
   MIPS.
4. User-space mounts the resolved tensors as soft prompts by prepending them to
   `inputs_embeds`.
5. Global reward can trigger episodic consolidation into new `LatentTensor`
   pages.

## TriviaQA Benchmark

The built-in TriviaQA evaluator loads the raw
`mandarjoshi/trivia_qa` `rc.wikipedia.nocontext` split directly, extracts a
small evidence bundle, and compares three conditions:

- `No Memory`
  - Question only
- `Text Memory`
  - Evidence text injected directly into the prompt
- `SSAMem`
  - Evidence converted into a pseudo-latent tensor and mounted through a pointer

This benchmark is intended as a first end-to-end evaluation harness for
pointer-driven memory quality, retrieval behavior, and communication efficiency.
