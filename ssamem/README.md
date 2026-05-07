# SSAMem

`ssamem` is a pointer-driven latent memory system for multi-agent systems
(MAS). It combines:

- a trained composer/projector for latent memory injection
- a trained query-to-memory retriever for SEARCH
- a pointer table and clustered experience bank
- a MAS execution loop that requests memory before answering

This package is the implementation of the SSAMem pipeline inside this repo.

## Current Package Layout

The runtime architecture is now grouped under:

```text
ssamem/core/
  pipeline.py
  userspace.py
  kernelspace.py
  mas.py
  memory_actions.py
```

Supporting packages:

```text
ssamem/training/            # SSA and composer/projector training
ssamem/retrieval_training/  # retrieval alignment training
ssamem/commands/            # command handlers
ssamem/cli/                 # CLI parser registration
ssamem/utils/               # config / io / metrics helpers
ssamem/workflows/           # higher-level experiment flows
```

## Method Overview

SSAMem treats memory as past MAS trajectories rather than plain supporting
documents.

The current pipeline has four major stages:

```text
1. Collect text MAS trajectories
2. Convert them into SSA latent data
3. Train:
   - composer/projector
   - retrieval key encoders
4. Build a clustered latent experience bank and run MAS with MemoryAgent SEARCH
```

At inference time:

```text
task query
-> MAS role issues SEARCH
-> MemoryAgent retrieves top-k latent memories
-> composer/projector mounts latent tensors into the role prompt
-> role answers using injected latent memory
```

## Core Architecture

SSAMem separates retrieval from injection.

### Retrieval path

- `QueryLatentEncoder`
  - encodes a task prompt into a retrieval query key
- `MemoryKeyEncoder`
  - encodes a stored trajectory latent into a memory key
- `MemoryAgent SEARCH`
  - compares query keys against memory keys and returns top-k memory addresses

### Injection path

- `composer / projector`
  - converts a trajectory latent into a prompt-mountable latent tensor
- `UserSpaceMAS`
  - injects mounted latent tensors into the LLM as soft prompts
- `MAS roles`
  - answer with latent memory already mounted

This means each memory page has two representations:

- `key_vector`
  - used for SEARCH
- `tensor_data`
  - used for latent prompt injection

## Pointer Table and Experience Clusters

The experience bank is organized as:

```text
summary key -> pointer -> cluster -> indexed latent memories
```

Each cluster stores:

- one pointer, such as `<PTR_0x017>`
- multiple latent memories
- exact addresses, such as `<PTR_0x017>:0007`

The pointer table provides a coarse summary-key view of the bank, while exact
addresses provide fine-grained memory access.



## Main Modules

- `cli/`
  - parser for `python -m ssamem`
- `commands/`
  - command handlers grouped by domain
- `workflows/`
  - higher-level experiment and evaluation flows
- `retrieval_training/`
  - query/memory retrieval alignment training
- `training/`
  - SSA dataset, manifest, and distillation code
- `core/userspace.py`
  - MAS execution loop and latent prompt mounting
- `core/kernelspace.py`
  - memory store, pointer table, clustering, SEARCH, GET
- `core/pipeline.py`
  - top-level system wiring across userspace and kernelspace
- `core/memory_actions.py`
  - parsing and enforcing SEARCH/GET/NONE behavior
- `core/mas.py`
  - MAS topology, role templates, and prompt rendering

## Runtime Layering

The end-to-end runtime can be read as:

```text
task query
-> core/pipeline.py
-> core/userspace.py asks for memory or runs a role turn
-> core/kernelspace.py resolves SEARCH / GET
-> retrieved latent memory is mounted back into userspace
-> MAS role continues and produces the final answer
```

## Key Commands

### Collect text trajectories

```bash
python -m ssamem collect-text-mas-trajectories ...
```

### Build SSA latent data

```bash
python -m ssamem build-ssa-data ...
```

### Train composer/projector

```bash
python -m ssamem train-ssa --config configs/ssamem_phaseA_trajectory_context_popqa2500_768_b4_answerstrong.yaml
```

### Train retrieval encoders

```bash
python -m ssamem train-retrieval ...
```

### Evaluate retrieval alone

```bash
python -m ssamem eval-retrieval ...
```

### Build the experience bank

```bash
python -m ssamem build-experience-bank ...
```

### Evaluate the full MAS memory loop

```bash
python -m ssamem eval-mas-memory-agent-loop ...
```

See [TrainingGuide.md](/data1/JustinLu090/SSAMem/TrainingGuide.md) for the
recommended end-to-end training and evaluation workflow.
