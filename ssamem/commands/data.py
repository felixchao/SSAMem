from __future__ import annotations

import json
from pathlib import Path

from ssamem.commands.builders import build_pipeline_from_args
from ssamem.commands.config_args import build_training_args_from_config
from ssamem.utils.config import deep_get, load_config_file


def run_build_ssa_data(args) -> None:
    if not args.no_latents:
        import torch

        torch.manual_seed(args.seed)
        from ssamem.training.space import build_ssa_data_from_traces

        pipeline = build_pipeline_from_args(args)
        samples = build_ssa_data_from_traces(
            input_path=args.input,
            output_dir=args.output,
            userspace=pipeline.userspace,
            latent_max_tokens=args.latent_max_tokens,
        )
        sample_count = len(samples)
    else:
        from ssamem.training.data import build_ssa_manifest_without_latents

        sample_count = build_ssa_manifest_without_latents(input_path=args.input, output_dir=args.output)
    print(
        json.dumps(
            {
                "samples": sample_count,
                "manifest": str(Path(args.output) / "manifest.jsonl"),
                "latents_built": not args.no_latents,
                "run_name": args.run_name or "build-ssa-data",
            },
            ensure_ascii=False,
            indent=2,
        )
    )

def run_prepare_ssa_traces(args) -> None:
    from ssamem.training.data import (
        ssa_records_from_hf_dataset,
        ssa_records_from_kodcode,
        ssa_records_from_popqa,
        generate_synthetic_api_ssa_records,
        save_ssa_trace_records,
    )

    if args.source == "synthetic-api":
        records = generate_synthetic_api_ssa_records(limit=args.limit, seed=args.seed)
    elif args.source == "popqa":
        records = ssa_records_from_popqa(split=args.split, limit=args.limit)
    elif args.source == "kodcode":
        records = ssa_records_from_kodcode(split=args.split, limit=args.limit)
    else:
        if not args.dataset_name:
            raise SystemExit("--dataset-name is required when --source=hf.")
        records = ssa_records_from_hf_dataset(
            dataset_name=args.dataset_name,
            subset=args.subset,
            split=args.split,
            limit=args.limit,
        )
    save_ssa_trace_records(records, args.output)
    print(
        json.dumps(
            {
                "records": len(records),
                "output": args.output,
                "source": args.source,
                "dataset_name": args.dataset_name,
                "run_name": args.run_name or "prepare-ssa-traces",
            },
            ensure_ascii=False,
            indent=2,
        )
    )

def run_collect_dpo_prefs(args) -> None:
    from ssamem.training.data import (
        collect_synthetic_pointer_preferences,
        pointer_preferences_from_rollouts,
        save_pointer_preferences,
    )

    config = load_config_file(args.config) if args.config else {}
    prompts: list[str] = []
    input_path = args.input or deep_get(config, "data.prompts")
    rollout_path = deep_get(config, "data.rollouts")
    if input_path and str(input_path).endswith(".jsonl"):
        try:
            samples = pointer_preferences_from_rollouts(input_path)
        except ValueError:
            samples = []
        if samples:
            save_pointer_preferences(samples, args.output)
            print(json.dumps({"preferences": len(samples), "output": args.output}, ensure_ascii=False, indent=2))
            return
    if rollout_path:
        samples = pointer_preferences_from_rollouts(rollout_path)
        save_pointer_preferences(samples, args.output)
        print(json.dumps({"preferences": len(samples), "output": args.output}, ensure_ascii=False, indent=2))
        return
    if input_path:
        for row in Path(input_path).read_text(encoding="utf-8").splitlines():
            if row.strip():
                try:
                    payload = json.loads(row)
                    prompts.append(str(payload.get("prompt") or payload.get("task_prompt") or row))
                except json.JSONDecodeError:
                    prompts.append(row.strip())
    else:
        prompts = [
            "前端已完成，請實作稅率計算 API。",
            "Worker needs the API contract memory page before coding.",
        ]
    pointers = deep_get(config, "rollout.pointers", args.pointers)
    positive_pointer = args.positive_pointer or deep_get(config, "rollout.positive_pointer")
    samples = collect_synthetic_pointer_preferences(
        prompts=prompts,
        pointers=pointers,
        positive_pointer=positive_pointer,
    )
    save_pointer_preferences(samples, args.output)
    print(json.dumps({"preferences": len(samples), "output": args.output}, ensure_ascii=False, indent=2))

def run_split_ssa_manifest(args) -> None:
    from ssamem.training.space import split_ssa_manifest

    payload = split_ssa_manifest(
        args.manifest,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))

def run_kfold_ssa_manifest(args) -> None:
    from ssamem.training.space import kfold_ssa_manifest

    payload = kfold_ssa_manifest(
        args.manifest,
        output_dir=args.output,
        folds=args.folds,
        seed=args.seed,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))

def run_filter_ssa_manifest(args) -> None:
    from ssamem.training.space import filter_ssa_manifest

    payload = filter_ssa_manifest(
        args.manifest,
        output_path=args.output,
        metadata_key=args.metadata_key,
        metadata_value=args.metadata_value,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))

def run_make_multimemory_ssa_manifest(args) -> None:
    from ssamem.training.space import make_multimemory_ssa_manifest

    payload = make_multimemory_ssa_manifest(
        args.manifest,
        output_path=args.output,
        distractors=args.distractors,
        seed=args.seed,
        avoid_same_target=not args.allow_same_target,
        shuffle_memories=args.shuffle_memories,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
