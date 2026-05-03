from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ssamem.commands.common import (
    _build_ssa_distiller_for_eval,
    _write_json,
    build_pipeline_from_args,
    build_training_args_from_config,
    deep_get,
    load_config_file,
    resolve_output_path,
)


def run_eval_training(args) -> None:
    import torch
    from torch.utils.data import DataLoader
    from ssamem.training.space import (
        SSADistiller,
        SSAManifestDataset,
        ExplicitTeacher,
        LatentStudent,
        ssa_collate,
        load_pointer_preferences,
    )

    torch.manual_seed(args.seed)
    config = load_config_file(args.config) if args.config else {}
    args = build_training_args_from_config(config, args) if config else args
    payload: dict[str, Any] = {}
    manifest_path = args.manifest or deep_get(config, "data.manifest")
    if manifest_path:
        pipeline = build_pipeline_from_args(args)
        distiller = SSADistiller(
            student=LatentStudent(pipeline.userspace),
            teacher=ExplicitTeacher(pipeline.userspace),
        )
        distiller.to(args.device)
        checkpoint_path = getattr(args, "checkpoint", None)
        if checkpoint_path:
            load_alignment_checkpoint(distiller, checkpoint_path, map_location=args.device)
            payload["checkpoint"] = checkpoint_path
            payload["checkpoint_loaded"] = True
        else:
            payload["checkpoint_loaded"] = False
        dataset = SSAManifestDataset(manifest_path, map_location=args.device)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=ssa_collate)
        distances = []
        with torch.no_grad():
            for idx, batch in enumerate(dataloader):
                if idx >= args.limit:
                    break
                distances.append(float(distiller.compute_step(batch).latent_distance.cpu().item()))
        payload["latent_distance_mean"] = sum(distances) / max(len(distances), 1)
        payload["latent_distance_count"] = len(distances)
    preferences_path = args.preferences or deep_get(config, "data.preferences")
    if preferences_path:
        prefs = load_pointer_preferences(preferences_path)
        payload["pointer_preferences"] = len(prefs)
        payload["pointer_preferences_valid"] = True
    print(json.dumps(payload, ensure_ascii=False, indent=2))

def run_eval_ssa_report(args) -> None:
    from ssamem.training.space import (
        SSAManifestDataset,
        evaluate_ssa_latent_distances,
        load_alignment_checkpoint,
    )

    config = load_config_file(args.config) if args.config else {}
    args = build_training_args_from_config(config, args) if config else args
    manifest_path = args.manifest or deep_get(config, "data.manifest")
    if not manifest_path:
        raise SystemExit("eval-ssa-report requires --manifest or data.manifest in config.")

    dataset = SSAManifestDataset(manifest_path, map_location=args.device)
    _, baseline_distiller = _build_ssa_distiller_for_eval(args)
    baseline = evaluate_ssa_latent_distances(distiller=baseline_distiller, dataset=dataset, limit=args.limit)

    payload: dict[str, Any] = {
        "manifest": manifest_path,
        "limit": args.limit,
        "baseline": baseline,
        "checkpoint_loaded": False,
    }
    checkpoint_path = getattr(args, "checkpoint", None) or deep_get(config, "training.output_path")
    if checkpoint_path:
        _, trained_distiller = _build_ssa_distiller_for_eval(args)
        load_alignment_checkpoint(trained_distiller, checkpoint_path, map_location=args.device)
        checkpoint_metrics = evaluate_ssa_latent_distances(distiller=trained_distiller, dataset=dataset, limit=args.limit)
        baseline_mean = float(baseline["summary"]["latent_distance"]["mean"])
        trained_mean = float(checkpoint_metrics["summary"]["latent_distance"]["mean"])
        improvement_absolute = baseline_mean - trained_mean
        improvement_relative = 0.0 if baseline_mean == 0.0 else improvement_absolute / baseline_mean
        payload.update(
            {
                "checkpoint": checkpoint_path,
                "checkpoint_loaded": True,
                "checkpoint_metrics": checkpoint_metrics,
                "improvement": {
                    "absolute": improvement_absolute,
                    "relative": improvement_relative,
                },
            }
        )
    output_path = resolve_output_path(args)
    if output_path:
        _write_json(output_path, payload)
        payload["output_path"] = output_path
    print(json.dumps(payload, ensure_ascii=False, indent=2))

def run_probe_ssa_generation(args) -> None:
    from ssamem.training.space import SSAManifestDataset, load_alignment_checkpoint, probe_ssa_generation

    config = load_config_file(args.config) if args.config else {}
    args = build_training_args_from_config(config, args) if config else args
    manifest_path = args.manifest or deep_get(config, "data.manifest")
    if not manifest_path:
        raise SystemExit("probe-ssa-generation requires --manifest or data.manifest in config.")
    checkpoint_path = getattr(args, "checkpoint", None) or deep_get(config, "training.output_path")
    pipeline, distiller = _build_ssa_distiller_for_eval(args)
    if checkpoint_path:
        load_alignment_checkpoint(distiller, checkpoint_path, map_location=args.device)
    dataset = SSAManifestDataset(manifest_path, map_location=args.device)
    payload = probe_ssa_generation(
        userspace=pipeline.userspace,
        dataset=dataset,
        student=distiller.student if checkpoint_path else None,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
        max_keywords=args.max_keywords,
    )
    payload["manifest"] = manifest_path
    payload["checkpoint"] = checkpoint_path
    payload["checkpoint_loaded"] = bool(checkpoint_path)
    output_path = resolve_output_path(args)
    if output_path:
        _write_json(output_path, payload)
        payload["output_path"] = output_path
    print(json.dumps(payload, ensure_ascii=False, indent=2))

def run_collect_lmpo_rollouts(args) -> None:
    from ssamem.training.space import (
        SSAManifestDataset,
        collect_single_latent_lmpo_rollouts,
        load_alignment_checkpoint,
    )

    config = load_config_file(args.config) if args.config else {}
    args = build_training_args_from_config(config, args) if config else args
    manifest_path = args.manifest or deep_get(config, "data.manifest")
    if not manifest_path:
        raise SystemExit("collect-lmpo-rollouts requires --manifest or data.manifest in config.")
    checkpoint_path = getattr(args, "checkpoint", None) or deep_get(config, "training.output_path")
    if not checkpoint_path:
        raise SystemExit("collect-lmpo-rollouts requires --checkpoint or training.output_path in config.")
    pipeline, distiller = _build_ssa_distiller_for_eval(args)
    load_alignment_checkpoint(distiller, checkpoint_path, map_location=args.device)
    dataset = SSAManifestDataset(manifest_path, map_location=args.device)
    payload = collect_single_latent_lmpo_rollouts(
        userspace=pipeline.userspace,
        dataset=dataset,
        student=distiller.student,
        output_path=args.output_path,
        limit=args.limit,
        rollouts_per_sample=args.rollouts_per_sample,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    payload.update(
        {
            "manifest": manifest_path,
            "checkpoint": checkpoint_path,
            "run_name": args.run_name or "collect-lmpo-rollouts",
            "seed": args.seed,
        }
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))

def run_build_lmpo_pairs(args) -> None:
    from ssamem.training.space import build_lmpo_pairs_from_rollouts

    payload = build_lmpo_pairs_from_rollouts(
        args.rollouts,
        output_path=args.output_path,
        margin=args.margin,
        max_pairs_per_sample=args.max_pairs_per_sample,
        require_chosen_target_hit=args.require_chosen_target_hit,
        reject_invalid_chosen=not args.allow_invalid_chosen,
    )
    payload.update({"run_name": args.run_name or "build-lmpo-pairs", "seed": args.seed})
    print(json.dumps(payload, ensure_ascii=False, indent=2))

def run_eval_pointer_routing(args) -> None:
    from ssamem.training.space import (
        evaluate_pointer_routing,
        initialize_pointer_vocabulary,
        load_pointer_preferences,
    )

    config = load_config_file(args.config) if args.config else {}
    args = build_training_args_from_config(config, args) if config else args
    preferences_path = args.preferences or deep_get(config, "data.preferences")
    if not preferences_path:
        raise SystemExit("eval-pointer-routing requires --preferences or data.preferences in config.")
    preferences = load_pointer_preferences(preferences_path)
    payload = {"preferences": preferences_path}
    if args.runtime_mode == "hf" and args.model_name_or_path:
        pipeline = build_pipeline_from_args(args)
        initialize_pointer_vocabulary(pipeline.userspace.tokenizer, pipeline.userspace.model)
        payload.update(
            evaluate_pointer_routing(
                preferences,
                model=pipeline.userspace.model,
                tokenizer=pipeline.userspace.tokenizer,
                device=args.device,
                seed=args.seed,
                limit=args.limit,
            )
        )
    else:
        payload.update(evaluate_pointer_routing(preferences, seed=args.seed, limit=args.limit))
    output_path = resolve_output_path(args)
    if output_path:
        _write_json(output_path, payload)
        payload["output_path"] = output_path
    print(json.dumps(payload, ensure_ascii=False, indent=2))

def run_ssa_suite(args) -> None:
    import torch
    from torch.utils.data import DataLoader
    from ssamem.training.space import (
        SSADistiller,
        SSAManifestDataset,
        ExplicitTeacher,
        LatentStudent,
        ssa_collate,
        evaluate_ssa_latent_distances,
        load_alignment_checkpoint,
        prepare_lora_for_training,
        probe_ssa_generation,
        split_ssa_manifest,
    )

    config = load_config_file(args.config) if args.config else {}
    args = build_training_args_from_config(config, args) if config else args
    manifest_path = args.manifest or deep_get(config, "data.manifest")
    if not manifest_path:
        raise SystemExit("run-ssa-suite requires --manifest or data.manifest in config.")

    suite_root = Path(args.suite_dir)
    suite_root.mkdir(parents=True, exist_ok=True)
    splits_dir = suite_root / "splits"
    reports_dir = suite_root / "reports"
    checkpoints_dir = suite_root / "checkpoints"
    reports_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    split_payload = split_ssa_manifest(
        manifest_path,
        output_dir=splits_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    _write_json(reports_dir / "split_summary.json", split_payload)

    train_manifest = split_payload["splits"]["train"]["manifest"]
    val_manifest = split_payload["splits"]["val"]["manifest"]
    test_manifest = split_payload["splits"]["test"]["manifest"]

    torch.manual_seed(args.seed)
    pipeline = build_pipeline_from_args(args)
    lora_cfg = getattr(args, "lora", {})
    if lora_cfg and args.runtime_mode == "hf":
        pipeline.userspace._runtime.model = prepare_lora_for_training(
            pipeline.userspace.model,
            r=int(lora_cfg.get("r", 128)),
            alpha=int(lora_cfg.get("alpha", 32)),
            target_modules=tuple(lora_cfg.get("target_modules", ["q_proj", "v_proj"])),
        )

    distiller = SSADistiller(
        student=LatentStudent(userspace=pipeline.userspace),
        teacher=ExplicitTeacher(userspace=pipeline.userspace),
        loss_type=getattr(args, "alignment", {}).get("loss_type", "smooth_l1"),
        distill_loss_div_std=bool(getattr(args, "alignment", {}).get("distill_loss_div_std", True)),
    )
    optimizer = torch.optim.Adam(distiller.parameters(), lr=args.lr)
    train_dataset = SSAManifestDataset(train_manifest, map_location=args.device)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=ssa_collate)
    history = distiller.fit(
        train_loader,
        optimizer=optimizer,
        epochs=args.epochs,
        device=args.device,
        log_every=args.log_every,
    )
    checkpoint_path = checkpoints_dir / "ssa_distiller.pt"
    torch.save(
        {
            "student_projection": distiller.student_projection.state_dict(),
            "memory_composer": distiller.student.memory_composer.state_dict(),
            "memory_projector": distiller.student.memory_projector.state_dict(),
            "history": history,
            "config": config,
        },
        checkpoint_path,
    )
    if hasattr(pipeline.userspace.model, "save_pretrained"):
        adapter_dir = checkpoints_dir / "adapter"
        pipeline.userspace.model.save_pretrained(adapter_dir)
    _write_json(
        reports_dir / "train_history.json",
        {
            "history": history,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "checkpoint": str(checkpoint_path),
        },
    )

    def _paired_report(split_name: str, split_manifest_path: str) -> dict[str, Any]:
        dataset = SSAManifestDataset(split_manifest_path, map_location=args.device)
        _, baseline_distiller = _build_ssa_distiller_for_eval(args)
        baseline = evaluate_ssa_latent_distances(distiller=baseline_distiller, dataset=dataset, limit=args.eval_limit)
        _, trained_distiller = _build_ssa_distiller_for_eval(args)
        load_alignment_checkpoint(trained_distiller, checkpoint_path, map_location=args.device)
        trained = evaluate_ssa_latent_distances(distiller=trained_distiller, dataset=dataset, limit=args.eval_limit)
        baseline_mean = float(baseline["summary"]["mean"])
        trained_mean = float(trained["summary"]["mean"])
        return {
            "split": split_name,
            "manifest": split_manifest_path,
            "baseline": baseline,
            "checkpoint_metrics": trained,
            "improvement": {
                "absolute": baseline_mean - trained_mean,
                "relative": 0.0 if baseline_mean == 0.0 else (baseline_mean - trained_mean) / baseline_mean,
            },
        }

    val_report = _paired_report("val", val_manifest)
    test_report = _paired_report("test", test_manifest)
    _write_json(reports_dir / "val_report.json", val_report)
    _write_json(reports_dir / "test_report.json", test_report)

    val_probe = probe_ssa_generation(
        userspace=pipeline.userspace,
        dataset=SSAManifestDataset(val_manifest, map_location=args.device),
        limit=args.probe_limit,
        max_new_tokens=getattr(args, "max_new_tokens", 48),
        max_keywords=args.max_keywords,
    )
    test_probe = probe_ssa_generation(
        userspace=pipeline.userspace,
        dataset=SSAManifestDataset(test_manifest, map_location=args.device),
        limit=args.probe_limit,
        max_new_tokens=getattr(args, "max_new_tokens", 48),
        max_keywords=args.max_keywords,
    )
    _write_json(reports_dir / "val_probe.json", val_probe)
    _write_json(reports_dir / "test_probe.json", test_probe)

    summary = {
        "suite_dir": str(suite_root),
        "checkpoint": str(checkpoint_path),
        "split_summary": split_payload,
        "val_improvement": val_report["improvement"],
        "test_improvement": test_report["improvement"],
        "val_probe_summary": {
            "no_memory": val_probe["no_memory"],
            "text_memory": val_probe["text_memory"],
            "latent_memory": val_probe["latent_memory"],
        },
        "test_probe_summary": {
            "no_memory": test_probe["no_memory"],
            "text_memory": test_probe["text_memory"],
            "latent_memory": test_probe["latent_memory"],
        },
    }
    summary_path = args.output_path or (suite_root / "suite_summary.json")
    _write_json(summary_path, summary)
    summary["output_path"] = str(summary_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

def run_evaluate_triviaqa(args) -> None:
    from ssamem.evaluate_triviaqa import evaluate_triviaqa

    pipeline = build_pipeline_from_args(args)
    summary = evaluate_triviaqa(
        pipeline=pipeline,
        split=args.split,
        limit=args.limit,
        max_snippets=args.max_snippets,
        max_evidence_chars=args.max_evidence_chars,
        top_k_prefetch=args.top_k_prefetch,
        latent_max_tokens=args.latent_max_tokens,
        output_path=resolve_output_path(args),
    )
    summary["run_name"] = args.run_name or "evaluate-triviaqa"
    summary["seed"] = args.seed
    summary["output_dir"] = args.output_dir
    summary["log_level"] = args.log_level
    print(json.dumps(summary, ensure_ascii=False, indent=2))
