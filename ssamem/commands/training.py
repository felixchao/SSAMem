from __future__ import annotations

import json
from pathlib import Path

from ssamem.commands.common import (
    _write_json,
    build_pipeline_from_args,
    build_synthetic_ssa_batches,
    build_training_args_from_config,
    deep_get,
    load_config_file,
    summarize_metric_history,
)


def run_train_ssa(args) -> None:
    import torch
    from torch.utils.data import DataLoader
    from ssamem.training.space import (
        SSADistiller,
        SSAManifestDataset,
        ExplicitTeacher,
        LatentStudent,
        load_alignment_checkpoint,
        ssa_collate,
        prepare_lora_for_training,
    )

    torch.manual_seed(args.seed)
    lmpo_overrides = {
        "lr": args.lr,
        "epochs": args.epochs,
        "beta": args.beta,
        "answer_ce_weight": args.answer_ce_weight,
        "max_pairs": args.max_pairs,
        "log_every": args.log_every,
        "grad_clip_norm": args.grad_clip_norm,
    }
    config = load_config_file(args.config) if args.config else {}
    args = build_training_args_from_config(config, args) if config else args
    for name, value in lmpo_overrides.items():
        setattr(args, name, value)
    pipeline = build_pipeline_from_args(args)
    lora_cfg = getattr(args, "lora", {})
    alignment_cfg = getattr(args, "alignment", {})
    if lora_cfg and args.runtime_mode == "hf":
        pipeline.userspace._runtime.model = prepare_lora_for_training(
            pipeline.userspace.model,
            r=int(lora_cfg.get("r", 128)),
            alpha=int(lora_cfg.get("alpha", 32)),
            target_modules=tuple(lora_cfg.get("target_modules", ["q_proj", "v_proj"])),
        )
        if not bool(alignment_cfg.get("train_lora_backbone", False)):
            for param in pipeline.userspace.model.parameters():
                param.requires_grad = False

    student = LatentStudent(userspace=pipeline.userspace)
    teacher = ExplicitTeacher(userspace=pipeline.userspace)
    distiller = SSADistiller(
        student=student,
        teacher=teacher,
        loss_type=alignment_cfg.get("loss_type", "smooth_l1"),
        distill_loss_div_std=bool(alignment_cfg.get("distill_loss_div_std", True)),
        answer_loss_weight=float(alignment_cfg.get("answer_loss_weight", 0.0)),
        preference_loss_weight=float(alignment_cfg.get("preference_loss_weight", 0.0)),
        preference_beta=float(alignment_cfg.get("preference_beta", 0.1)),
    )
    if getattr(args, "checkpoint", None):
        load_alignment_checkpoint(distiller, args.checkpoint, map_location=args.device)

    optimizer = torch.optim.Adam((param for param in distiller.parameters() if param.requires_grad), lr=args.lr)
    manifest_path = args.manifest or deep_get(config, "data.manifest")
    if manifest_path:
        dataset = SSAManifestDataset(manifest_path, map_location=args.device)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=ssa_collate)
    else:
        dataloader = build_synthetic_ssa_batches(pipeline.userspace.hidden_size, args.steps)
    losses = distiller.fit(
        dataloader,
        optimizer=optimizer,
        epochs=args.epochs,
        device=args.device,
        grad_clip_norm=getattr(args, "grad_clip_norm", None),
        log_every=args.log_every,
    )
    output_path = args.output_path or deep_get(config, "training.output_path")
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "student_projection": distiller.student_projection.state_dict(),
                "memory_composer": distiller.student.memory_composer.state_dict(),
                "memory_projector": distiller.student.memory_projector.state_dict(),
                "history": losses,
                "config": config,
            },
            output_path,
        )
    history_path = getattr(args, "history_path", None)
    if history_path is None and output_path:
        history_path = str(Path(output_path).with_suffix(".history.json"))
    history_summary = summarize_metric_history(losses)
    if history_path:
        _write_json(history_path, {"history": losses, "summary": history_summary})
        _write_json(str(Path(history_path).with_suffix(".summary.json")), history_summary)
    if hasattr(pipeline.userspace.model, "save_pretrained") and deep_get(config, "training.adapter_output_dir"):
        pipeline.userspace.model.save_pretrained(deep_get(config, "training.adapter_output_dir"))
    print(
        json.dumps(
            {
                "steps": args.steps,
                "epochs": args.epochs,
                "history": losses,
                "history_path": history_path,
                "history_summary": history_summary,
                "checkpoint": output_path,
                "run_name": args.run_name or "train-ssa",
                "seed": args.seed,
                "output_dir": args.output_dir,
                "log_level": args.log_level,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

def run_train_pointer_dpo(args) -> None:
    import torch
    from ssamem.training.space import (
        initialize_pointer_vocabulary,
        load_pointer_preferences,
        prepare_lora_for_training,
        train_pointer_dpo,
    )

    torch.manual_seed(args.seed)
    config = load_config_file(args.config) if args.config else {}
    args = build_training_args_from_config(config, args) if config else args
    if not args.model_name_or_path and args.runtime_mode == "hf":
        raise SystemExit("train-pointer-dpo requires --model-name-or-path or training.base_model in config.")
    pipeline = build_pipeline_from_args(args)
    initialize_pointer_vocabulary(pipeline.userspace.tokenizer, pipeline.userspace.model)
    lora_cfg = getattr(args, "lora", {})
    if lora_cfg and args.runtime_mode == "hf":
        pipeline.userspace._runtime.model = prepare_lora_for_training(
            pipeline.userspace.model,
            r=int(lora_cfg.get("r", 128)),
            alpha=int(lora_cfg.get("alpha", 32)),
            target_modules=tuple(lora_cfg.get("target_modules", ["q_proj", "v_proj"])),
        )
    preferences_path = args.preferences or deep_get(config, "data.preferences")
    if not preferences_path:
        raise SystemExit("train-pointer-dpo requires --preferences or data.preferences in config.")
    output_dir = args.output_dir or deep_get(config, "training.dpo.output_dir") or "pointer_dpo_adapter"
    dpo_cfg = getattr(args, "dpo", {})
    result = train_pointer_dpo(
        model=pipeline.userspace.model,
        tokenizer=pipeline.userspace.tokenizer,
        preference_samples=load_pointer_preferences(preferences_path),
        output_dir=output_dir,
        beta=float(dpo_cfg.get("beta", 0.1)),
        max_prompt_length=int(dpo_cfg.get("max_prompt_length", 1024)),
        max_completion_length=int(dpo_cfg.get("max_completion_length", 16)),
        per_device_train_batch_size=int(dpo_cfg.get("per_device_train_batch_size", 2)),
        gradient_accumulation_steps=int(dpo_cfg.get("gradient_accumulation_steps", 32)),
        learning_rate=float(dpo_cfg.get("learning_rate", 5e-5)),
        num_train_epochs=float(dpo_cfg.get("num_train_epochs", 1.0)),
    )
    if hasattr(pipeline.userspace.model, "save_pretrained"):
        pipeline.userspace.model.save_pretrained(output_dir)
    print(json.dumps({"output_dir": output_dir, "train_result": str(result)}, ensure_ascii=False, indent=2))

def run_train_lmpo_projector(args) -> None:
    import torch
    from ssamem.training.space import (
        SSADistiller,
        ExplicitTeacher,
        LatentStudent,
        load_alignment_checkpoint,
        train_lmpo_projector,
    )

    torch.manual_seed(args.seed)
    lmpo_overrides = {
        "lr": args.lr,
        "epochs": args.epochs,
        "beta": args.beta,
        "answer_ce_weight": args.answer_ce_weight,
        "hidden_anchor_weight": args.hidden_anchor_weight,
        "max_pairs": args.max_pairs,
        "log_every": args.log_every,
        "grad_clip_norm": args.grad_clip_norm,
    }
    config = load_config_file(args.config) if args.config else {}
    args = build_training_args_from_config(config, args) if config else args
    for name, value in lmpo_overrides.items():
        setattr(args, name, value)
    pipeline = build_pipeline_from_args(args)
    teacher = ExplicitTeacher(pipeline.userspace)
    student = LatentStudent(pipeline.userspace)
    reference_student = LatentStudent(pipeline.userspace)
    student_distiller = SSADistiller(student=student, teacher=teacher)
    reference_distiller = SSADistiller(student=reference_student, teacher=teacher)
    load_alignment_checkpoint(student_distiller, args.checkpoint, map_location=args.device)
    load_alignment_checkpoint(reference_distiller, args.checkpoint, map_location=args.device)
    student_distiller.to(args.device)
    reference_distiller.to(args.device)
    payload = train_lmpo_projector(
        student=student,
        teacher=teacher,
        reference_student=reference_student,
        pairs_path=args.pairs,
        output_path=args.output_path,
        lr=args.lr,
        epochs=args.epochs,
        beta=args.beta,
        answer_ce_weight=args.answer_ce_weight,
        hidden_anchor_weight=args.hidden_anchor_weight,
        max_pairs=args.max_pairs,
        log_every=args.log_every,
        grad_clip_norm=args.grad_clip_norm,
        device=args.device,
        student_projection_state=student_distiller.student_projection.state_dict(),
    )
    history_path = str(Path(args.output_path).with_suffix(".history.json"))
    _write_json(history_path, {"history": payload["history"], "summary": payload["summary"]})
    payload.pop("history", None)
    payload.update(
        {
            "pairs_path": args.pairs,
            "init_checkpoint": args.checkpoint,
            "history_path": history_path,
            "run_name": args.run_name or "train-lmpo-projector",
            "seed": args.seed,
        }
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
