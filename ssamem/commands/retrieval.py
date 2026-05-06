from __future__ import annotations

import json
from pathlib import Path

from ssamem.commands.common import (
    _write_json,
    build_pipeline_from_args,
    build_training_args_from_config,
    deep_get,
    load_config_file,
    summarize_metric_history,
)


def _apply_retrieval_config(args, config: dict):
    cli_values = {
        "retrieval_key_dim": getattr(args, "retrieval_key_dim", 0) or 0,
        "query_max_tokens": getattr(args, "query_max_tokens", 96),
        "temperature": getattr(args, "temperature", 0.07),
        "query_field": getattr(args, "query_field", "task_prompt"),
        "lr": getattr(args, "lr", 1e-4),
        "epochs": getattr(args, "epochs", 1),
        "batch_size": getattr(args, "batch_size", 8),
        "eval_batch_size": getattr(args, "eval_batch_size", 16),
        "log_every": getattr(args, "log_every", 10),
        "grad_clip_norm": getattr(args, "grad_clip_norm", 1.0),
        "output_path": getattr(args, "output_path", None),
        "history_path": getattr(args, "history_path", None),
    }
    args = build_training_args_from_config(config, args) if config else args
    retrieval_cfg = dict(config.get("retrieval") or {}) if config else {}
    args.retrieval_key_dim = int(retrieval_cfg.get("key_dim", cli_values["retrieval_key_dim"]))
    args.query_max_tokens = int(retrieval_cfg.get("query_max_tokens", cli_values["query_max_tokens"]))
    args.temperature = float(retrieval_cfg.get("temperature", cli_values["temperature"]))
    args.query_field = str(retrieval_cfg.get("query_field", cli_values["query_field"]))
    args.lr = float(retrieval_cfg.get("learning_rate", cli_values["lr"]))
    args.epochs = int(retrieval_cfg.get("epochs", cli_values["epochs"]))
    args.batch_size = int(retrieval_cfg.get("batch_size", cli_values["batch_size"]))
    args.eval_batch_size = int(retrieval_cfg.get("eval_batch_size", cli_values["eval_batch_size"]))
    args.log_every = int(retrieval_cfg.get("log_every", cli_values["log_every"]))
    args.grad_clip_norm = retrieval_cfg.get("grad_clip_norm", cli_values["grad_clip_norm"])
    args.output_path = cli_values["output_path"] or retrieval_cfg.get("output_path")
    args.history_path = cli_values["history_path"] or retrieval_cfg.get("history_path")
    return args


def _build_retrieval_model(args, pipeline):
    from ssamem.retrieval_training import RetrievalAlignmentModel, RetrievalModelConfig

    key_dim = int(args.retrieval_key_dim or pipeline.userspace.hidden_size)
    model_config = RetrievalModelConfig(
        hidden_size=pipeline.userspace.hidden_size,
        key_dim=key_dim,
        query_max_tokens=args.query_max_tokens,
        temperature=args.temperature,
    )
    model = RetrievalAlignmentModel(pipeline.userspace, model_config).to(args.device)
    return model


def _fill_retrieval_config_from_checkpoint(args) -> None:
    if not getattr(args, "checkpoint", None):
        return
    if getattr(args, "retrieval_key_dim", 0):
        return
    import torch

    payload = torch.load(args.checkpoint, map_location="cpu")
    checkpoint_config = dict(payload.get("retrieval_config") or {})
    if not checkpoint_config:
        return
    args.retrieval_key_dim = int(checkpoint_config.get("key_dim", args.retrieval_key_dim or 0))
    args.query_max_tokens = int(checkpoint_config.get("query_max_tokens", args.query_max_tokens))
    args.temperature = float(checkpoint_config.get("temperature", args.temperature))


def run_train_retrieval(args) -> None:
    import torch
    from torch.utils.data import DataLoader
    from ssamem.retrieval_training import (
        RetrievalManifestDataset,
        evaluate_retrieval_alignment,
        load_retrieval_checkpoint,
        retrieval_collate,
        save_retrieval_checkpoint,
        train_retrieval_alignment,
    )

    torch.manual_seed(args.seed)
    config = load_config_file(args.config) if args.config else {}
    args = _apply_retrieval_config(args, config)
    _fill_retrieval_config_from_checkpoint(args)
    manifest_path = args.manifest or deep_get(config, "data.manifest")
    if not manifest_path:
        raise SystemExit("train-retrieval requires --manifest or data.manifest in config.")
    pipeline = build_pipeline_from_args(args)
    model = _build_retrieval_model(args, pipeline)
    if args.checkpoint:
        load_retrieval_checkpoint(model, args.checkpoint, map_location=args.device)

    dataset = RetrievalManifestDataset(manifest_path, map_location=args.device, query_field=args.query_field)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=retrieval_collate,
    )
    optimizer = torch.optim.Adam((param for param in model.parameters() if param.requires_grad), lr=args.lr)
    history = train_retrieval_alignment(
        model,
        dataloader,
        optimizer,
        epochs=args.epochs,
        grad_clip_norm=args.grad_clip_norm,
        log_every=args.log_every,
    )
    output_path = args.output_path or deep_get(config, "retrieval.output_path")
    if not output_path:
        raise SystemExit("train-retrieval requires --output-path or retrieval.output_path in config.")
    save_retrieval_checkpoint(model, output_path, history=history, config=config)

    eval_payload = None
    valid_manifest = args.valid_manifest or deep_get(config, "data.valid_manifest")
    if valid_manifest:
        eval_dataset = RetrievalManifestDataset(valid_manifest, map_location=args.device, query_field=args.query_field)
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=retrieval_collate,
        )
        eval_payload = evaluate_retrieval_alignment(model, eval_loader).to_dict()

    history_path = args.history_path
    if history_path is None:
        history_path = str(Path(output_path).with_suffix(".history.json"))
    summary = summarize_metric_history(history)
    _write_json(history_path, {"history": history, "summary": summary, "eval": eval_payload})
    _write_json(str(Path(history_path).with_suffix(".summary.json")), summary)

    print(
        json.dumps(
            {
                "manifest": manifest_path,
                "valid_manifest": valid_manifest,
                "checkpoint": output_path,
                "history_path": history_path,
                "history_summary": summary,
                "eval": eval_payload,
                "retrieval_key_dim": model.config.key_dim,
                "query_max_tokens": model.config.query_max_tokens,
                "temperature": model.config.temperature,
                "run_name": args.run_name or "train-retrieval",
                "seed": args.seed,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def run_eval_retrieval(args) -> None:
    import torch
    from torch.utils.data import DataLoader
    from ssamem.retrieval_training import (
        RetrievalManifestDataset,
        evaluate_retrieval_alignment,
        load_retrieval_checkpoint,
        retrieval_collate,
    )

    torch.manual_seed(args.seed)
    config = load_config_file(args.config) if args.config else {}
    args = _apply_retrieval_config(args, config)
    _fill_retrieval_config_from_checkpoint(args)
    manifest_path = args.manifest or deep_get(config, "data.valid_manifest") or deep_get(config, "data.manifest")
    if not manifest_path:
        raise SystemExit("eval-retrieval requires --manifest or data.manifest/data.valid_manifest in config.")
    if not args.checkpoint:
        raise SystemExit("eval-retrieval requires --checkpoint.")

    pipeline = build_pipeline_from_args(args)
    model = _build_retrieval_model(args, pipeline)
    load_retrieval_checkpoint(model, args.checkpoint, map_location=args.device)

    query_dataset = RetrievalManifestDataset(manifest_path, map_location=args.device, query_field=args.query_field)
    query_loader = DataLoader(
        query_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=retrieval_collate,
    )
    candidate_loader = None
    if args.candidate_manifest:
        candidate_dataset = RetrievalManifestDataset(
            args.candidate_manifest,
            map_location=args.device,
            query_field=args.query_field,
        )
        candidate_loader = DataLoader(
            candidate_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=retrieval_collate,
        )

    result = evaluate_retrieval_alignment(model, query_loader, candidate_loader).to_dict()
    payload = {
        "manifest": manifest_path,
        "candidate_manifest": args.candidate_manifest,
        "checkpoint": args.checkpoint,
        "metrics": result,
        "retrieval_key_dim": model.config.key_dim,
        "query_max_tokens": model.config.query_max_tokens,
        "temperature": model.config.temperature,
    }
    if args.output_path:
        _write_json(args.output_path, payload)
        payload["output_path"] = args.output_path
    print(json.dumps(payload, ensure_ascii=False, indent=2))
