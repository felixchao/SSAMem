from __future__ import annotations

import json

from ssamem.commands.common import (
    _build_ssa_distiller_for_eval,
    _write_json,
    build_training_args_from_config,
    deep_get,
    load_config_file,
    resolve_output_path,
)


def run_eval_mas_latent_memory(args) -> None:
    import torch
    from ssamem.workflows.experience import evaluate_mas_latent_memory
    from ssamem.training.space import SSAManifestDataset, load_alignment_checkpoint

    torch.manual_seed(args.seed)
    config = load_config_file(args.config) if args.config else {}
    args = build_training_args_from_config(config, args) if config else args
    manifest_path = args.manifest or deep_get(config, "data.manifest")
    if not manifest_path:
        raise SystemExit("eval-mas-latent-memory requires --manifest or data.manifest in config.")

    pipeline, distiller = _build_ssa_distiller_for_eval(args)
    load_alignment_checkpoint(distiller, args.checkpoint, map_location=args.device)
    dataset = SSAManifestDataset(manifest_path, map_location=args.device)
    payload = evaluate_mas_latent_memory(
        pipeline=pipeline,
        distiller=distiller,
        dataset=dataset,
        checkpoint=args.checkpoint,
        manifest=manifest_path,
        mas_style=args.mas_style,
        task_domain=args.task_domain,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
    )
    output_path = resolve_output_path(args)
    if output_path:
        _write_json(output_path, payload)
        payload["output_path"] = output_path
    print(json.dumps(payload, ensure_ascii=False, indent=2))

def run_build_experience_bank(args) -> None:
    import torch
    from ssamem.workflows.experience import preload_experience_bank
    from ssamem.training.space import SSAManifestDataset, load_alignment_checkpoint

    torch.manual_seed(args.seed)
    config = load_config_file(args.config) if args.config else {}
    args = build_training_args_from_config(config, args) if config else args
    manifest_path = args.manifest or deep_get(config, "data.manifest")
    if not manifest_path:
        raise SystemExit("build-experience-bank requires --manifest or data.manifest in config.")
    pipeline, distiller = _build_ssa_distiller_for_eval(args)
    load_alignment_checkpoint(distiller, args.checkpoint, map_location=args.device)
    dataset = SSAManifestDataset(manifest_path, map_location=args.device)
    pointer_rows = preload_experience_bank(
        pipeline,
        distiller,
        dataset,
        limit=args.limit,
        cluster_method=args.cluster_method,
        kmeans_clusters=args.kmeans_clusters,
        kmeans_max_iter=args.kmeans_max_iter,
        seed=args.seed,
    )
    payload = {
        "manifest": manifest_path,
        "checkpoint": args.checkpoint,
        "count": len(pointer_rows),
        "clusters": len(pipeline.kernel.memory_agent.memory_clusters),
        "memories": len(pipeline.kernel.memory_agent.memory_store),
        "cluster_method": args.cluster_method,
        "kmeans_clusters": args.kmeans_clusters if args.cluster_method == "offline-kmeans" else None,
        "pointer_table": pointer_rows,
    }
    output_path = resolve_output_path(args)
    if output_path:
        _write_json(output_path, payload)
        payload["output_path"] = output_path
    pointer_table_path = getattr(args, "pointer_table_path", None)
    if pointer_table_path:
        _write_json(pointer_table_path, {"pointer_table": pointer_rows})
        payload["pointer_table_path"] = pointer_table_path
    print(json.dumps(payload, ensure_ascii=False, indent=2))

def run_eval_mas_memory_search(args) -> None:
    import torch
    from ssamem.workflows.experience import evaluate_mas_memory_search
    from ssamem.training.space import SSAManifestDataset, load_alignment_checkpoint

    torch.manual_seed(args.seed)
    config = load_config_file(args.config) if args.config else {}
    args = build_training_args_from_config(config, args) if config else args
    pipeline, distiller = _build_ssa_distiller_for_eval(args)
    load_alignment_checkpoint(distiller, args.checkpoint, map_location=args.device)
    bank_dataset = SSAManifestDataset(args.bank_manifest, map_location=args.device)
    eval_dataset = SSAManifestDataset(args.eval_manifest, map_location=args.device)
    payload = evaluate_mas_memory_search(
        pipeline=pipeline,
        distiller=distiller,
        bank_dataset=bank_dataset,
        eval_dataset=eval_dataset,
        bank_manifest=args.bank_manifest,
        eval_manifest=args.eval_manifest,
        checkpoint=args.checkpoint,
        bank_limit=args.bank_limit,
        limit=args.limit,
        top_k_prefetch=args.top_k_prefetch,
        cluster_method=args.cluster_method,
        kmeans_clusters=args.kmeans_clusters,
        kmeans_max_iter=args.kmeans_max_iter,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        mas_style=args.mas_style,
        task_domain=args.task_domain,
    )
    output_path = resolve_output_path(args)
    if output_path:
        _write_json(output_path, payload)
        payload["output_path"] = output_path
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def run_eval_mas_memory_agent_loop(args) -> None:
    import torch
    from ssamem.training.space import SSAManifestDataset, load_alignment_checkpoint
    from ssamem.workflows.experience import evaluate_mas_memory_agent_loop

    torch.manual_seed(args.seed)
    config = load_config_file(args.config) if args.config else {}
    args = build_training_args_from_config(config, args) if config else args
    pipeline, distiller = _build_ssa_distiller_for_eval(args)
    load_alignment_checkpoint(distiller, args.checkpoint, map_location=args.device)
    bank_dataset = SSAManifestDataset(args.bank_manifest, map_location=args.device)
    eval_dataset = SSAManifestDataset(args.eval_manifest, map_location=args.device)
    payload = evaluate_mas_memory_agent_loop(
        pipeline=pipeline,
        distiller=distiller,
        bank_dataset=bank_dataset,
        eval_dataset=eval_dataset,
        bank_manifest=args.bank_manifest,
        eval_manifest=args.eval_manifest,
        checkpoint=args.checkpoint,
        bank_limit=args.bank_limit,
        limit=args.limit,
        top_k=args.top_k,
        memory_request_policy=args.memory_request_policy,
        cluster_method=args.cluster_method,
        kmeans_clusters=args.kmeans_clusters,
        kmeans_max_iter=args.kmeans_max_iter,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        mas_style=args.mas_style,
        task_domain=args.task_domain,
    )
    output_path = resolve_output_path(args)
    if output_path:
        _write_json(output_path, payload)
        payload["output_path"] = output_path
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def run_query_memory_agent(args) -> None:
    import torch
    from ssamem.training.space import SSAManifestDataset, load_alignment_checkpoint
    from ssamem.workflows.experience import preload_experience_bank

    if not args.query and not args.address:
        raise SystemExit("query-memory-agent requires --query for SEARCH or --address for GET.")

    torch.manual_seed(args.seed)
    config = load_config_file(args.config) if args.config else {}
    args = build_training_args_from_config(config, args) if config else args
    manifest_path = args.manifest or deep_get(config, "data.manifest")
    if not manifest_path:
        raise SystemExit("query-memory-agent requires --manifest or data.manifest in config.")

    pipeline, distiller = _build_ssa_distiller_for_eval(args)
    load_alignment_checkpoint(distiller, args.checkpoint, map_location=args.device)
    dataset = SSAManifestDataset(manifest_path, map_location=args.device)
    pointer_rows = preload_experience_bank(
        pipeline,
        distiller,
        dataset,
        limit=args.limit,
        cluster_method=args.cluster_method,
        kmeans_clusters=args.kmeans_clusters,
        kmeans_max_iter=args.kmeans_max_iter,
        seed=args.seed,
    )

    payload = {
        "manifest": manifest_path,
        "checkpoint": args.checkpoint,
        "bank_count": len(pointer_rows),
        "cluster_method": args.cluster_method,
        "kmeans_clusters": args.kmeans_clusters if args.cluster_method == "offline-kmeans" else None,
        "pointer_table": pipeline.kernel.memory_agent.pointer_table_rows(max_entries=args.limit),
    }

    if args.address:
        memory = pipeline.get_memory(args.address)
        payload["mode"] = "GET"
        payload["address"] = args.address
        payload["memory"] = {
            "local_index": memory.local_index,
            "summary": memory.memory_summary,
            "target_text": memory.metadata.get("target_text") or memory.latent_tensor.metadata.get("target_text"),
            "metadata": memory.metadata,
            "latent_shape": list(memory.latent_tensor.tensor_data.shape),
        }
    else:
        hits = pipeline.search_memory(args.query, top_k=args.top_k)
        payload["mode"] = "SEARCH"
        payload["query"] = args.query
        payload["hits"] = [
            {
                "address": hit.exact_address,
                "pointer": hit.pointer,
                "score": hit.score,
                "cluster_id": hit.cluster_id,
                "local_index": hit.local_index,
                "target_text": hit.latent.metadata.get("target_text"),
                "metadata": hit.latent.metadata,
                "latent_shape": list(hit.latent.tensor_data.shape),
            }
            for hit in hits
        ]

    output_path = resolve_output_path(args)
    if output_path:
        _write_json(output_path, payload)
        payload["output_path"] = output_path
    print(json.dumps(payload, ensure_ascii=False, indent=2))
