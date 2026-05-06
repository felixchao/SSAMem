from __future__ import annotations

"""Experience-bank preloading and MAS memory-evaluation workflows."""

from typing import Any

import torch
from transformers import GenerationConfig

from ssamem.data_models import LatentTensor
from ssamem.training.space import SSAManifestDataset, _resolve_latent_path, target_text_hit


def _summarize_binary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0}
    return {"count": len(values), "mean": sum(values) / len(values)}


def _registered_address(pipeline, pointer: str, *, sample_index: int) -> tuple[str, int, int]:
    cluster = pipeline.kernel.memory_agent.get_cluster_by_pointer(pointer)
    matching = [memory for memory in cluster.memories if memory.metadata.get("sample_index") == sample_index]
    local_index = matching[-1].local_index if matching else max(memory.local_index for memory in cluster.memories)
    return f"{pointer}:{local_index:04d}", local_index, cluster.cluster_size


def _project_experience_latents(
    pipeline,
    distiller,
    dataset: SSAManifestDataset,
    *,
    limit: int,
    retrieval_model=None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for idx, sample in enumerate(dataset.samples):
        if idx >= limit:
            break
        latent_path = _resolve_latent_path(sample, dataset.root_dir)
        latent_tensor = torch.load(latent_path, map_location=pipeline.userspace.device)
        with torch.no_grad():
            projected_latent = distiller.student.project_latents([sample.student_text()], [latent_tensor])[0].detach()
        prepared_tensor = pipeline.kernel.memory_agent._prepare_tensor_data(projected_latent)
        if retrieval_model is None:
            key_vector = pipeline.kernel.memory_agent.build_key_vector(prepared_tensor)
        else:
            with torch.no_grad():
                # Retrieval alignment is trained against the original trajectory latent,
                # while prompt injection uses the composer-projected latent.
                key_vector = retrieval_model.encode_memories([latent_tensor])[0].detach().float().cpu()
        records.append(
            {
                "sample_index": idx,
                "sample": sample,
                "latent_path": latent_path,
                "latent": LatentTensor(
                    tensor_data=prepared_tensor,
                    key_vector=key_vector,
                    utility_score=1.0,
                    metadata={
                        "source": "experience_bank",
                        "sample_index": idx,
                        "task_prompt": sample.task_prompt,
                        "target_text": sample.target_text,
                        "latent_tensor_path": str(latent_path),
                        "topic": sample.task_prompt,
                        "summary": sample.task_prompt,
                    },
                ),
            }
        )
    return records


def _pointer_row_for_record(pipeline, record: dict[str, Any], pointer: str) -> dict[str, Any]:
    sample = record["sample"]
    idx = int(record["sample_index"])
    address, local_index, cluster_size = _registered_address(pipeline, pointer, sample_index=idx)
    return {
        "sample_index": idx,
        "key": sample.task_prompt,
        "target_text": sample.target_text,
        "pointer": pointer,
        "address": address,
        "local_index": local_index,
        "cluster_size": cluster_size,
        "latent_tensor_path": str(record["latent_path"]),
    }


def _kmeans_assignments(
    key_vectors: torch.Tensor,
    *,
    num_clusters: int,
    max_iter: int = 50,
    seed: int = 7,
) -> tuple[torch.Tensor, torch.Tensor]:
    count = int(key_vectors.size(0))
    if count == 0:
        return torch.empty(0, dtype=torch.long), torch.empty(0, key_vectors.size(-1))
    k = max(1, min(int(num_clusters), count))
    vectors = torch.nn.functional.normalize(key_vectors.float(), dim=-1)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    init_indices = torch.randperm(count, generator=generator)[:k].to(vectors.device)
    centroids = vectors[init_indices].clone()
    assignments = torch.full((count,), -1, dtype=torch.long, device=vectors.device)
    for _ in range(max(int(max_iter), 1)):
        next_assignments = (vectors @ centroids.T).argmax(dim=1)
        if torch.equal(next_assignments, assignments):
            assignments = next_assignments
            break
        assignments = next_assignments
        for cluster_index in range(k):
            mask = assignments == cluster_index
            if bool(mask.any()):
                centroids[cluster_index] = torch.nn.functional.normalize(vectors[mask].mean(dim=0), dim=0)
    return assignments.cpu(), centroids.detach().cpu()


def _preload_experience_bank_online(pipeline, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pointer_rows: list[dict[str, Any]] = []
    for record in records:
        pointer = pipeline.kernel.memory_agent.register_latent(record["latent"])
        pointer_rows.append(_pointer_row_for_record(pipeline, record, pointer))
    return pointer_rows


def _preload_experience_bank_kmeans(
    pipeline,
    records: list[dict[str, Any]],
    *,
    kmeans_clusters: int,
    kmeans_max_iter: int,
    seed: int,
) -> list[dict[str, Any]]:
    if not records:
        return []
    key_vectors = torch.stack([record["latent"].key_vector.detach().cpu().float() for record in records], dim=0)
    assignments, centroids = _kmeans_assignments(
        key_vectors,
        num_clusters=kmeans_clusters,
        max_iter=kmeans_max_iter,
        seed=seed,
    )
    pointer_rows: list[dict[str, Any]] = []
    for cluster_index in range(int(centroids.size(0))):
        member_indices = [idx for idx, assignment in enumerate(assignments.tolist()) if assignment == cluster_index]
        if not member_indices:
            continue
        centroid = centroids[cluster_index]
        member_vectors = key_vectors[member_indices]
        scores = torch.nn.functional.normalize(member_vectors, dim=-1) @ torch.nn.functional.normalize(centroid, dim=0)
        representative_member = member_indices[int(scores.argmax().item())]
        ordered_indices = [representative_member] + [idx for idx in member_indices if idx != representative_member]
        latents = [records[idx]["latent"] for idx in ordered_indices]
        representative = records[representative_member]["sample"]
        cluster = pipeline.kernel.memory_agent.register_latent_cluster(
            latents,
            summary_key_text=representative.task_prompt,
            summary_key_embedding=centroid.to(latents[0].key_vector.device, dtype=latents[0].key_vector.dtype),
            metadata={
                "source": "experience_bank",
                "cluster_method": "offline-kmeans",
                "kmeans_cluster_index": cluster_index,
            },
        )
        for idx in ordered_indices:
            pointer_rows.append(_pointer_row_for_record(pipeline, records[idx], cluster.pointer_id or ""))
    return sorted(pointer_rows, key=lambda row: int(row["sample_index"]))


def preload_experience_bank(
    pipeline,
    distiller,
    dataset: SSAManifestDataset,
    *,
    limit: int,
    cluster_method: str = "online",
    kmeans_clusters: int = 50,
    kmeans_max_iter: int = 50,
    seed: int = 7,
    retrieval_model=None,
) -> list[dict[str, Any]]:
    records = _project_experience_latents(
        pipeline,
        distiller,
        dataset,
        limit=limit,
        retrieval_model=retrieval_model,
    )
    if cluster_method == "online":
        return _preload_experience_bank_online(pipeline, records)
    if cluster_method == "offline-kmeans":
        return _preload_experience_bank_kmeans(
            pipeline,
            records,
            kmeans_clusters=kmeans_clusters,
            kmeans_max_iter=kmeans_max_iter,
            seed=seed,
        )
    raise ValueError(f"Unsupported cluster_method: {cluster_method}")


def memory_agent_protocol_context(pipeline, *, max_entries: int | None = None) -> str:
    return (
        f"{pipeline.pointer_table_context(max_entries=max_entries)}\n\n"
        "[Memory Agent Protocol]\n"
        "SEARCH: choose a pointer by comparing the task with summary keys, then retrieve the most relevant latent memories.\n"
        "GET: if an exact address is already known, request it directly, e.g. <PTR_0x001>:0000.\n"
        "The mounted latent memory is injected invisibly as soft prompt vectors; cite only the final answer, not the pointer."
    )


def evaluate_mas_latent_memory(
    *,
    pipeline,
    distiller,
    dataset: SSAManifestDataset,
    checkpoint: str,
    manifest: str,
    mas_style: str,
    task_domain: str | None,
    limit: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    generation_config = GenerationConfig(
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=pipeline.userspace.tokenizer.pad_token_id,
        eos_token_id=pipeline.userspace.tokenizer.eos_token_id,
    )

    examples: list[dict[str, Any]] = []
    no_hits: list[float] = []
    text_hits: list[float] = []
    latent_hits: list[float] = []
    for idx, sample in enumerate(dataset.samples):
        if idx >= limit:
            break
        latent_path = _resolve_latent_path(sample, dataset.root_dir)
        latent_tensor = torch.load(latent_path, map_location=pipeline.userspace.device)
        with torch.no_grad():
            projected_latent = distiller.student.project_latents([sample.student_text()], [latent_tensor])[0].detach()
        pointer = pipeline.register_memory_tensor(
            projected_latent,
            utility_score=1.0,
            metadata={"source": "eval-mas-latent-memory", "sample_index": idx},
        )
        address, local_index, cluster_size = _registered_address(pipeline, pointer, sample_index=idx)

        no_trace = pipeline.run_multi_agent_task(
            task_description=sample.task_prompt,
            mas_style=mas_style,
            task_domain=task_domain,
            memory_content="",
            generation_config=generation_config,
            top_k_prefetch=0,
        )
        text_trace = pipeline.run_multi_agent_task(
            task_description=sample.task_prompt,
            mas_style=mas_style,
            task_domain=task_domain,
            memory_content=sample.context_text,
            generation_config=generation_config,
            top_k_prefetch=0,
        )
        latent_trace = pipeline.run_multi_agent_task(
            task_description=sample.task_prompt,
            mas_style=mas_style,
            task_domain=task_domain,
            memory_content=address,
            generation_config=generation_config,
            top_k_prefetch=0,
        )

        no_output = no_trace.final_output or ""
        text_output = text_trace.final_output or ""
        latent_output = latent_trace.final_output or ""
        no_hit = target_text_hit(sample.target_text, no_output)
        text_hit = target_text_hit(sample.target_text, text_output)
        latent_hit = target_text_hit(sample.target_text, latent_output)
        no_hits.append(float(no_hit))
        text_hits.append(float(text_hit))
        latent_hits.append(float(latent_hit))
        examples.append(
            {
                "index": idx,
                "task_prompt": sample.task_prompt,
                "target_text": sample.target_text,
                "registered_pointer": pointer,
                "registered_address": address,
                "registered_local_index": local_index,
                "registered_cluster_size": cluster_size,
                "no_memory_final": no_output,
                "text_memory_final": text_output,
                "latent_memory_final": latent_output,
                "no_memory_target_hit": no_hit,
                "text_memory_target_hit": text_hit,
                "latent_memory_target_hit": latent_hit,
                "latent_turns": [_turn_summary(turn) for turn in latent_trace.turns],
            }
        )

    return {
        "manifest": manifest,
        "checkpoint": checkpoint,
        "checkpoint_loaded": True,
        "mas_style": mas_style,
        "task_domain": task_domain,
        "count": len(examples),
        "target_hit": {
            "no_memory": _summarize_binary(no_hits),
            "text_memory": _summarize_binary(text_hits),
            "latent_memory": _summarize_binary(latent_hits),
        },
        "examples": examples,
    }


def evaluate_mas_memory_search(
    *,
    pipeline,
    distiller,
    bank_dataset: SSAManifestDataset,
    eval_dataset: SSAManifestDataset,
    bank_manifest: str,
    eval_manifest: str,
    checkpoint: str,
    bank_limit: int,
    limit: int,
    top_k_prefetch: int,
    max_new_tokens: int,
    mas_style: str,
    task_domain: str | None,
    cluster_method: str = "online",
    kmeans_clusters: int = 50,
    kmeans_max_iter: int = 50,
    seed: int = 7,
    retrieval_model=None,
) -> dict[str, Any]:
    pointer_rows = preload_experience_bank(
        pipeline,
        distiller,
        bank_dataset,
        limit=bank_limit,
        cluster_method=cluster_method,
        kmeans_clusters=kmeans_clusters,
        kmeans_max_iter=kmeans_max_iter,
        seed=seed,
        retrieval_model=retrieval_model,
    )
    generation_config = GenerationConfig(
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=pipeline.userspace.tokenizer.pad_token_id,
        eos_token_id=pipeline.userspace.tokenizer.eos_token_id,
    )

    examples: list[dict[str, Any]] = []
    no_hits: list[float] = []
    text_hits: list[float] = []
    search_hits: list[float] = []
    retrieval_hits: list[float] = []
    for idx, sample in enumerate(eval_dataset.samples):
        if idx >= limit:
            break
        no_trace = pipeline.run_multi_agent_task(
            task_description=sample.task_prompt,
            mas_style=mas_style,
            task_domain=task_domain,
            memory_content="",
            generation_config=generation_config,
            top_k_prefetch=0,
        )
        text_trace = pipeline.run_multi_agent_task(
            task_description=sample.task_prompt,
            mas_style=mas_style,
            task_domain=task_domain,
            memory_content=sample.context_text,
            generation_config=generation_config,
            top_k_prefetch=0,
        )
        search_trace = pipeline.run_multi_agent_task(
            task_description=sample.task_prompt,
            mas_style=mas_style,
            task_domain=task_domain,
            memory_content=memory_agent_protocol_context(pipeline, max_entries=bank_limit),
            include_pointer_table=False,
            generation_config=generation_config,
            top_k_prefetch=top_k_prefetch,
        )

        retrieved_addresses = [address for turn in search_trace.turns for address in turn.mounted_pointer_ids]
        retrieved_targets = _retrieved_targets(pipeline, retrieved_addresses)
        retrieval_hit = any(str(target) == str(sample.target_text) for target in retrieved_targets)
        no_output = no_trace.final_output or ""
        text_output = text_trace.final_output or ""
        search_output = search_trace.final_output or ""
        no_hit = target_text_hit(sample.target_text, no_output)
        text_hit = target_text_hit(sample.target_text, text_output)
        search_hit = target_text_hit(sample.target_text, search_output)
        no_hits.append(float(no_hit))
        text_hits.append(float(text_hit))
        search_hits.append(float(search_hit))
        retrieval_hits.append(float(retrieval_hit))
        examples.append(
            {
                "index": idx,
                "task_prompt": sample.task_prompt,
                "target_text": sample.target_text,
                "no_memory_final": no_output,
                "text_memory_final": text_output,
                "search_memory_final": search_output,
                "no_memory_target_hit": no_hit,
                "text_memory_target_hit": text_hit,
                "search_memory_target_hit": search_hit,
                "retrieval_target_hit": retrieval_hit,
                "retrieved_addresses": retrieved_addresses,
                "retrieved_targets": retrieved_targets,
                "search_turns": [_turn_summary(turn) for turn in search_trace.turns],
            }
        )

    return {
        "bank_manifest": bank_manifest,
        "eval_manifest": eval_manifest,
        "checkpoint": checkpoint,
        "bank_count": len(pointer_rows),
        "bank_clusters": len(pipeline.kernel.memory_agent.memory_clusters),
        "cluster_method": cluster_method,
        "kmeans_clusters": kmeans_clusters if cluster_method == "offline-kmeans" else None,
        "retrieval_checkpoint_loaded": retrieval_model is not None,
        "pointer_table": pipeline.kernel.memory_agent.pointer_table_rows(max_entries=bank_limit),
        "memory_agent_protocol": "SEARCH over summary keys when no exact address is known; GET by exact address when provided.",
        "count": len(examples),
        "target_hit": {
            "no_memory": _summarize_binary(no_hits),
            "text_memory": _summarize_binary(text_hits),
            "search_memory": _summarize_binary(search_hits),
        },
        "retrieval_target_hit": _summarize_binary(retrieval_hits),
        "examples": examples,
    }


def evaluate_mas_memory_agent_loop(
    *,
    pipeline,
    distiller,
    bank_dataset: SSAManifestDataset,
    eval_dataset: SSAManifestDataset,
    bank_manifest: str,
    eval_manifest: str,
    checkpoint: str,
    bank_limit: int,
    limit: int,
    top_k: int,
    memory_request_policy: str,
    max_new_tokens: int,
    mas_style: str,
    task_domain: str | None,
    cluster_method: str = "online",
    kmeans_clusters: int = 50,
    kmeans_max_iter: int = 50,
    seed: int = 7,
    retrieval_model=None,
) -> dict[str, Any]:
    pointer_rows = preload_experience_bank(
        pipeline,
        distiller,
        bank_dataset,
        limit=bank_limit,
        cluster_method=cluster_method,
        kmeans_clusters=kmeans_clusters,
        kmeans_max_iter=kmeans_max_iter,
        seed=seed,
        retrieval_model=retrieval_model,
    )
    generation_config = GenerationConfig(
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=pipeline.userspace.tokenizer.pad_token_id,
        eos_token_id=pipeline.userspace.tokenizer.eos_token_id,
    )
    request_generation_config = GenerationConfig(
        do_sample=False,
        max_new_tokens=48,
        pad_token_id=pipeline.userspace.tokenizer.pad_token_id,
        eos_token_id=pipeline.userspace.tokenizer.eos_token_id,
    )

    examples: list[dict[str, Any]] = []
    loop_hits: list[float] = []
    retrieval_hits: list[float] = []
    request_counts = {"SEARCH": 0, "GET": 0, "NONE": 0}
    for idx, sample in enumerate(eval_dataset.samples):
        if idx >= limit:
            break
        loop_trace = pipeline.run_multi_agent_task_with_memory_actions(
            task_description=sample.task_prompt,
            mas_style=mas_style,
            task_domain=task_domain,
            memory_content="[Use the MemoryAgent if needed.]",
            pointer_table_max_entries=bank_limit,
            generation_config=generation_config,
            request_generation_config=request_generation_config,
            default_top_k=top_k,
            memory_request_policy=memory_request_policy,
        )
        retrieved_addresses = [address for turn in loop_trace.turns for address in turn.mounted_pointer_ids]
        retrieved_targets = _retrieved_targets(pipeline, retrieved_addresses)
        retrieval_hit = any(str(target) == str(sample.target_text) for target in retrieved_targets)
        loop_output = loop_trace.final_output or ""
        loop_hit = target_text_hit(sample.target_text, loop_output)
        loop_hits.append(float(loop_hit))
        retrieval_hits.append(float(retrieval_hit))
        for turn in loop_trace.turns:
            mode = (turn.memory_request or {}).get("mode", "NONE")
            request_counts[mode] = request_counts.get(mode, 0) + 1
        examples.append(
            {
                "index": idx,
                "task_prompt": sample.task_prompt,
                "target_text": sample.target_text,
                "agent_loop_final": loop_output,
                "agent_loop_target_hit": loop_hit,
                "retrieval_target_hit": retrieval_hit,
                "retrieved_addresses": retrieved_addresses,
                "retrieved_targets": retrieved_targets,
                "turns": [_turn_summary(turn) for turn in loop_trace.turns],
            }
        )

    return {
        "bank_manifest": bank_manifest,
        "eval_manifest": eval_manifest,
        "checkpoint": checkpoint,
        "bank_count": len(pointer_rows),
        "bank_clusters": len(pipeline.kernel.memory_agent.memory_clusters),
        "cluster_method": cluster_method,
        "kmeans_clusters": kmeans_clusters if cluster_method == "offline-kmeans" else None,
        "retrieval_checkpoint_loaded": retrieval_model is not None,
        "pointer_table": pipeline.kernel.memory_agent.pointer_table_rows(max_entries=bank_limit),
        "count": len(examples),
        "target_hit": {"agent_memory_loop": _summarize_binary(loop_hits)},
        "retrieval_target_hit": _summarize_binary(retrieval_hits),
        "memory_request_counts": request_counts,
        "memory_request_policy": memory_request_policy,
        "examples": examples,
    }


def _retrieved_targets(pipeline, addresses: list[str]) -> list[str | None]:
    targets: list[str | None] = []
    for address in addresses:
        pointer, _, local = address.partition(":")
        try:
            memory = pipeline.kernel.memory_agent.get_memory_by_address(pointer, int(local))
        except Exception:
            continue
        targets.append(memory.metadata.get("target_text") or memory.latent_tensor.metadata.get("target_text"))
    return targets


def _turn_summary(turn) -> dict[str, Any]:
    return {
        "role": turn.role,
        "mounted_pointer_ids": turn.mounted_pointer_ids,
        "mounted_latent_count": turn.mounted_latent_count,
        "mounted_latent_tokens": turn.mounted_latent_tokens,
        "memory_request": turn.memory_request,
        "memory_observation": turn.memory_observation,
        "request_response": turn.request_response,
        "response": turn.response,
    }
