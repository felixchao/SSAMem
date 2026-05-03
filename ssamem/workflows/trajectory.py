from __future__ import annotations

"""Pure-text MAS trajectory collection workflow."""

import json
from pathlib import Path
from typing import Any


def _row_task_prompt(row: dict[str, Any], idx: int) -> str:
    return str(
        row.get("task_prompt")
        or row.get("task_description")
        or row.get("prompt")
        or row.get("question")
        or row.get("instruction")
        or f"Task {idx}"
    ).strip()


def _row_context_text(row: dict[str, Any]) -> str:
    context = row.get("context_text") or row.get("context") or row.get("memory")
    if context:
        return str(context).strip()
    answer = row.get("answer") or row.get("answers") or row.get("possible_answers")
    if answer:
        return f"Known answer aliases: {json.dumps(answer, ensure_ascii=False)}"
    return ""


def _row_target_text(row: dict[str, Any]) -> str | None:
    for key in ("target_text", "target_agent_output", "final_answer", "answer", "answers", "possible_answers"):
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            return value if value.strip() else None
        if isinstance(value, (list, tuple)):
            return str(value[0]) if value else None
        return str(value)
    return None


def _target_hit(target: str | None, output: str | None) -> bool:
    import re

    if not target or not output:
        return False
    target_norm = re.sub(r"[^a-z0-9 ]+", " ", str(target).lower()).strip()
    output_norm = re.sub(r"[^a-z0-9 ]+", " ", str(output).lower())
    return bool(target_norm and target_norm in output_norm)


def _latentmem_like_message_graph(trace, *, context_text: str) -> dict[str, Any]:
    nodes = []
    links = []
    for turn in trace.turns:
        nodes.append(
            {
                "id": turn.role,
                "message": {
                    "system_prompt_template": None,
                    "system_prompt_fields": {},
                    "user_prompt_template": turn.prompt,
                    "user_prompt_fields": {
                        "memory_content": context_text,
                        "task_description": trace.task_description,
                    },
                    "response": turn.response,
                    "state": {
                        "role": turn.role,
                        "architecture": trace.architecture,
                        "mounted_latent_count": turn.mounted_latent_count,
                        "mounted_latent_tokens": turn.mounted_latent_tokens,
                    },
                },
            }
        )
        for upstream_role in turn.upstream_roles:
            links.append({"source": upstream_role, "target": turn.role})
    return {
        "state": trace.task_description,
        "action": trace.final_output or "",
        "observation": "",
        "mas_message_graph_data": json.dumps(
            {"directed": True, "multigraph": False, "graph": {}, "nodes": nodes, "links": links},
            ensure_ascii=False,
        ),
    }


def collect_text_mas_trajectories(args, *, build_pipeline_from_args) -> dict[str, Any]:
    import torch
    from transformers import GenerationConfig
    from ssamem.training.space import _jsonl_rows

    torch.manual_seed(args.seed)
    pipeline = build_pipeline_from_args(args)
    generation_config = GenerationConfig(
        do_sample=False,
        max_new_tokens=args.max_new_tokens,
        pad_token_id=pipeline.userspace.tokenizer.pad_token_id,
        eos_token_id=pipeline.userspace.tokenizer.eos_token_id,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    rewards: list[float] = []
    with output_path.open("w", encoding="utf-8") as handle:
        for idx, row in enumerate(_jsonl_rows(args.input)):
            if idx >= args.limit:
                break
            task_prompt = _row_task_prompt(row, idx)
            context_text = _row_context_text(row)
            target_text = _row_target_text(row)
            trace = pipeline.run_multi_agent_task(
                task_description=task_prompt,
                mas_style=args.mas_style,
                task_domain=args.task_domain,
                memory_content=context_text,
                generation_config=generation_config,
                top_k_prefetch=0,
            )
            final_output = trace.final_output or ""
            reward = 1.0 if _target_hit(target_text, final_output) else 0.0
            rewards.append(reward)
            handle.write(
                json.dumps(
                    {
                        "task_id": row.get("trace_id") or row.get("id") or idx,
                        "task_prompt": task_prompt,
                        "context_text": context_text,
                        "target_agent_output": target_text,
                        "final_answer": final_output,
                        "reward": reward,
                        "mas_style": args.mas_style,
                        "task_domain": args.task_domain,
                        "trajectory": [
                            {
                                "agent": turn.role,
                                "prompt": turn.prompt,
                                "output": turn.response,
                                "upstream_roles": turn.upstream_roles,
                            }
                            for turn in trace.turns
                        ],
                        "message_graph": _latentmem_like_message_graph(trace, context_text=context_text),
                        "metadata": {
                            "source": "text_mas_rollout",
                            "input_index": idx,
                            "input_metadata": row.get("metadata") or {},
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1
            if count == 1 or count % 10 == 0:
                print(f"collect-text-mas-trajectories count={count} reward_mean={sum(rewards) / len(rewards):.4g}", flush=True)
    return {
        "count": count,
        "output": str(output_path),
        "reward_mean": 0.0 if not rewards else sum(rewards) / len(rewards),
        "run_name": args.run_name or "collect-text-mas-trajectories",
    }
