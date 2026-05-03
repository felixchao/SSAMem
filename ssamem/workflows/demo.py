from __future__ import annotations

"""Small standalone demo workflows used by the CLI."""

import json


def run_demo(args, *, build_pipeline_from_args) -> None:
    import torch

    pipeline = build_pipeline_from_args(args)
    torch.manual_seed(args.seed)
    pointer_a = pipeline.register_memory_tensor(
        torch.randn(6, pipeline.userspace.hidden_size),
        metadata={"topic": "young-player-award", "source": "demo-memory-a"},
    )
    pointer_b = pipeline.register_memory_tensor(
        torch.randn(6, pipeline.userspace.hidden_size),
        metadata={"topic": "football-history", "source": "demo-memory-b"},
    )
    result = pipeline.run_message(
        sender_id="planner",
        receiver_id="userspace_role_1",
        content=f"請參考 {pointer_a} 回答問題：2006年第一位獲得最佳年輕球員獎的是誰？",
        top_k_prefetch=args.top_k_prefetch,
    )
    consolidation = pipeline.consolidate_episode(
        hidden_states=torch.randn(12, pipeline.userspace.hidden_size),
        global_reward=1.0,
        metadata={"episode": "demo"},
    )
    print(
        json.dumps(
            {
                "registered_pointers": [pointer_a, pointer_b],
                "mounted_pointers": result.mounted_pointer_ids,
                "userspace_prompt": result.prompt,
                "userspace_output": result.output_text,
                "consolidated_pointer": consolidation.pointer,
                "prefetched_count": len(result.resolved_message.prefetched_hits),
                "storage_root": args.storage_root,
                "run_name": args.run_name or "demo",
                "seed": args.seed,
                "output_dir": args.output_dir,
                "log_level": args.log_level,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def build_demo_mas_scenario(task_domain: str | None) -> tuple[str, str, dict]:
    scenarios = {
        "triviaqa": (
            "Question: Who was the first winner of the 2006 Best Young Player Award?",
            "Lionel Messi won the FIFPro Best Young Player Award in 2006. He was recognized as the first winner of that award.",
            {"topic": "triviaqa-demo", "question_type": "factoid"},
        ),
        "popqa": (
            "Question: What is the capital of Australia?",
            "The capital city of Australia is Canberra. It is not Sydney or Melbourne.",
            {"topic": "popqa-demo", "question_type": "factoid"},
        ),
        "pddl": (
            "Plan a valid sequence of actions to move block A onto block B.",
            "Goal: move block A onto block B. Initial state: A is on the table, B is clear, the robot hand is empty.",
            {"topic": "pddl-demo", "task_type": "planning"},
        ),
        "kodcode": (
            "Write Python code for a function `first_unique_char(s: str) -> str` that returns the first non-repeating character.",
            "Implement a Python function that returns the first non-repeating character in a string. If no such character exists, return an empty string.",
            {"topic": "kodcode-demo", "task_type": "coding"},
        ),
        "alfworld": (
            "Choose the next valid ALFWorld action to make progress on heating the mug.",
            "You are in a kitchen. The mug is on the counter. The microwave is closed. The task is to heat the mug with the microwave.",
            {"topic": "alfworld-demo", "task_type": "action"},
        ),
    }
    if task_domain in scenarios:
        return scenarios[task_domain]
    return (
        "Please refer to the mounted memory, plan a response strategy, and provide the final answer.",
        "This memory page contains a generic demo context for multi-agent coordination.",
        {"topic": "demo-mas", "task_type": "generic"},
    )


def run_demo_mas(args, *, build_pipeline_from_args) -> None:
    import torch

    pipeline = build_pipeline_from_args(args)
    torch.manual_seed(args.seed)
    task_description, evidence_text, metadata = build_demo_mas_scenario(args.task_domain)
    latent_tensor = (
        pipeline.userspace.encode_text_as_latent(evidence_text, max_tokens=128, strategy="last_hidden")
        if args.task_domain
        else torch.randn(6, pipeline.userspace.hidden_size)
    )
    pointer = pipeline.register_memory_tensor(latent_tensor, metadata=metadata)
    trace = pipeline.run_multi_agent_task(
        task_description=f"Please use {pointer} when helpful.\n\n{task_description}",
        mas_style=args.mas_style,
        task_domain=args.task_domain,
        top_k_prefetch=args.top_k_prefetch,
    )
    print(
        json.dumps(
            {
                "architecture": trace.architecture,
                "task_domain": args.task_domain,
                "task_description": task_description,
                "registered_pointer": pointer,
                "turn_count": len(trace.turns),
                "final_output": trace.final_output,
                "roles": [turn.role for turn in trace.turns],
                "run_name": args.run_name or "demo-mas",
                "seed": args.seed,
                "output_dir": args.output_dir,
                "log_level": args.log_level,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
