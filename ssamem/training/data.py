from __future__ import annotations

"""SSA trace conversion and pointer-preference data helpers."""

import ast
from dataclasses import dataclass, field
import json
from pathlib import Path
import random
import re
from typing import Any


POINTER_TOKEN_PATTERN = re.compile(r"^<PTR_0x[0-9A-Fa-f]+>$")


@dataclass
class SSATraceRecord:
    task_prompt: str
    context_text: str
    answer_prefix: str = "The answer is:"
    target_text: str | None = None
    trace_id: str | int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json_row(self) -> dict[str, Any]:
        row = {
            "task_prompt": self.task_prompt,
            "context_text": self.context_text,
            "answer_prefix": self.answer_prefix,
            "metadata": self.metadata,
        }
        if self.target_text is not None:
            row["target_text"] = self.target_text
        if self.trace_id is not None:
            row["trace_id"] = self.trace_id
        return row


@dataclass
class PointerPreferenceSample:
    prompt: str
    chosen: str
    rejected: str
    chosen_reward: float = 1.0
    rejected_reward: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not POINTER_TOKEN_PATTERN.match(self.chosen):
            raise ValueError(f"`chosen` is not a valid pointer token: {self.chosen}")
        if not POINTER_TOKEN_PATTERN.match(self.rejected):
            raise ValueError(f"`rejected` is not a valid pointer token: {self.rejected}")
        if self.chosen == self.rejected:
            raise ValueError("`chosen` and `rejected` pointers must differ.")


def save_ssa_trace_records(records: list[SSATraceRecord], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_json_row(), ensure_ascii=False) + "\n")


def iter_jsonl_rows(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc


def save_pointer_preferences(samples: list[PointerPreferenceSample], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            sample.validate()
            handle.write(json.dumps(sample.__dict__, ensure_ascii=False) + "\n")


def pointer_preferences_from_rollouts(path: str | Path) -> list[PointerPreferenceSample]:
    samples: list[PointerPreferenceSample] = []
    for idx, row in enumerate(iter_jsonl_rows(path)):
        prompt = str(row.get("prompt") or row.get("task_prompt") or row.get("task_description") or "").strip()
        if not prompt:
            raise ValueError(f"Rollout row {idx} must include a prompt/task.")
        if row.get("chosen") and row.get("rejected"):
            sample = PointerPreferenceSample(
                prompt=prompt,
                chosen=str(row["chosen"]),
                rejected=str(row["rejected"]),
                chosen_reward=float(row.get("chosen_reward", 1.0)),
                rejected_reward=float(row.get("rejected_reward", 0.0)),
                metadata=dict(row.get("metadata") or {}),
            )
            sample.validate()
            samples.append(sample)
            continue
        candidates = list(row.get("candidates") or row.get("rollouts") or [])
        if len(candidates) < 2:
            raise ValueError(f"Rollout row {idx} must include at least two pointer candidates.")
        ranked = sorted(candidates, key=lambda item: float(item.get("reward", 0.0)), reverse=True)
        chosen_candidate = ranked[0]
        rejected_candidate = ranked[-1]
        sample = PointerPreferenceSample(
            prompt=prompt,
            chosen=str(chosen_candidate.get("pointer")),
            rejected=str(rejected_candidate.get("pointer")),
            chosen_reward=float(chosen_candidate.get("reward", 0.0)),
            rejected_reward=float(rejected_candidate.get("reward", 0.0)),
            metadata={"source_rollout_id": row.get("rollout_id", idx), **dict(row.get("metadata") or {})},
        )
        sample.validate()
        samples.append(sample)
    return samples


def collect_synthetic_pointer_preferences(
    *,
    prompts: list[str],
    pointers: list[str],
    positive_pointer: str | None = None,
) -> list[PointerPreferenceSample]:
    if len(pointers) < 2:
        raise ValueError("At least two pointers are required to build DPO preferences.")
    samples = []
    for idx, prompt in enumerate(prompts):
        chosen = positive_pointer or pointers[idx % len(pointers)]
        rejected_choices = [pointer for pointer in pointers if pointer != chosen]
        rejected = random.choice(rejected_choices)
        samples.append(
            PointerPreferenceSample(
                prompt=prompt,
                chosen=chosen,
                rejected=rejected,
                chosen_reward=1.0,
                rejected_reward=0.0,
                metadata={"collector": "synthetic"},
            )
        )
    return samples


def build_ssa_manifest_without_latents(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    default_answer_prefix: str = "The answer is:",
) -> int:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.jsonl"
    count = 0
    with manifest_path.open("w", encoding="utf-8") as handle:
        for idx, row in enumerate(iter_jsonl_rows(input_path)):
            task_prompt = str(row.get("task_prompt") or row.get("task_description") or row.get("prompt") or "").strip()
            context_text = str(row.get("context_text") or row.get("context") or row.get("memory") or "").strip()
            if not context_text and row.get("turns"):
                context_text = "\n".join(
                    f"{turn.get('role', 'agent')}: {turn.get('response') or turn.get('content') or ''}"
                    for turn in row["turns"]
                ).strip()
            if not context_text:
                context_text = _format_agent_trajectory(row)
            if not task_prompt or not context_text:
                raise ValueError(f"Trace row {idx} must provide task prompt and context text.")
            metadata = dict(row.get("metadata") or {})
            metadata.setdefault("source_trace_id", row.get("trace_id", idx))
            handle.write(
                json.dumps(
                    {
                        "task_prompt": task_prompt,
                        "context_text": context_text,
                        "answer_prefix": str(row.get("answer_prefix") or default_answer_prefix),
                        "target_text": _extract_target_text(row),
                        "latent_tensor_path": None,
                        "metadata": metadata,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1
    return count


def generate_synthetic_api_ssa_records(*, limit: int = 200, seed: int = 7) -> list[SSATraceRecord]:
    rng = random.Random(seed)
    resources = [
        ("tax", "tax/calculate", "subtotal: float, region: string", "total: float", "Apply region tax rate before rounding."),
        ("invoice", "invoice/total", "subtotal: float, discount_code?: string", "total: float", "VIP10 subtracts 10 percent before tax."),
        ("refund", "refund/estimate", "order_total: float, days_since_purchase: int", "refund_amount: float", "No refund after 30 days."),
        ("shipping", "shipping/quote", "weight_kg: float, country: string", "shipping_fee: float", "International shipping adds a customs surcharge."),
        ("auth", "auth/permission", "role: string, action: string", "allowed: bool", "Only admins may delete production resources."),
        ("inventory", "inventory/reserve", "sku: string, quantity: int", "reservation_id: string", "Reject requests above available stock."),
        ("pricing", "pricing/preview", "plan: string, seats: int", "monthly_price: float", "Enterprise plans require a sales contact."),
        ("ledger", "ledger/post", "account_id: string, amount: float", "entry_id: string", "Negative amount means debit."),
    ]
    rates = ["5%", "7.5%", "8%", "10%", "12%"]
    errors = ["400 on missing required fields", "403 on permission failure", "409 on conflicting state", "422 on invalid values"]
    records: list[SSATraceRecord] = []
    for idx in range(limit):
        domain, endpoint, inputs, output, rule = rng.choice(resources)
        rate = rng.choice(rates)
        error_rule = rng.choice(errors)
        records.append(
            SSATraceRecord(
                task_prompt=f"Worker must implement POST /{endpoint}. Use the mounted memory page for the exact contract.",
                context_text=(
                    f"API contract for /{endpoint}.\n"
                    f"Domain: {domain}.\n"
                    "HTTP method: POST.\n"
                    f"Input schema: {inputs}.\n"
                    f"Output schema: {output}.\n"
                    f"Business rule: {rule}\n"
                    f"Tax or service rate when applicable: {rate}.\n"
                    f"Error behavior: {error_rule}.\n"
                    "Implementation guidance: validate inputs first, apply business rules second, then return JSON."
                ),
                trace_id=f"synthetic-api-{idx:06d}",
                metadata={
                    "task_domain": "api_implementation",
                    "source": "synthetic_api_seed",
                    "endpoint": f"/{endpoint}",
                    "domain": domain,
                },
            )
        )
    return records


def _stringify_maybe_json(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _format_agent_trajectory(row: dict[str, Any]) -> str:
    messages = row.get("messages") or row.get("trajectory") or row.get("trace") or row.get("turns") or []
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except json.JSONDecodeError:
            return messages.strip()
    if isinstance(messages, dict):
        messages = messages.get("turns") or messages.get("messages") or [messages]
    turns = []
    for idx, message in enumerate(messages if isinstance(messages, list) else []):
        if isinstance(message, dict):
            role = message.get("role") or message.get("sender") or message.get("agent") or message.get("name") or f"agent_{idx}"
            prompt = message.get("prompt") or message.get("input") or message.get("observation") or ""
            output = message.get("output") or message.get("response") or message.get("content") or message.get("text") or message.get("message") or ""
            if prompt and output:
                turns.append(f"[{role}]\nInput: {prompt}\nOutput: {output}")
            else:
                turns.append(f"[{role}] {prompt or output}")
        elif message is not None:
            turns.append(str(message))
    return "\n\n".join(turn.strip() for turn in turns if turn and turn.strip())


def _first_answer(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") or stripped.startswith("("):
            try:
                parsed = json.loads(stripped)
                return _first_answer(parsed)
            except json.JSONDecodeError:
                try:
                    return _first_answer(ast.literal_eval(stripped))
                except (ValueError, SyntaxError):
                    return stripped
        return stripped
    if isinstance(value, (list, tuple)):
        return _first_answer(value[0]) if value else ""
    return str(value).strip()


def _extract_target_text(row: dict[str, Any]) -> str | None:
    explicit_target = row.get("target_text")
    if explicit_target is not None:
        if isinstance(explicit_target, str):
            return explicit_target if explicit_target.strip() else None
        text = _first_answer(explicit_target)
        return text if text else None

    for key in ("target_agent_output", "agent_output", "final_answer", "completion", "solution"):
        value = row.get(key)
        if value is not None:
            text = _first_answer(value)
            return text if text else None
    answer = row.get("answer") or row.get("answers") or row.get("possible_answers")
    text = _first_answer(answer)
    return text if text else None


def _record_from_popqa(row: dict[str, Any], idx: int) -> SSATraceRecord:
    question = str(row.get("prompt") or row.get("question") or "").strip()
    target_text = _extract_target_text(row)
    evidence = str(row.get("context_text") or row.get("context") or row.get("memory") or "").strip()
    if not evidence:
        evidence = f"Known answer aliases: {_stringify_maybe_json(row.get('answer') or row.get('possible_answers') or target_text)}"
    return SSATraceRecord(
        task_prompt=question or f"Answer PopQA question {idx}.",
        context_text=evidence,
        answer_prefix="The answer is:",
        target_text=f" {target_text}" if target_text else None,
        trace_id=f"popqa-{idx:06d}",
        metadata={"task_domain": "factoid_qa", "source": "popqa"},
    )


def _record_from_kodcode(row: dict[str, Any], idx: int) -> SSATraceRecord:
    prompt = str(row.get("prompt") or row.get("question") or "").strip()
    solution = str(row.get("solution") or row.get("completion") or "").strip()
    test_text = _stringify_maybe_json(row.get("test"))
    test_info = _stringify_maybe_json(row.get("test_info"))
    context_text = str(row.get("context_text") or row.get("context") or "").strip()
    if not context_text:
        context_text = (
            f"Reference solution:\n{solution or 'N/A'}\n\n"
            f"Unit tests:\n{test_text or 'N/A'}\n\n"
            f"Test metadata:\n{test_info or 'N/A'}"
        )
    return SSATraceRecord(
        task_prompt=prompt or f"Solve KodCode task {idx}.",
        context_text=context_text,
        answer_prefix="\n```python\n",
        target_text=solution if solution else None,
        trace_id=f"kodcode-{idx:06d}",
        metadata={"task_domain": "coding_problem", "source": "kodcode", "test_info": row.get("test_info")},
    )


def _record_from_apibench(row: dict[str, Any], idx: int) -> SSATraceRecord:
    api_data = row.get("api_data")
    if isinstance(api_data, str):
        try:
            api_data = json.loads(api_data)
        except json.JSONDecodeError:
            api_data = {"raw_api_data": api_data}
    api_data = api_data if isinstance(api_data, dict) else {}
    instruction = str(row.get("instruction") or row.get("query") or row.get("code") or "").strip()
    api_call = str(row.get("api_call") or api_data.get("api_call") or "").strip()
    provider = str(row.get("provider") or api_data.get("framework") or api_data.get("api_provider") or "").strip()
    description = str(api_data.get("description") or api_data.get("functionality") or api_data.get("api_arguments") or "").strip()
    task_prompt = instruction.split("###Output:", 1)[0].replace("###Instruction:", "").strip()
    if not task_prompt:
        task_prompt = f"Write the correct API call for APIBench item {idx}."
    return SSATraceRecord(
        task_prompt=task_prompt,
        context_text=(
            f"Provider/framework: {provider or 'unknown'}.\n"
            f"API name: {api_data.get('api_name', 'unknown')}.\n"
            f"API call: {api_call or 'unknown'}.\n"
            f"Description/functionality: {description or 'not provided'}.\n"
            f"Environment requirements: {api_data.get('python_environment_requirements', 'not provided')}."
        ),
        trace_id=f"apibench-{idx:06d}",
        metadata={"task_domain": "api_calling", "source": "apibench", "provider": provider},
    )


def _record_from_codesearchnet(row: dict[str, Any], idx: int) -> SSATraceRecord:
    comment = str(row.get("comment") or row.get("docstring") or row.get("doc") or "").strip()
    code = str(row.get("code") or row.get("func_code_string") or "").strip()
    func_name = str(row.get("func_name") or row.get("function_name") or "function").strip()
    language = str(row.get("language") or "unknown").strip()
    return SSATraceRecord(
        task_prompt=comment or f"Implement or explain `{func_name}`.",
        context_text=f"Language: {language}.\nFunction name: {func_name}.\nReference implementation:\n{code}",
        trace_id=f"codesearchnet-{idx:06d}",
        metadata={"task_domain": "code_understanding", "source": "codesearchnet", "language": language},
    )


def _record_from_apps(row: dict[str, Any], idx: int) -> SSATraceRecord:
    return SSATraceRecord(
        task_prompt=str(row.get("question") or f"Solve programming task {idx}.").strip(),
        context_text=(
            f"Starter code:\n{row.get('starter_code') or 'N/A'}\n\n"
            f"Input/output tests:\n{_stringify_maybe_json(row.get('input_output')) or 'N/A'}\n\n"
            f"Reference solution bundle:\n{_stringify_maybe_json(row.get('solutions'))[:2400] or 'N/A'}"
        ),
        trace_id=f"apps-{idx:06d}",
        metadata={"task_domain": "coding_problem", "source": "apps", "difficulty": row.get("difficulty")},
    )


def _record_from_agent_trajectory(row: dict[str, Any], idx: int) -> SSATraceRecord:
    task_prompt = str(
        row.get("current_prompt")
        or row.get("agent_prompt")
        or row.get("task")
        or row.get("prompt")
        or row.get("task_prompt")
        or "Use the retrieved MAS trajectory to produce the next agent response."
    ).strip()
    return SSATraceRecord(
        task_prompt=task_prompt,
        context_text=_format_agent_trajectory(row) or _stringify_maybe_json(row),
        answer_prefix=str(row.get("answer_prefix") or "Next agent output:"),
        target_text=_extract_target_text(row),
        trace_id=f"agent-trajectory-{idx:06d}",
        metadata={
            "task_domain": "agent_trajectory",
            "source": "agent_trajectories",
            "agent_role": row.get("agent") or row.get("role") or row.get("active_agent"),
            "reward": row.get("reward") or row.get("score"),
        },
    )


def ssa_records_from_hf_dataset(
    *,
    dataset_name: str,
    split: str = "train",
    subset: str | None = None,
    limit: int = 200,
) -> list[SSATraceRecord]:
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise RuntimeError("Public dataset conversion requires `datasets`.") from exc

    dataset = load_dataset(dataset_name, subset, split=split) if subset else load_dataset(dataset_name, split=split)
    records: list[SSATraceRecord] = []
    lower_name = dataset_name.lower()
    for idx, row in enumerate(dataset):
        if idx >= limit:
            break
        row = dict(row)
        if "apibench" in lower_name:
            record = _record_from_apibench(row, idx)
        elif "popqa" in lower_name:
            record = _record_from_popqa(row, idx)
        elif "kodcode" in lower_name:
            record = _record_from_kodcode(row, idx)
        elif "codesearchnet" in lower_name:
            record = _record_from_codesearchnet(row, idx)
        elif lower_name.endswith("/apps") or "apps" in lower_name:
            record = _record_from_apps(row, idx)
        elif "agent_trajectories" in lower_name or "tau-bench" in lower_name:
            record = _record_from_agent_trajectory(row, idx)
        else:
            task_prompt = str(row.get("task_prompt") or row.get("prompt") or row.get("question") or row.get("instruction") or "").strip()
            context_text = str(row.get("context_text") or row.get("context") or row.get("answer") or row.get("code") or _stringify_maybe_json(row)).strip()
            record = SSATraceRecord(
                task_prompt=task_prompt or f"Use record {idx} as task context.",
                context_text=context_text,
                target_text=_extract_target_text(row),
                trace_id=f"hf-{idx:06d}",
                metadata={"task_domain": "generic", "source": dataset_name},
            )
        records.append(record)
    return records


def ssa_records_from_popqa(*, split: str = "test", limit: int = 200) -> list[SSATraceRecord]:
    return ssa_records_from_hf_dataset(dataset_name="akariasai/PopQA", split=split, limit=limit)


def ssa_records_from_kodcode(*, split: str = "train", limit: int = 200) -> list[SSATraceRecord]:
    return ssa_records_from_hf_dataset(dataset_name="KodCode/KodCode-Light-RL-10K", split=split, limit=limit)
