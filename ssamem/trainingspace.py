from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import random
import re
import statistics
from typing import Any, Iterable, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset
from transformers import GenerationConfig

from ssamem.userspace import UserSpaceMAS


POINTER_TOKEN_PATTERN = re.compile(r"^<PTR_0x[0-9A-Fa-f]+>$")


@dataclass
class SSASample:
    task_prompt: str
    context_text: str
    answer_prefix: str = "The answer is:"
    latent_tensor_path: str | None = None
    target_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def explicit_text(self) -> str:
        return f"{self.task_prompt}\n\n{self.context_text}\n\n{self.answer_prefix}".strip()

    def student_text(self) -> str:
        return f"{self.task_prompt}\n\n{self.answer_prefix}".strip()


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
class SSABatch:
    latent_tensors: Sequence[torch.Tensor]
    explicit_cot_texts: Sequence[str]
    student_prompts: Optional[Sequence[str]] = None
    answer_prefixes: Optional[Sequence[str]] = None
    target_texts: Optional[Sequence[str]] = None


@dataclass
class DistillationStepOutput:
    loss: torch.Tensor
    student_hidden: torch.Tensor
    teacher_hidden: torch.Tensor
    latent_distance: torch.Tensor
    hidden_loss: torch.Tensor | None = None
    answer_loss: torch.Tensor | None = None
    preference_loss: torch.Tensor | None = None
    preference_margin: torch.Tensor | None = None


class SSAManifestDataset(Dataset):
    def __init__(self, manifest_path: str | Path, *, map_location: str | torch.device = "cpu") -> None:
        self.manifest_path = Path(manifest_path)
        self.root_dir = self.manifest_path.parent
        self.map_location = map_location
        self.samples = load_ssa_manifest(self.manifest_path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> SSABatch:
        sample = self.samples[index]
        if sample.latent_tensor_path is None:
            raise ValueError(f"SSA sample {index} does not include a latent tensor path.")
        latent_path = Path(sample.latent_tensor_path)
        if not latent_path.is_absolute():
            latent_path = self.root_dir / latent_path
        latent_tensor = torch.load(latent_path, map_location=self.map_location)
        return SSABatch(
            latent_tensors=[latent_tensor],
            explicit_cot_texts=[sample.explicit_text()],
            student_prompts=[sample.student_text()],
            answer_prefixes=[sample.answer_prefix],
            target_texts=None if sample.target_text is None else [sample.target_text],
        )


def _jsonl_rows(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc


def load_ssa_manifest(path: str | Path) -> list[SSASample]:
    return [SSASample(**row) for row in _jsonl_rows(path)]


def save_ssa_manifest(samples: Sequence[SSASample], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")


def _sample_with_rewritten_latent_path(
    sample: SSASample,
    *,
    manifest_root: Path,
    output_root: Path,
) -> SSASample:
    latent_tensor_path = sample.latent_tensor_path
    if latent_tensor_path:
        latent_path = Path(latent_tensor_path)
        if not latent_path.is_absolute():
            latent_path = (manifest_root / latent_path).resolve()
        try:
            latent_tensor_path = str(Path("..") / latent_path.relative_to(output_root.parent.resolve()))
        except ValueError:
            latent_tensor_path = str(latent_path)
    return SSASample(
        task_prompt=sample.task_prompt,
        context_text=sample.context_text,
        answer_prefix=sample.answer_prefix,
        latent_tensor_path=latent_tensor_path,
        target_text=sample.target_text,
        metadata=dict(sample.metadata),
    )


def split_ssa_manifest(
    manifest_path: str | Path,
    *,
    output_dir: str | Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 7,
) -> dict[str, Any]:
    total_ratio = train_ratio + val_ratio + test_ratio
    if total_ratio <= 0:
        raise ValueError("Split ratios must sum to a positive value.")

    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_ssa_manifest(manifest_path)
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)

    sample_count = len(shuffled)
    normalized_train = train_ratio / total_ratio
    normalized_val = val_ratio / total_ratio
    train_end = int(sample_count * normalized_train)
    val_end = train_end + int(sample_count * normalized_val)

    split_map = {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }

    manifest_root = manifest_path.parent.resolve()
    payload: dict[str, Any] = {"source_manifest": str(manifest_path), "seed": seed, "splits": {}}
    for split_name, split_samples in split_map.items():
        rewritten: list[SSASample] = []
        for sample in split_samples:
            rewritten.append(
                _sample_with_rewritten_latent_path(
                    sample,
                    manifest_root=manifest_root,
                    output_root=output_dir,
                )
            )
        split_path = output_dir / f"{split_name}.jsonl"
        save_ssa_manifest(rewritten, split_path)
        payload["splits"][split_name] = {"count": len(rewritten), "manifest": str(split_path)}
    payload["total"] = sample_count
    return payload


def kfold_ssa_manifest(
    manifest_path: str | Path,
    *,
    output_dir: str | Path,
    folds: int = 5,
    seed: int = 7,
) -> dict[str, Any]:
    if folds < 2:
        raise ValueError("`folds` must be at least 2.")

    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_ssa_manifest(manifest_path)
    if folds > len(samples):
        raise ValueError(f"`folds` ({folds}) cannot exceed sample count ({len(samples)}).")

    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    manifest_root = manifest_path.parent.resolve()
    fold_sizes = [len(shuffled) // folds for _ in range(folds)]
    for idx in range(len(shuffled) % folds):
        fold_sizes[idx] += 1

    payload: dict[str, Any] = {
        "source_manifest": str(manifest_path),
        "seed": seed,
        "folds": folds,
        "total": len(shuffled),
        "folds_detail": [],
    }
    start = 0
    for fold_idx, fold_size in enumerate(fold_sizes):
        end = start + fold_size
        val_samples = shuffled[start:end]
        train_samples = shuffled[:start] + shuffled[end:]
        fold_dir = output_dir / f"fold_{fold_idx:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_rewritten = [
            _sample_with_rewritten_latent_path(sample, manifest_root=manifest_root, output_root=fold_dir)
            for sample in train_samples
        ]
        val_rewritten = [
            _sample_with_rewritten_latent_path(sample, manifest_root=manifest_root, output_root=fold_dir)
            for sample in val_samples
        ]
        train_path = fold_dir / "train.jsonl"
        val_path = fold_dir / "val.jsonl"
        save_ssa_manifest(train_rewritten, train_path)
        save_ssa_manifest(val_rewritten, val_path)
        payload["folds_detail"].append(
            {
                "fold": fold_idx,
                "train_count": len(train_rewritten),
                "val_count": len(val_rewritten),
                "train_manifest": str(train_path),
                "val_manifest": str(val_path),
            }
        )
        start = end
    return payload


def filter_ssa_manifest(
    manifest_path: str | Path,
    *,
    output_path: str | Path,
    metadata_key: str,
    metadata_value: str,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    samples = load_ssa_manifest(manifest_path)
    manifest_root = manifest_path.parent.resolve()
    output_root = output_path.parent.resolve()
    filtered: list[SSASample] = []
    for sample in samples:
        if str(sample.metadata.get(metadata_key)) != metadata_value:
            continue
        latent_tensor_path = sample.latent_tensor_path
        if latent_tensor_path:
            latent_path = Path(latent_tensor_path)
            if not latent_path.is_absolute():
                latent_path = (manifest_root / latent_path).resolve()
            if latent_path.is_relative_to(output_root):
                latent_tensor_path = str(latent_path.relative_to(output_root))
            else:
                latent_tensor_path = str(latent_path)
        filtered.append(
            SSASample(
                task_prompt=sample.task_prompt,
                context_text=sample.context_text,
                answer_prefix=sample.answer_prefix,
                latent_tensor_path=latent_tensor_path,
                target_text=sample.target_text,
                metadata=dict(sample.metadata),
            )
        )
    save_ssa_manifest(filtered, output_path)
    return {
        "source_manifest": str(manifest_path),
        "output_manifest": str(output_path),
        "metadata_key": metadata_key,
        "metadata_value": metadata_value,
        "count": len(filtered),
        "total": len(samples),
    }


def save_ssa_trace_records(records: Sequence[SSATraceRecord], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_json_row(), ensure_ascii=False) + "\n")


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
        task_prompt = (
            f"Worker must implement POST /{endpoint}. Use the mounted memory page for the exact contract."
        )
        context_text = (
            f"API contract for /{endpoint}.\n"
            f"Domain: {domain}.\n"
            f"HTTP method: POST.\n"
            f"Input schema: {inputs}.\n"
            f"Output schema: {output}.\n"
            f"Business rule: {rule}\n"
            f"Tax or service rate when applicable: {rate}.\n"
            f"Error behavior: {error_rule}.\n"
            "Implementation guidance: validate inputs first, apply business rules second, then return JSON."
        )
        records.append(
            SSATraceRecord(
                task_prompt=task_prompt,
                context_text=context_text,
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


def _record_from_apibench(row: dict[str, Any], idx: int) -> SSATraceRecord:
    api_data = row.get("api_data")
    if isinstance(api_data, str):
        try:
            api_data = json.loads(api_data.replace("'", '"'))
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
    context_text = (
        f"Provider/framework: {provider or 'unknown'}.\n"
        f"API name: {api_data.get('api_name', 'unknown')}.\n"
        f"API call: {api_call or 'unknown'}.\n"
        f"Description/functionality: {description or 'not provided'}.\n"
        f"Environment requirements: {api_data.get('python_environment_requirements', 'not provided')}."
    )
    return SSATraceRecord(
        task_prompt=task_prompt,
        context_text=context_text,
        trace_id=f"apibench-{idx:06d}",
        metadata={"task_domain": "api_calling", "source": "apibench", "provider": provider},
    )


def _record_from_codesearchnet(row: dict[str, Any], idx: int) -> SSATraceRecord:
    comment = str(row.get("comment") or row.get("docstring") or row.get("doc") or "").strip()
    code = str(row.get("code") or row.get("func_code_string") or "").strip()
    func_name = str(row.get("func_name") or row.get("function_name") or "function").strip()
    language = str(row.get("language") or "unknown").strip()
    task_prompt = comment or f"Implement or explain `{func_name}`."
    context_text = (
        f"Language: {language}.\n"
        f"Function name: {func_name}.\n"
        f"Reference implementation:\n{code}"
    )
    return SSATraceRecord(
        task_prompt=task_prompt,
        context_text=context_text,
        trace_id=f"codesearchnet-{idx:06d}",
        metadata={"task_domain": "code_understanding", "source": "codesearchnet", "language": language},
    )


def _record_from_apps(row: dict[str, Any], idx: int) -> SSATraceRecord:
    question = str(row.get("question") or "").strip()
    starter_code = str(row.get("starter_code") or "").strip()
    input_output = _stringify_maybe_json(row.get("input_output"))
    solutions = _stringify_maybe_json(row.get("solutions"))
    context_text = (
        f"Starter code:\n{starter_code or 'N/A'}\n\n"
        f"Input/output tests:\n{input_output or 'N/A'}\n\n"
        f"Reference solution bundle:\n{solutions[:2400] or 'N/A'}"
    )
    return SSATraceRecord(
        task_prompt=question or f"Solve programming task {idx}.",
        context_text=context_text,
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
    context_text = _format_agent_trajectory(row) or _stringify_maybe_json(row)
    return SSATraceRecord(
        task_prompt=task_prompt,
        context_text=context_text,
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
                trace_id=f"hf-{idx:06d}",
                metadata={"task_domain": "generic", "source": dataset_name},
            )
        records.append(record)
    return records


def load_pointer_preferences(path: str | Path) -> list[PointerPreferenceSample]:
    samples = [PointerPreferenceSample(**row) for row in _jsonl_rows(path)]
    for sample in samples:
        sample.validate()
    return samples


def save_pointer_preferences(samples: Sequence[PointerPreferenceSample], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            sample.validate()
            handle.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")


def pointer_preferences_from_rollouts(path: str | Path) -> list[PointerPreferenceSample]:
    samples: list[PointerPreferenceSample] = []
    for idx, row in enumerate(_jsonl_rows(path)):
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
            metadata={
                "source_rollout_id": row.get("rollout_id", idx),
                **dict(row.get("metadata") or {}),
            },
        )
        sample.validate()
        samples.append(sample)
    return samples


def pointer_tokens(max_index: int = 0xFFF) -> list[str]:
    return [f"<PTR_0x{idx:03X}>" for idx in range(1, max_index + 1)]


def initialize_pointer_vocabulary(tokenizer, model: Optional[nn.Module] = None, *, max_index: int = 0xFFF) -> int:
    tokens = pointer_tokens(max_index)
    added = 0
    if hasattr(tokenizer, "add_special_tokens"):
        added = int(tokenizer.add_special_tokens({"additional_special_tokens": tokens}))
        if model is not None and added > 0 and hasattr(model, "resize_token_embeddings"):
            model.resize_token_embeddings(len(tokenizer))
    else:
        for token in tokens:
            if hasattr(tokenizer, "encode"):
                tokenizer.encode(token, add_special_tokens=False)
        added = len(tokens)
    return added


def prepare_lora_for_training(
    model: nn.Module,
    *,
    r: int = 128,
    alpha: int = 32,
    target_modules: Sequence[str] = ("q_proj", "v_proj"),
    dropout: float = 0.05,
):
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except ModuleNotFoundError as exc:
        raise RuntimeError("QLoRA/LoRA training requires `peft`.") from exc

    if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
        model = prepare_model_for_kbit_training(model)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if getattr(model, "config", None) is not None:
        model.config.use_cache = False
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=list(target_modules),
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_config)


def build_ssa_data_from_traces(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    userspace: Optional[UserSpaceMAS] = None,
    latent_max_tokens: int = 128,
    default_answer_prefix: str = "The answer is:",
) -> list[SSASample]:
    output_root = Path(output_dir)
    latents_dir = output_root / "latents"
    latents_dir.mkdir(parents=True, exist_ok=True)
    samples: list[SSASample] = []

    for idx, row in enumerate(_jsonl_rows(input_path)):
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

        answer_prefix = str(row.get("answer_prefix") or default_answer_prefix)
        target_text = _extract_target_text(row)
        metadata = dict(row.get("metadata") or {})
        metadata.setdefault("source_trace_id", row.get("trace_id", idx))

        latent_rel_path = None
        if userspace is not None:
            latent_tensor = userspace.encode_text_as_latent(
                context_text,
                max_tokens=latent_max_tokens,
                strategy="last_hidden",
            ).detach().cpu()
            latent_rel_path = f"latents/latent_{idx:06d}.pt"
            torch.save(latent_tensor, output_root / latent_rel_path)
            metadata["latent_shape"] = list(latent_tensor.shape)
            metadata["context_token_count"] = int(latent_tensor.size(0))

        samples.append(
            SSASample(
                task_prompt=task_prompt,
                context_text=context_text,
                answer_prefix=answer_prefix,
                latent_tensor_path=latent_rel_path,
                target_text=target_text,
                metadata=metadata,
            )
        )

    save_ssa_manifest(samples, output_root / "manifest.jsonl")
    return samples


def ssa_collate(batch: Sequence[SSABatch]) -> SSABatch:
    latent_tensors: list[torch.Tensor] = []
    explicit_texts: list[str] = []
    student_prompts: list[str] = []
    answer_prefixes: list[str] = []
    target_texts: list[str] = []
    has_targets = False
    for item in batch:
        latent_tensors.extend(item.latent_tensors)
        explicit_texts.extend(item.explicit_cot_texts)
        student_prompts.extend(item.student_prompts or [""] * len(item.latent_tensors))
        answer_prefixes.extend(item.answer_prefixes or [""] * len(item.latent_tensors))
        if item.target_texts is not None:
            has_targets = True
            target_texts.extend(item.target_texts)
        else:
            target_texts.extend([""] * len(item.latent_tensors))
    return SSABatch(
        latent_tensors=latent_tensors,
        explicit_cot_texts=explicit_texts,
        student_prompts=student_prompts,
        answer_prefixes=answer_prefixes,
        target_texts=target_texts if has_targets else None,
    )


def _resolve_latent_path(sample: SSASample, root_dir: str | Path) -> Path:
    if sample.latent_tensor_path is None:
        raise ValueError("SSA sample does not include a latent tensor path.")
    latent_path = Path(sample.latent_tensor_path)
    if not latent_path.is_absolute():
        latent_path = Path(root_dir) / latent_path
    return latent_path


def _summarize_scalar_series(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    if len(values) == 1:
        return {"count": 1, "mean": values[0], "std": 0.0, "min": values[0], "max": values[0]}
    return {
        "count": len(values),
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.pstdev(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def load_alignment_checkpoint(distiller: "SSADistiller", checkpoint_path: str | Path, *, map_location: str | torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    projection_state = checkpoint.get("student_projection", checkpoint)
    distiller.student_projection.load_state_dict(projection_state)
    memory_projector_state = checkpoint.get("memory_projector")
    if memory_projector_state is not None:
        distiller.student.memory_projector.load_state_dict(memory_projector_state)
    memory_composer_state = checkpoint.get("memory_composer")
    if memory_composer_state is not None:
        distiller.student.memory_composer.load_state_dict(memory_composer_state)


def evaluate_ssa_latent_distances(
    *,
    distiller: "SSADistiller",
    dataset: SSAManifestDataset,
    limit: Optional[int] = None,
) -> dict[str, Any]:
    metrics_by_name: dict[str, list[float]] = {
        "loss": [],
        "latent_distance": [],
        "hidden_loss": [],
        "answer_loss": [],
    }
    per_sample: list[dict[str, Any]] = []
    with torch.no_grad():
        for idx in range(len(dataset)):
            if limit is not None and idx >= limit:
                break
            batch = dataset[idx]
            step_output = distiller.compute_step(batch)
            row_metrics = {
                "loss": float(step_output.loss.detach().cpu().item()),
                "latent_distance": float(step_output.latent_distance.detach().cpu().item()),
                "hidden_loss": float(step_output.hidden_loss.detach().cpu().item()) if step_output.hidden_loss is not None else None,
                "answer_loss": float(step_output.answer_loss.detach().cpu().item()) if step_output.answer_loss is not None else None,
            }
            for name, value in row_metrics.items():
                if value is not None:
                    metrics_by_name[name].append(value)
            sample = dataset.samples[idx]
            per_sample.append(
                {
                    "index": idx,
                    "source_trace_id": sample.metadata.get("source_trace_id"),
                    "task_domain": sample.metadata.get("task_domain"),
                    **{name: value for name, value in row_metrics.items() if value is not None},
                }
            )
    summary = {
        name: _summarize_scalar_series(values)
        for name, values in metrics_by_name.items()
        if values
    }
    return {"summary": summary, "per_sample": per_sample}


_PROBE_STOPWORDS = {
    "the", "and", "with", "from", "that", "this", "then", "when", "into", "must",
    "use", "for", "are", "is", "was", "were", "will", "would", "should", "have",
    "has", "had", "after", "before", "only", "may", "your", "their", "there",
    "return", "returns", "output", "input", "schema", "method", "apply", "using",
    "please", "answer", "context", "task", "worker", "implement", "implementation",
}


def extract_probe_keywords(context_text: str, *, max_keywords: int = 8) -> list[str]:
    raw_tokens = re.findall(r"/[A-Za-z0-9_/-]+|[A-Za-z_][A-Za-z0-9_/-]{2,}|[0-9]+(?:\.[0-9]+)?%?", context_text)
    seen: set[str] = set()
    keywords: list[str] = []
    for token in raw_tokens:
        normalized = token.lower()
        if normalized in seen or normalized in _PROBE_STOPWORDS:
            continue
        seen.add(normalized)
        keywords.append(token)
        if len(keywords) >= max_keywords:
            break
    return keywords


def compute_keyword_recall(reference_keywords: Sequence[str], generated_text: str) -> float:
    if not reference_keywords:
        return 0.0
    haystack = generated_text.lower()
    hits = sum(1 for keyword in reference_keywords if keyword.lower() in haystack)
    return float(hits / max(len(reference_keywords), 1))


def target_text_hit(target_text: str | None, generated_text: str) -> bool:
    if not target_text:
        return False
    normalized_target = re.sub(r"[^a-z0-9 ]+", " ", target_text.lower()).strip()
    if not normalized_target:
        return False
    normalized_output = re.sub(r"[^a-z0-9 ]+", " ", generated_text.lower())
    return normalized_target in normalized_output


def probe_ssa_generation(
    *,
    userspace: UserSpaceMAS,
    dataset: SSAManifestDataset,
    student: Optional["LatentStudent"] = None,
    limit: int = 10,
    max_new_tokens: int = 48,
    max_keywords: int = 8,
) -> dict[str, Any]:
    generation_config = GenerationConfig(
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=userspace.tokenizer.pad_token_id,
        eos_token_id=userspace.tokenizer.eos_token_id,
    )
    examples: list[dict[str, Any]] = []
    latent_scores: list[float] = []
    text_scores: list[float] = []
    no_memory_scores: list[float] = []
    no_memory_target_hits: list[float] = []
    text_memory_target_hits: list[float] = []
    latent_memory_target_hits: list[float] = []

    for idx, sample in enumerate(dataset.samples):
        if idx >= limit:
            break
        latent_tensor = torch.load(_resolve_latent_path(sample, dataset.root_dir), map_location=userspace.device)
        prompt = sample.student_text()
        no_memory_output = userspace.run_role_turn(
            content=prompt,
            mounted_latents=None,
            generation_config=generation_config,
        ).text
        text_memory_output = userspace.run_role_turn(
            content=sample.explicit_text(),
            mounted_latents=None,
            generation_config=generation_config,
        ).text
        mounted_latent = latent_tensor
        if student is not None:
            mounted_latent = student.project_latents([prompt], [latent_tensor])[0]
        latent_memory_output = userspace.run_role_turn(
            content=prompt,
            mounted_latents=[mounted_latent],
            generation_config=generation_config,
        ).text
        keywords = extract_probe_keywords(sample.context_text, max_keywords=max_keywords)
        no_memory_score = compute_keyword_recall(keywords, no_memory_output)
        text_score = compute_keyword_recall(keywords, text_memory_output)
        latent_score = compute_keyword_recall(keywords, latent_memory_output)
        no_memory_target_hit = target_text_hit(sample.target_text, no_memory_output)
        text_memory_target_hit = target_text_hit(sample.target_text, text_memory_output)
        latent_memory_target_hit = target_text_hit(sample.target_text, latent_memory_output)
        no_memory_scores.append(no_memory_score)
        text_scores.append(text_score)
        latent_scores.append(latent_score)
        no_memory_target_hits.append(float(no_memory_target_hit))
        text_memory_target_hits.append(float(text_memory_target_hit))
        latent_memory_target_hits.append(float(latent_memory_target_hit))
        examples.append(
            {
                "index": idx,
                "source_trace_id": sample.metadata.get("source_trace_id"),
                "keywords": keywords,
                "target_text": sample.target_text,
                "task_prompt": sample.task_prompt,
                "no_memory_output": no_memory_output,
                "text_memory_output": text_memory_output,
                "latent_memory_output": latent_memory_output,
                "no_memory_keyword_recall": no_memory_score,
                "text_memory_keyword_recall": text_score,
                "latent_memory_keyword_recall": latent_score,
                "no_memory_target_hit": no_memory_target_hit,
                "text_memory_target_hit": text_memory_target_hit,
                "latent_memory_target_hit": latent_memory_target_hit,
            }
        )

    return {
        "count": len(examples),
        "no_memory": _summarize_scalar_series(no_memory_scores),
        "text_memory": _summarize_scalar_series(text_scores),
        "latent_memory": _summarize_scalar_series(latent_scores),
        "target_hit": {
            "no_memory": _summarize_scalar_series(no_memory_target_hits),
            "text_memory": _summarize_scalar_series(text_memory_target_hits),
            "latent_memory": _summarize_scalar_series(latent_memory_target_hits),
        },
        "examples": examples,
    }


def _last_nonpad_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(attention_mask.size(1), device=attention_mask.device).unsqueeze(0)
    masked_positions = positions.masked_fill(attention_mask.long() == 0, 0)
    return masked_positions.max(dim=1).values


def gather_anchor_hidden(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    indices = _last_nonpad_indices(attention_mask).to(hidden_states.device)
    batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
    return hidden_states[batch_indices, indices]


class ExplicitTeacher(nn.Module):
    def __init__(self, userspace: UserSpaceMAS) -> None:
        super().__init__()
        self.userspace = userspace

    @property
    def hidden_size(self) -> int:
        return self.userspace.hidden_size

    def forward(self, texts: Sequence[str]) -> torch.Tensor:
        tokenized = self.userspace.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokenized["input_ids"].to(self.userspace.device)
        attention_mask = tokenized["attention_mask"].to(self.userspace.device)
        outputs = self.userspace.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        return gather_anchor_hidden(outputs.hidden_states[-1], attention_mask)


class LatentPrefixProjector(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        hidden_multiplier: int = 2,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        hidden_size = max(input_size, output_size) * hidden_multiplier
        layers: list[nn.Module] = []
        if use_layer_norm:
            layers.append(nn.LayerNorm(input_size))
        layers.extend(
            [
                nn.Linear(input_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, output_size),
            ]
        )
        self.network = nn.Sequential(*layers)
        self.residual = input_size == output_size

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        projected = self.network(latent)
        if self.residual:
            projected = projected + latent
        return projected


class QueryLatentComposer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        *,
        prefix_length: int = 8,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("`hidden_size` must be divisible by `num_heads` for QueryLatentComposer.")
        self.hidden_size = hidden_size
        self.prefix_length = prefix_length
        self.query_latents = nn.Parameter(torch.randn(prefix_length, hidden_size) * 0.02)
        self.prompt_to_query = nn.Linear(hidden_size, hidden_size)
        self.memory_norm = nn.LayerNorm(hidden_size)
        self.query_norm = nn.LayerNorm(hidden_size)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )

    def forward(self, prompt_embedding: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        if latent.dim() != 2:
            raise ValueError(f"`latent` must have shape [K, D], got {tuple(latent.shape)}.")
        query = self.query_latents.to(device=latent.device, dtype=latent.dtype).unsqueeze(0)
        prompt_query = self.prompt_to_query(prompt_embedding).view(1, 1, -1)
        query = self.query_norm(query + prompt_query)
        memory = self.memory_norm(latent).unsqueeze(0)
        attended, _ = self.cross_attention(query=query, key=memory, value=memory, need_weights=False)
        return (attended + self.ffn(attended)).squeeze(0)


class LatentStudent(nn.Module):
    def __init__(
        self,
        userspace: UserSpaceMAS,
        memory_projector: Optional[nn.Module] = None,
        memory_composer: Optional[nn.Module] = None,
        *,
        composer_prefix_length: int = 8,
    ) -> None:
        super().__init__()
        self.userspace = userspace
        self.memory_composer = memory_composer or QueryLatentComposer(
            hidden_size=userspace.hidden_size,
            prefix_length=composer_prefix_length,
        )
        self.memory_projector = memory_projector or LatentPrefixProjector(
            input_size=userspace.hidden_size,
            output_size=userspace.hidden_size,
        )

    @property
    def hidden_size(self) -> int:
        return self.userspace.hidden_size

    def _prompt_anchor_embeddings(self, prompts: Sequence[str]) -> torch.Tensor:
        tokenized = self.userspace.tokenizer(
            list(prompts),
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokenized["input_ids"].to(self.userspace.device)
        attention_mask = tokenized["attention_mask"].to(self.userspace.device)
        embedding_layer = self.userspace.model.get_input_embeddings()
        text_embeds = embedding_layer(input_ids).to(self.userspace.model.dtype)
        indices = _last_nonpad_indices(attention_mask).to(self.userspace.device)
        batch_indices = torch.arange(text_embeds.size(0), device=self.userspace.device)
        return text_embeds[batch_indices, indices]

    def project_latents(self, prompts: Sequence[str], latent_tensors: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        projected_latents = []
        prompt_anchors = self._prompt_anchor_embeddings(prompts).float()
        self.memory_composer.to(device=self.userspace.device, dtype=torch.float32)
        self.memory_projector.to(device=self.userspace.device, dtype=torch.float32)
        for idx, latent in enumerate(latent_tensors):
            if latent.dim() == 3:
                if latent.size(0) != 1:
                    raise ValueError("Batched latent tensors are not supported in LatentStudent.")
                latent = latent.squeeze(0)
            latent = latent.to(device=self.userspace.device, dtype=torch.float32)
            composed = self.memory_composer(prompt_anchors[idx], latent)
            projected_latents.append(self.memory_projector(composed).to(dtype=self.userspace.model.dtype))
        return projected_latents

    def _batch_with_projected_latents(
        self,
        prompts: Sequence[str],
        latent_tensors: Sequence[torch.Tensor],
        projection_prompts: Optional[Sequence[str]] = None,
    ):
        projection_prompts = projection_prompts or prompts
        if len(projection_prompts) != len(prompts):
            raise ValueError("Projection prompt batch size must match model prompt batch size.")
        latent_batches = [[latent] for latent in self.project_latents(projection_prompts, latent_tensors)]
        batch = self.userspace.prepare_soft_prompt_batch(prompts=prompts, mounted_latents=latent_batches)
        return batch

    def _target_label_tensor(self, batch, prompts: Sequence[str]) -> torch.Tensor:
        labels = []
        for idx, prompt in enumerate(prompts):
            prefix_len = batch.prefix_lengths[idx]
            text_len = batch.text_lengths[idx]
            text_ids = batch.input_ids[idx, -text_len:].detach().clone()
            prompt_ids = self.userspace.tokenizer(
                prompt,
                truncation=True,
                return_tensors="pt",
            )["input_ids"].to(self.userspace.device)
            prompt_len = min(int(prompt_ids.size(1)), int(text_len))
            sample_labels = torch.full(
                (prefix_len + text_len,),
                -100,
                dtype=torch.long,
                device=self.userspace.device,
            )
            if prompt_len < text_len:
                sample_labels[prefix_len + prompt_len :] = text_ids[prompt_len:]
            labels.append(sample_labels)

        max_seq_len = batch.inputs_embeds.size(1)
        padded_labels = []
        for sample_labels in labels:
            pad_len = max_seq_len - sample_labels.size(0)
            if pad_len > 0:
                sample_labels = torch.cat(
                    [
                        torch.full((pad_len,), -100, dtype=torch.long, device=self.userspace.device),
                        sample_labels,
                    ],
                    dim=0,
                )
            padded_labels.append(sample_labels)
        return torch.stack(padded_labels, dim=0)

    def forward(self, prompts: Sequence[str], latent_tensors: Sequence[torch.Tensor]) -> torch.Tensor:
        batch = self._batch_with_projected_latents(prompts=prompts, latent_tensors=latent_tensors)
        outputs = self.userspace.model(
            inputs_embeds=batch.inputs_embeds,
            attention_mask=batch.attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        return gather_anchor_hidden(outputs.hidden_states[-1], batch.attention_mask)

    def answer_loss(
        self,
        prompts: Sequence[str],
        latent_tensors: Sequence[torch.Tensor],
        target_texts: Sequence[str],
    ) -> torch.Tensor:
        full_texts = [f"{prompt}{target}" for prompt, target in zip(prompts, target_texts)]
        # Composer/projector must see the same prompt it will see at inference time.
        # The target is present only in the LM input/labels used for CE supervision.
        batch = self._batch_with_projected_latents(
            prompts=full_texts,
            latent_tensors=latent_tensors,
            projection_prompts=prompts,
        )
        label_tensor = self._target_label_tensor(batch, prompts)
        if not torch.any(label_tensor != -100):
            return batch.inputs_embeds.sum() * 0.0
        outputs = self.userspace.model(
            inputs_embeds=batch.inputs_embeds,
            attention_mask=batch.attention_mask,
            labels=label_tensor,
            use_cache=False,
            return_dict=True,
        )
        return outputs.loss

    def target_logprobs(
        self,
        prompts: Sequence[str],
        latent_tensors: Sequence[torch.Tensor],
        target_texts: Sequence[str],
    ) -> torch.Tensor:
        full_texts = [f"{prompt}{target}" for prompt, target in zip(prompts, target_texts)]
        batch = self._batch_with_projected_latents(
            prompts=full_texts,
            latent_tensors=latent_tensors,
            projection_prompts=prompts,
        )
        label_tensor = self._target_label_tensor(batch, prompts)
        outputs = self.userspace.model(
            inputs_embeds=batch.inputs_embeds,
            attention_mask=batch.attention_mask,
            use_cache=False,
            return_dict=True,
        )
        logits = outputs.logits[:, :-1, :].float().clamp(min=-80.0, max=80.0)
        log_probs = logits.log_softmax(dim=-1)
        shifted_labels = label_tensor[:, 1:]
        target_mask = shifted_labels != -100
        safe_labels = shifted_labels.masked_fill(~target_mask, 0)
        shifted_log_probs = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
        sample_logprobs = []
        for idx in range(len(prompts)):
            finite_mask = target_mask[idx] & torch.isfinite(shifted_log_probs[idx])
            if not torch.any(finite_mask):
                sample_logprobs.append(shifted_log_probs[idx].sum() * 0.0)
            else:
                sample_logprobs.append(shifted_log_probs[idx][finite_mask].mean())
        return torch.stack(sample_logprobs, dim=0)


class SSADistiller(nn.Module):
    def __init__(
        self,
        student: LatentStudent,
        teacher: ExplicitTeacher,
        *,
        loss_type: str = "smooth_l1",
        distill_loss_div_std: bool = True,
        answer_loss_weight: float = 0.0,
        preference_loss_weight: float = 0.0,
        preference_beta: float = 0.1,
        train_teacher: bool = False,
    ) -> None:
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.loss_type = loss_type
        self.distill_loss_div_std = distill_loss_div_std
        self.answer_loss_weight = float(answer_loss_weight)
        self.preference_loss_weight = float(preference_loss_weight)
        self.preference_beta = float(preference_beta)
        self._shared_backbone = self.teacher.userspace.model is self.student.userspace.model
        self.student_projection = nn.Linear(student.hidden_size, teacher.hidden_size)
        if student.hidden_size == teacher.hidden_size:
            nn.init.eye_(self.student_projection.weight)
            nn.init.zeros_(self.student_projection.bias)

        if not train_teacher and not self._shared_backbone:
            for param in self.teacher.parameters():
                param.requires_grad = False
            self.teacher.eval()

    def _distillation_loss(self, student_hidden: torch.Tensor, teacher_hidden: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "l1":
            loss = F.l1_loss(student_hidden, teacher_hidden.detach())
        elif self.loss_type in {"l2", "mse"}:
            loss = F.mse_loss(student_hidden, teacher_hidden.detach())
        elif self.loss_type == "smooth_l1":
            loss = F.smooth_l1_loss(student_hidden, teacher_hidden.detach())
        else:
            raise ValueError(f"Unsupported SSA loss type: {self.loss_type}")
        if self.distill_loss_div_std:
            loss = loss / teacher_hidden.detach().std().clamp(min=1e-6)
        return loss

    def compute_step(self, batch: SSABatch) -> DistillationStepOutput:
        if len(batch.latent_tensors) != len(batch.explicit_cot_texts):
            raise ValueError("Latent tensor batch size must match teacher text batch size.")

        prompts = batch.student_prompts or ["Decode the mounted latent memory into a compact reasoning state."] * len(batch.latent_tensors)
        student_hidden = self.student(prompts=prompts, latent_tensors=batch.latent_tensors)
        teacher_model = self.teacher.userspace.model
        teacher_was_training = bool(getattr(teacher_model, "training", False))
        teacher_model.eval()
        with torch.set_grad_enabled(False):
            teacher_hidden = self.teacher(batch.explicit_cot_texts)
        if teacher_was_training:
            teacher_model.train()

        self.student_projection.to(device=student_hidden.device, dtype=torch.float32)
        projected_student_hidden = self.student_projection(student_hidden.float())
        teacher_hidden = teacher_hidden.float()
        hidden_loss = self._distillation_loss(projected_student_hidden, teacher_hidden)
        answer_loss = None
        preference_loss = None
        preference_margin = None
        loss = hidden_loss
        if self.answer_loss_weight > 0.0 and batch.target_texts:
            answer_loss = self.student.answer_loss(
                prompts=prompts,
                latent_tensors=batch.latent_tensors,
                target_texts=batch.target_texts,
            )
            loss = loss + self.answer_loss_weight * answer_loss
        if self.preference_loss_weight > 0.0 and batch.target_texts and len(batch.latent_tensors) > 1:
            positive_logprobs = self.student.target_logprobs(
                prompts=prompts,
                latent_tensors=batch.latent_tensors,
                target_texts=batch.target_texts,
            )
            negative_latents = list(batch.latent_tensors[1:]) + [batch.latent_tensors[0]]
            negative_logprobs = self.student.target_logprobs(
                prompts=prompts,
                latent_tensors=negative_latents,
                target_texts=batch.target_texts,
            )
            raw_margin = positive_logprobs - negative_logprobs
            finite_margin = raw_margin[torch.isfinite(raw_margin)]
            if finite_margin.numel() > 0:
                preference_margin = finite_margin.clamp(min=-100.0, max=100.0)
                preference_loss = -F.logsigmoid(self.preference_beta * preference_margin).mean()
                loss = loss + self.preference_loss_weight * preference_loss
            else:
                preference_margin = raw_margin.detach().new_zeros(())
                preference_loss = loss.detach().new_zeros(())
        latent_distance = torch.linalg.vector_norm(projected_student_hidden.detach() - teacher_hidden.detach(), dim=-1).mean()
        return DistillationStepOutput(
            loss=loss,
            student_hidden=projected_student_hidden,
            teacher_hidden=teacher_hidden.detach(),
            latent_distance=latent_distance,
            hidden_loss=hidden_loss.detach(),
            answer_loss=None if answer_loss is None else answer_loss.detach(),
            preference_loss=None if preference_loss is None else preference_loss.detach(),
            preference_margin=None if preference_margin is None else preference_margin.detach().mean(),
        )

    def fit(
        self,
        dataloader: Iterable[SSABatch],
        optimizer: torch.optim.Optimizer,
        *,
        epochs: int = 1,
        grad_clip_norm: Optional[float] = None,
        device: Optional[torch.device | str] = None,
        log_every: int = 0,
    ) -> list[dict[str, float]]:
        if device is not None:
            self.to(device)

        history = []
        self.train()
        if not self._shared_backbone:
            self.teacher.eval()

        global_step = 0
        for epoch_idx in range(epochs):
            for batch in dataloader:
                global_step += 1
                optimizer.zero_grad(set_to_none=True)
                step_output = self.compute_step(batch)
                step_output.loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip_norm)
                optimizer.step()
                metrics = {
                    "loss": float(step_output.loss.detach().cpu().item()),
                    "latent_distance": float(step_output.latent_distance.detach().cpu().item()),
                }
                if step_output.hidden_loss is not None:
                    metrics["hidden_loss"] = float(step_output.hidden_loss.detach().cpu().item())
                if step_output.answer_loss is not None:
                    metrics["answer_loss"] = float(step_output.answer_loss.detach().cpu().item())
                if step_output.preference_loss is not None:
                    metrics["preference_loss"] = float(step_output.preference_loss.detach().cpu().item())
                if step_output.preference_margin is not None:
                    metrics["preference_margin"] = float(step_output.preference_margin.detach().cpu().item())
                history.append(metrics)
                if log_every > 0 and (global_step == 1 or global_step % log_every == 0):
                    metric_text = " ".join(f"{name}={value:.6g}" for name, value in metrics.items())
                    print(
                        "train-ssa "
                        f"epoch={epoch_idx + 1}/{epochs} "
                        f"step={global_step} "
                        f"{metric_text}",
                        flush=True,
                    )

        return history


def pointer_preferences_to_dpo_rows(samples: Sequence[PointerPreferenceSample]) -> list[dict[str, str]]:
    rows = []
    for sample in samples:
        sample.validate()
        rows.append({"prompt": sample.prompt, "chosen": sample.chosen, "rejected": sample.rejected})
    return rows


def train_pointer_dpo(
    *,
    model,
    tokenizer,
    preference_samples: Sequence[PointerPreferenceSample],
    output_dir: str | Path,
    beta: float = 0.1,
    max_prompt_length: int = 1024,
    max_completion_length: int = 16,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 32,
    learning_rate: float = 5e-5,
    num_train_epochs: float = 1.0,
):
    try:
        from datasets import Dataset as HFDataset
        from trl import DPOConfig, DPOTrainer
    except ModuleNotFoundError as exc:
        raise RuntimeError("Pointer DPO training requires `datasets` and `trl`.") from exc

    rows = pointer_preferences_to_dpo_rows(preference_samples)
    dataset = HFDataset.from_list(rows)
    args = DPOConfig(
        output_dir=str(output_dir),
        beta=beta,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        gradient_checkpointing=True,
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    return trainer.train()


def _completion_logprob(model, tokenizer, prompt: str, completion: str, device: str | torch.device) -> float:
    prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    prompt_completion_ids = tokenizer(prompt + completion, add_special_tokens=False, return_tensors="pt")
    input_ids = prompt_completion_ids["input_ids"].to(device)
    attention_mask = prompt_completion_ids["attention_mask"].to(device)
    prompt_length = int(prompt_ids["input_ids"].size(1))
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        logits = outputs.logits[:, :-1, :]
        target_ids = input_ids[:, 1:]
        log_probs = logits.log_softmax(dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
        completion_log_probs = token_log_probs[:, max(prompt_length - 1, 0):]
    return float(completion_log_probs.sum().detach().cpu().item())


def evaluate_pointer_routing(
    preference_samples: Sequence[PointerPreferenceSample],
    *,
    model=None,
    tokenizer=None,
    device: str | torch.device = "cpu",
    seed: int = 7,
    limit: Optional[int] = None,
) -> dict[str, Any]:
    rng = random.Random(seed)
    samples = list(preference_samples[:limit] if limit is not None else preference_samples)
    if not samples:
        return {"count": 0, "random_accuracy": 0.0, "majority_accuracy": 0.0}

    chosen_counts: dict[str, int] = {}
    for sample in samples:
        sample.validate()
        chosen_counts[sample.chosen] = chosen_counts.get(sample.chosen, 0) + 1
    majority_pointer = max(chosen_counts.items(), key=lambda item: item[1])[0]

    random_correct = 0
    majority_correct = 0
    model_correct = 0
    model_margins: list[float] = []
    per_sample: list[dict[str, Any]] = []

    for idx, sample in enumerate(samples):
        random_pick = rng.choice([sample.chosen, sample.rejected])
        majority_pick = majority_pointer if majority_pointer in {sample.chosen, sample.rejected} else sample.rejected
        random_correct += int(random_pick == sample.chosen)
        majority_correct += int(majority_pick == sample.chosen)

        row = {
            "index": idx,
            "prompt": sample.prompt,
            "chosen": sample.chosen,
            "rejected": sample.rejected,
            "random_pick": random_pick,
            "majority_pick": majority_pick,
        }
        if model is not None and tokenizer is not None:
            chosen_logprob = _completion_logprob(model, tokenizer, sample.prompt, sample.chosen, device)
            rejected_logprob = _completion_logprob(model, tokenizer, sample.prompt, sample.rejected, device)
            model_pick = sample.chosen if chosen_logprob >= rejected_logprob else sample.rejected
            margin = chosen_logprob - rejected_logprob
            model_correct += int(model_pick == sample.chosen)
            model_margins.append(margin)
            row.update(
                {
                    "model_pick": model_pick,
                    "chosen_logprob": chosen_logprob,
                    "rejected_logprob": rejected_logprob,
                    "margin": margin,
                }
            )
        per_sample.append(row)

    payload: dict[str, Any] = {
        "count": len(samples),
        "majority_pointer": majority_pointer,
        "random_accuracy": float(random_correct / len(samples)),
        "majority_accuracy": float(majority_correct / len(samples)),
        "per_sample": per_sample,
    }
    if model is not None and tokenizer is not None:
        payload["model_accuracy"] = float(model_correct / len(samples))
        payload["model_margin"] = _summarize_scalar_series(model_margins)
    return payload


def collect_synthetic_pointer_preferences(
    *,
    prompts: Sequence[str],
    pointers: Sequence[str],
    positive_pointer: Optional[str] = None,
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
