from __future__ import annotations

"""Training and evaluation loops for retrieval-key alignment."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

from ssamem.retrieval_training.data import RetrievalBatch
from ssamem.retrieval_training.model import RetrievalAlignmentModel, RetrievalModelConfig


@dataclass
class RetrievalEvalResult:
    count: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    mean_rank: float

    def to_dict(self) -> dict:
        return {
            "count": self.count,
            "recall_at_1": self.recall_at_1,
            "recall_at_3": self.recall_at_3,
            "recall_at_5": self.recall_at_5,
            "recall_at_10": self.recall_at_10,
            "mrr": self.mrr,
            "mean_rank": self.mean_rank,
        }


def train_retrieval_alignment(
    model: RetrievalAlignmentModel,
    dataloader: Iterable[RetrievalBatch],
    optimizer: torch.optim.Optimizer,
    *,
    epochs: int = 1,
    grad_clip_norm: float | None = None,
    log_every: int = 10,
) -> list[dict[str, float]]:
    history: list[dict[str, float]] = []
    model.train()
    step = 0
    for epoch_index in range(int(epochs)):
        for batch in dataloader:
            step += 1
            optimizer.zero_grad(set_to_none=True)
            output = model(batch.task_prompts, batch.latent_tensors)
            output.loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
            optimizer.step()
            metrics = {
                "loss": float(output.loss.detach().cpu().item()),
                "recall_at_1": output.recall_at_1,
                "recall_at_5": output.recall_at_5,
                "mrr": output.mrr,
            }
            history.append(metrics)
            if log_every > 0 and (step == 1 or step % log_every == 0):
                metric_text = " ".join(f"{name}={value:.6g}" for name, value in metrics.items())
                print(
                    f"train-retrieval epoch={epoch_index + 1}/{epochs} step={step} {metric_text}",
                    flush=True,
                )
    return history


@torch.no_grad()
def _encode_candidate_memories(
    model: RetrievalAlignmentModel,
    dataloader: Iterable[RetrievalBatch],
) -> tuple[torch.Tensor, list[str]]:
    memory_keys: list[torch.Tensor] = []
    sample_ids: list[str] = []
    model.eval()
    for batch in dataloader:
        keys = model.encode_memories(batch.latent_tensors).detach().cpu()
        memory_keys.append(keys)
        sample_ids.extend(batch.sample_ids)
    if not memory_keys:
        return torch.empty(0, model.config.key_dim), []
    return torch.cat(memory_keys, dim=0), sample_ids


@torch.no_grad()
def evaluate_retrieval_alignment(
    model: RetrievalAlignmentModel,
    query_dataloader: Iterable[RetrievalBatch],
    candidate_dataloader: Iterable[RetrievalBatch] | None = None,
) -> RetrievalEvalResult:
    model.eval()
    candidate_loader = candidate_dataloader or query_dataloader
    candidate_keys, candidate_ids = _encode_candidate_memories(model, candidate_loader)
    if candidate_keys.numel() == 0:
        return RetrievalEvalResult(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    id_to_position = {sample_id: pos for pos, sample_id in enumerate(candidate_ids)}
    candidate_keys = candidate_keys.to(next(model.parameters()).device)

    ranks: list[int] = []
    for batch in query_dataloader:
        query_keys = model.encode_queries(batch.task_prompts)
        scores = query_keys @ candidate_keys.T
        sorted_indices = torch.argsort(scores, dim=1, descending=True).cpu()
        for row_index, sample_id in enumerate(batch.sample_ids):
            positive_position = id_to_position.get(sample_id)
            if positive_position is None:
                continue
            matches = sorted_indices[row_index].eq(int(positive_position))
            rank = int(matches.float().argmax().item()) + 1
            ranks.append(rank)

    if not ranks:
        return RetrievalEvalResult(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    rank_tensor = torch.tensor(ranks, dtype=torch.float32)
    return RetrievalEvalResult(
        count=len(ranks),
        recall_at_1=float((rank_tensor <= 1).float().mean().item()),
        recall_at_3=float((rank_tensor <= 3).float().mean().item()),
        recall_at_5=float((rank_tensor <= 5).float().mean().item()),
        recall_at_10=float((rank_tensor <= 10).float().mean().item()),
        mrr=float((1.0 / rank_tensor).mean().item()),
        mean_rank=float(rank_tensor.mean().item()),
    )


def save_retrieval_checkpoint(
    model: RetrievalAlignmentModel,
    path: str | Path,
    *,
    history: list[dict[str, float]] | None = None,
    config: dict | None = None,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "retrieval_config": model.config.to_dict(),
            "query_projection": model.query_encoder.projection.state_dict(),
            "memory_projection": model.memory_encoder.projection.state_dict(),
            "history": history or [],
            "config": config or {},
        },
        output_path,
    )


def _load_projection_state(module: torch.nn.Module, payload: dict, preferred_key: str, legacy_key: str) -> None:
    if preferred_key in payload:
        module.load_state_dict(payload[preferred_key])
        return
    legacy_state = dict(payload.get(legacy_key) or {})
    prefix = "projection."
    projection_state = {
        key[len(prefix):]: value
        for key, value in legacy_state.items()
        if key.startswith(prefix)
    }
    if not projection_state:
        raise KeyError(f"Checkpoint does not contain {preferred_key} or {legacy_key}.projection weights.")
    module.load_state_dict(projection_state)


def load_retrieval_checkpoint(
    model: RetrievalAlignmentModel,
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict:
    payload = torch.load(path, map_location=map_location)
    _load_projection_state(model.query_encoder.projection, payload, "query_projection", "query_encoder")
    _load_projection_state(model.memory_encoder.projection, payload, "memory_projection", "memory_encoder")
    return payload


def retrieval_config_from_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> RetrievalModelConfig:
    payload = torch.load(path, map_location=map_location)
    return RetrievalModelConfig(**payload["retrieval_config"])
