from __future__ import annotations

"""Trainable query and memory retrieval-key encoders."""

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ssamem.userspace import UserSpaceMAS


@dataclass
class RetrievalModelConfig:
    hidden_size: int
    key_dim: int
    query_max_tokens: int = 96
    temperature: float = 0.07

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RetrievalAlignmentOutput:
    loss: torch.Tensor
    logits: torch.Tensor
    query_keys: torch.Tensor
    memory_keys: torch.Tensor
    recall_at_1: float
    recall_at_5: float
    mrr: float


def _ensure_2d_latent(latent: torch.Tensor) -> torch.Tensor:
    if latent.dim() == 3:
        if latent.size(0) != 1:
            raise ValueError(f"Expected [K,D] or [1,K,D] latent, got {tuple(latent.shape)}.")
        latent = latent.squeeze(0)
    if latent.dim() != 2:
        raise ValueError(f"Expected [K,D] latent, got {tuple(latent.shape)}.")
    return latent


def positive_ranks(logits: torch.Tensor) -> torch.Tensor:
    labels = torch.arange(logits.size(0), device=logits.device)
    sorted_indices = torch.argsort(logits, dim=1, descending=True)
    matches = sorted_indices.eq(labels.unsqueeze(1))
    return matches.float().argmax(dim=1).long() + 1


class QueryLatentEncoder(nn.Module):
    """Frozen LLM text encoder plus a trainable retrieval projection head."""

    def __init__(self, userspace: UserSpaceMAS, *, key_dim: int, query_max_tokens: int = 96) -> None:
        super().__init__()
        self.userspace = userspace
        self.query_max_tokens = int(query_max_tokens)
        self.projection = nn.Sequential(
            nn.LayerNorm(userspace.hidden_size),
            nn.Linear(userspace.hidden_size, key_dim),
        )

    def _encode_text_hidden(self, texts: list[str]) -> torch.Tensor:
        tokenizer = self.userspace.tokenizer
        model = self.userspace.model
        tokenized = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.query_max_tokens,
            return_tensors="pt",
        )
        input_ids = tokenized["input_ids"].to(self.userspace.device)
        attention_mask = tokenized["attention_mask"].to(self.userspace.device)
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            hidden = outputs.hidden_states[-1].detach().float()
        mask = attention_mask.unsqueeze(-1).float()
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def forward(self, texts: str | list[str]) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]
        hidden = self._encode_text_hidden(texts)
        device = next(self.projection.parameters()).device
        keys = self.projection(hidden.to(device))
        return F.normalize(keys.float(), dim=-1)


class MemoryKeyEncoder(nn.Module):
    """Trainable retrieval projection over latent memory tensors."""

    def __init__(self, *, hidden_size: int, key_dim: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, key_dim),
        )

    def forward(self, latent_tensors: list[torch.Tensor]) -> torch.Tensor:
        device = next(self.projection.parameters()).device
        pooled = []
        for latent in latent_tensors:
            prepared = _ensure_2d_latent(latent).to(device=device, dtype=torch.float32)
            pooled.append(prepared.mean(dim=0))
        keys = self.projection(torch.stack(pooled, dim=0))
        return F.normalize(keys.float(), dim=-1)


class RetrievalAlignmentModel(nn.Module):
    """Symmetric InfoNCE model for query-to-memory retrieval alignment."""

    def __init__(self, userspace: UserSpaceMAS, config: RetrievalModelConfig) -> None:
        super().__init__()
        self.config = config
        self.query_encoder = QueryLatentEncoder(
            userspace,
            key_dim=config.key_dim,
            query_max_tokens=config.query_max_tokens,
        )
        self.memory_encoder = MemoryKeyEncoder(hidden_size=config.hidden_size, key_dim=config.key_dim)

    def encode_queries(self, task_prompts: list[str]) -> torch.Tensor:
        return self.query_encoder(task_prompts)

    def encode_memories(self, latent_tensors: list[torch.Tensor]) -> torch.Tensor:
        return self.memory_encoder(latent_tensors)

    def forward(self, task_prompts: list[str], latent_tensors: list[torch.Tensor]) -> RetrievalAlignmentOutput:
        query_keys = self.encode_queries(task_prompts)
        memory_keys = self.encode_memories(latent_tensors)
        logits = query_keys @ memory_keys.T / float(self.config.temperature)
        labels = torch.arange(logits.size(0), device=logits.device)
        loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
        ranks = positive_ranks(logits.detach())
        return RetrievalAlignmentOutput(
            loss=loss,
            logits=logits,
            query_keys=query_keys,
            memory_keys=memory_keys,
            recall_at_1=float((ranks <= 1).float().mean().item()),
            recall_at_5=float((ranks <= 5).float().mean().item()),
            mrr=float((1.0 / ranks.float()).mean().item()),
        )
