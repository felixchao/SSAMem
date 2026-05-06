from __future__ import annotations

"""Dataset utilities for retrieval-key alignment."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

from ssamem.training.space import SSASample, _resolve_latent_path, load_ssa_manifest


@dataclass
class RetrievalBatch:
    task_prompts: list[str]
    student_prompts: list[str]
    latent_tensors: list[torch.Tensor]
    sample_indices: list[int]
    sample_ids: list[str]
    target_texts: list[str | None]
    metadata: list[dict[str, Any]]


class RetrievalManifestDataset(Dataset):
    """SSA manifest dataset viewed as query-memory positive pairs."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
        query_field: str = "task_prompt",
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.root_dir = self.manifest_path.parent
        self.map_location = map_location
        self.query_field = query_field
        self.samples = load_ssa_manifest(self.manifest_path)

    def __len__(self) -> int:
        return len(self.samples)

    def _query_text(self, sample: SSASample) -> str:
        if self.query_field == "student_text":
            return sample.student_text()
        if self.query_field == "explicit_text":
            return sample.explicit_text()
        return sample.task_prompt

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        latent_path = _resolve_latent_path(sample, self.root_dir)
        latent_tensor = torch.load(latent_path, map_location=self.map_location)
        return {
            "task_prompt": self._query_text(sample),
            "student_prompt": sample.student_text(),
            "latent_tensor": latent_tensor,
            "sample_index": index,
            "sample_id": sample.task_prompt,
            "target_text": sample.target_text,
            "metadata": dict(sample.metadata),
        }


def retrieval_collate(items: Sequence[dict[str, Any]]) -> RetrievalBatch:
    return RetrievalBatch(
        task_prompts=[str(item["task_prompt"]) for item in items],
        student_prompts=[str(item["student_prompt"]) for item in items],
        latent_tensors=[item["latent_tensor"] for item in items],
        sample_indices=[int(item["sample_index"]) for item in items],
        sample_ids=[str(item.get("sample_id") or item["task_prompt"]) for item in items],
        target_texts=[item.get("target_text") for item in items],
        metadata=[dict(item.get("metadata") or {}) for item in items],
    )
