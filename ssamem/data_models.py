from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Dict, Iterable, Optional
from uuid import uuid4

import torch

PTR_PATTERN = re.compile(r"<PTR_0x[0-9A-Fa-f]+>")


@dataclass
class LatentTensor:
    tensor_data: torch.Tensor
    key_vector: torch.Tensor
    utility_score: float = 1.0
    storage_id: str = field(default_factory=lambda: uuid4().hex)
    pointer: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.tensor_data.dim() != 2:
            raise ValueError(
                "`tensor_data` must have shape [K, D], "
                f"but received {tuple(self.tensor_data.shape)}."
            )
        if self.key_vector.dim() != 1:
            raise ValueError(
                "`key_vector` must have shape [D_key], "
                f"but received {tuple(self.key_vector.shape)}."
            )
        self.utility_score = float(self.utility_score)

    @property
    def latent_length(self) -> int:
        return int(self.tensor_data.size(0))

    @property
    def hidden_size(self) -> int:
        return int(self.tensor_data.size(-1))


@dataclass
class PageTable:
    mapping: Dict[str, str] = field(default_factory=dict)
    _next_index: int = 1

    def allocate(self, storage_id: str) -> str:
        pointer = f"<PTR_0x{self._next_index:03X}>"
        self.mapping[pointer] = storage_id
        self._next_index += 1
        return pointer

    def resolve(self, pointer: str) -> str:
        if pointer not in self.mapping:
            raise KeyError(f"Pointer {pointer} is not present in the page table.")
        return self.mapping[pointer]

    def resolve_many(self, pointers: Iterable[str]) -> list[str]:
        return [self.resolve(pointer) for pointer in pointers]


@dataclass
class AgentMessage:
    sender_id: str
    receiver_id: str
    content: str
    metadata: dict = field(default_factory=dict)

    def extract_pointers(self) -> list[str]:
        return PTR_PATTERN.findall(self.content)


@dataclass
class PointerSearchHit:
    pointer: str
    score: float
    latent: LatentTensor


@dataclass
class ResolvedIPCMessage:
    message: AgentMessage
    explicit_hits: list[PointerSearchHit] = field(default_factory=list)
    prefetched_hits: list[PointerSearchHit] = field(default_factory=list)

    @property
    def mounted_latents(self) -> list[torch.Tensor]:
        hits = self.explicit_hits + self.prefetched_hits
        return [hit.latent.tensor_data for hit in hits]


@dataclass
class ConsolidationResult:
    stored: bool
    pointer: Optional[str] = None
    latent: Optional[LatentTensor] = None
    global_reward: float = 0.0
    reason: Optional[str] = None


@dataclass
class PipelineRunResult:
    resolved_message: ResolvedIPCMessage
    prompt: str
    output_text: str
    mounted_pointer_ids: list[str]


@dataclass
class MASTurn:
    role: str
    upstream_roles: list[str] = field(default_factory=list)
    prompt: str = ""
    response: str = ""
    mounted_pointer_ids: list[str] = field(default_factory=list)


@dataclass
class MASExecutionTrace:
    architecture: str
    task_description: str
    turns: list[MASTurn] = field(default_factory=list)
    final_output: Optional[str] = None
