from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any, Dict, Iterable, Optional
from uuid import uuid4

import torch

PTR_PATTERN = re.compile(r"<PTR_0x[0-9A-Fa-f]+>")
PTR_ADDRESS_PATTERN = re.compile(r"(<PTR_0x[0-9A-Fa-f]+>)(?::(\d+))?")


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
class PointerAddress:
    pointer: str
    local_index: Optional[int] = None

    @classmethod
    def parse(cls, raw: str) -> "PointerAddress":
        match = PTR_ADDRESS_PATTERN.fullmatch(raw.strip())
        if match is None:
            raise ValueError(f"Invalid pointer address: {raw}")
        pointer = match.group(1)
        local_index = match.group(2)
        return cls(pointer=pointer, local_index=None if local_index is None else int(local_index))

    def to_string(self) -> str:
        if self.local_index is None:
            return self.pointer
        return f"{self.pointer}:{self.local_index:04d}"


@dataclass
class ClusterMemory:
    local_index: int
    latent_tensor: LatentTensor
    memory_key_embedding: Optional[torch.Tensor] = None
    memory_summary: str = ""
    source_agent: Optional[str] = None
    timestamp: Optional[str] = None
    utility_score: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.utility_score = float(self.utility_score)
        if self.memory_key_embedding is not None and self.memory_key_embedding.dim() != 1:
            raise ValueError(
                "`memory_key_embedding` must have shape [D_key], "
                f"but received {tuple(self.memory_key_embedding.shape)}."
            )

    @property
    def storage_id(self) -> str:
        return self.latent_tensor.storage_id


@dataclass
class MemoryCluster:
    cluster_id: str = field(default_factory=lambda: uuid4().hex)
    pointer_id: Optional[str] = None
    summary_key_text: str = ""
    summary_key_embedding: Optional[torch.Tensor] = None
    centroid_embedding: Optional[torch.Tensor] = None
    memories: list[ClusterMemory] = field(default_factory=list)
    utility_score: float = 1.0
    version: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.utility_score = float(self.utility_score)
        if self.summary_key_embedding is not None and self.summary_key_embedding.dim() != 1:
            raise ValueError(
                "`summary_key_embedding` must have shape [D_key], "
                f"but received {tuple(self.summary_key_embedding.shape)}."
            )
        if self.centroid_embedding is not None and self.centroid_embedding.dim() != 1:
            raise ValueError(
                "`centroid_embedding` must have shape [D_key], "
                f"but received {tuple(self.centroid_embedding.shape)}."
            )
        self.memories.sort(key=lambda memory: memory.local_index)

    @property
    def cluster_size(self) -> int:
        return len(self.memories)

    def next_local_index(self) -> int:
        if not self.memories:
            return 0
        return max(memory.local_index for memory in self.memories) + 1

    def append_memory(self, memory: ClusterMemory) -> None:
        if any(existing.local_index == memory.local_index for existing in self.memories):
            raise ValueError(
                f"Cluster {self.cluster_id} already contains local index {memory.local_index}."
            )
        self.memories.append(memory)
        self.memories.sort(key=lambda item: item.local_index)

    def get_memory(self, local_index: int) -> ClusterMemory:
        for memory in self.memories:
            if memory.local_index == local_index:
                return memory
        raise KeyError(f"Cluster {self.cluster_id} does not contain local index {local_index}.")


@dataclass
class PageTableEntry:
    pointer_id: str
    cluster_id: str
    summary_key_text: str = ""
    summary_key_embedding: Optional[torch.Tensor] = None
    cluster_size: int = 0
    utility_score: float = 1.0
    last_update_step: Optional[int] = None
    version: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)
    legacy_storage_id: Optional[str] = None

    def __post_init__(self) -> None:
        self.utility_score = float(self.utility_score)
        if self.summary_key_embedding is not None and self.summary_key_embedding.dim() != 1:
            raise ValueError(
                "`summary_key_embedding` must have shape [D_key], "
                f"but received {tuple(self.summary_key_embedding.shape)}."
            )

    @property
    def storage_ref(self) -> str:
        return self.legacy_storage_id or self.cluster_id

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.summary_key_embedding is not None:
            payload["summary_key_embedding"] = self.summary_key_embedding.detach().cpu().tolist()
        return payload

    @classmethod
    def from_payload(cls, pointer_id: str, payload: Any) -> "PageTableEntry":
        if isinstance(payload, str):
            return cls(
                pointer_id=pointer_id,
                cluster_id=payload,
                cluster_size=1,
                legacy_storage_id=payload,
            )
        summary_embedding = payload.get("summary_key_embedding")
        if summary_embedding is not None:
            summary_embedding = torch.tensor(summary_embedding, dtype=torch.float32)
        return cls(
            pointer_id=payload.get("pointer_id", pointer_id),
            cluster_id=payload.get("cluster_id") or payload.get("storage_id") or pointer_id,
            summary_key_text=payload.get("summary_key_text", ""),
            summary_key_embedding=summary_embedding,
            cluster_size=int(payload.get("cluster_size", 0)),
            utility_score=float(payload.get("utility_score", 1.0)),
            last_update_step=payload.get("last_update_step"),
            version=int(payload.get("version", 1)),
            metadata=dict(payload.get("metadata") or {}),
            legacy_storage_id=payload.get("legacy_storage_id"),
        )


@dataclass
class PageTable:
    mapping: Dict[str, PageTableEntry] = field(default_factory=dict)
    _next_index: int = 1

    def allocate(self, storage_id: str) -> str:
        pointer = f"<PTR_0x{self._next_index:03X}>"
        self.mapping[pointer] = PageTableEntry(
            pointer_id=pointer,
            cluster_id=storage_id,
            cluster_size=1,
            legacy_storage_id=storage_id,
        )
        self._next_index += 1
        return pointer

    def allocate_cluster(
        self,
        cluster_id: str,
        *,
        summary_key_text: str = "",
        summary_key_embedding: Optional[torch.Tensor] = None,
        cluster_size: int = 0,
        utility_score: float = 1.0,
        last_update_step: Optional[int] = None,
        version: int = 1,
        metadata: Optional[dict[str, Any]] = None,
    ) -> PageTableEntry:
        pointer = f"<PTR_0x{self._next_index:03X}>"
        entry = PageTableEntry(
            pointer_id=pointer,
            cluster_id=cluster_id,
            summary_key_text=summary_key_text,
            summary_key_embedding=summary_key_embedding,
            cluster_size=cluster_size,
            utility_score=utility_score,
            last_update_step=last_update_step,
            version=version,
            metadata=dict(metadata or {}),
        )
        self.mapping[pointer] = entry
        self._next_index += 1
        return entry

    def register_entry(self, entry: PageTableEntry) -> None:
        self.mapping[entry.pointer_id] = entry

    def resolve(self, pointer: str) -> str:
        return self.resolve_entry(pointer).storage_ref

    def resolve_entry(self, pointer: str) -> PageTableEntry:
        if pointer not in self.mapping:
            raise KeyError(f"Pointer {pointer} is not present in the page table.")
        return self.mapping[pointer]

    def resolve_many(self, pointers: Iterable[str]) -> list[str]:
        return [self.resolve(pointer) for pointer in pointers]

    def iter_entries(self) -> Iterable[PageTableEntry]:
        return self.mapping.values()

    def iter_storage_ids(self) -> Iterable[str]:
        for entry in self.mapping.values():
            if entry.legacy_storage_id is not None:
                yield entry.legacy_storage_id


@dataclass
class AgentMessage:
    sender_id: str
    receiver_id: str
    content: str
    metadata: dict = field(default_factory=dict)

    def extract_pointers(self) -> list[str]:
        return PTR_PATTERN.findall(self.content)

    def extract_pointer_addresses(self) -> list[PointerAddress]:
        return [PointerAddress(pointer=match.group(1), local_index=None if match.group(2) is None else int(match.group(2))) for match in PTR_ADDRESS_PATTERN.finditer(self.content)]


@dataclass
class PointerSearchHit:
    pointer: str
    score: float
    latent: LatentTensor
    cluster_id: Optional[str] = None
    local_index: Optional[int] = None

    @property
    def exact_address(self) -> str:
        if self.local_index is None:
            return self.pointer
        return f"{self.pointer}:{self.local_index:04d}"


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
    explicit_pointer_ids: list[str] = field(default_factory=list)
    prefetched_pointer_ids: list[str] = field(default_factory=list)
    prefetched_scores: list[float] = field(default_factory=list)
    retrieval_query: str = ""
    mounted_latent_count: int = 0
    mounted_latent_tokens: int = 0


@dataclass
class MASTurn:
    role: str
    upstream_roles: list[str] = field(default_factory=list)
    prompt: str = ""
    response: str = ""
    mounted_pointer_ids: list[str] = field(default_factory=list)
    explicit_pointer_ids: list[str] = field(default_factory=list)
    prefetched_pointer_ids: list[str] = field(default_factory=list)
    prefetched_scores: list[float] = field(default_factory=list)
    retrieval_query: str = ""
    mounted_latent_count: int = 0
    mounted_latent_tokens: int = 0


@dataclass
class MASExecutionTrace:
    architecture: str
    task_description: str
    turns: list[MASTurn] = field(default_factory=list)
    final_output: Optional[str] = None
