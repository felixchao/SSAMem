from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch

from ssamem.data_models import ClusterMemory, LatentTensor, MemoryCluster, PageTable, PageTableEntry


class DiskPageTableStore:
    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.page_table_path = self.root_dir / "page_table.json"

    def save(self, page_table: PageTable) -> None:
        payload = {
            "mapping": {
                pointer: entry.to_payload()
                for pointer, entry in page_table.mapping.items()
            },
            "next_index": page_table._next_index,
        }
        self.page_table_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    def load(self) -> PageTable:
        if not self.page_table_path.exists():
            return PageTable()
        payload = json.loads(self.page_table_path.read_text())
        raw_mapping = payload.get("mapping", {})
        return PageTable(
            mapping={
                pointer: PageTableEntry.from_payload(pointer, entry_payload)
                for pointer, entry_payload in raw_mapping.items()
            },
            _next_index=int(payload.get("next_index", 1)),
        )


class DiskLatentStore:
    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.latents_dir = self.root_dir / "latents"
        self.latents_dir.mkdir(parents=True, exist_ok=True)

    def latent_path(self, storage_id: str) -> Path:
        return self.latents_dir / f"{storage_id}.pt"

    def save_latent(self, latent: LatentTensor) -> Path:
        payload = {
            "tensor_data": latent.tensor_data.detach().cpu(),
            "key_vector": latent.key_vector.detach().cpu(),
            "utility_score": float(latent.utility_score),
            "storage_id": latent.storage_id,
            "pointer": latent.pointer,
            "metadata": latent.metadata,
        }
        output_path = self.latent_path(latent.storage_id)
        torch.save(payload, output_path)
        return output_path

    def load_latent(
        self,
        storage_id: str,
        *,
        map_location: Optional[str | torch.device] = "cpu",
    ) -> LatentTensor:
        payload = torch.load(self.latent_path(storage_id), map_location=map_location)
        return LatentTensor(
            tensor_data=payload["tensor_data"],
            key_vector=payload["key_vector"],
            utility_score=float(payload["utility_score"]),
            storage_id=payload["storage_id"],
            pointer=payload.get("pointer"),
            metadata=payload.get("metadata", {}),
        )

    def load_all(
        self,
        page_table: PageTable,
        *,
        map_location: Optional[str | torch.device] = "cpu",
    ) -> dict[str, LatentTensor]:
        loaded: dict[str, LatentTensor] = {}
        for storage_id in page_table.iter_storage_ids():
            latent_path = self.latent_path(storage_id)
            if latent_path.exists():
                loaded[storage_id] = self.load_latent(storage_id, map_location=map_location)
        return loaded

    def save_all(self, memory_store: dict[str, LatentTensor]) -> None:
        for latent in memory_store.values():
            self.save_latent(latent)


class DiskClusterStore:
    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.clusters_dir = self.root_dir / "clusters"
        self.clusters_dir.mkdir(parents=True, exist_ok=True)

    def cluster_path(self, cluster_id: str) -> Path:
        return self.clusters_dir / f"{cluster_id}.pt"

    def save_cluster(self, cluster: MemoryCluster) -> Path:
        payload = {
            "cluster_id": cluster.cluster_id,
            "pointer_id": cluster.pointer_id,
            "summary_key_text": cluster.summary_key_text,
            "summary_key_embedding": None
            if cluster.summary_key_embedding is None
            else cluster.summary_key_embedding.detach().cpu(),
            "centroid_embedding": None
            if cluster.centroid_embedding is None
            else cluster.centroid_embedding.detach().cpu(),
            "utility_score": float(cluster.utility_score),
            "version": int(cluster.version),
            "metadata": cluster.metadata,
            "memories": [
                {
                    "local_index": int(memory.local_index),
                    "latent_tensor": {
                        "tensor_data": memory.latent_tensor.tensor_data.detach().cpu(),
                        "key_vector": memory.latent_tensor.key_vector.detach().cpu(),
                        "utility_score": float(memory.latent_tensor.utility_score),
                        "storage_id": memory.latent_tensor.storage_id,
                        "pointer": memory.latent_tensor.pointer,
                        "metadata": memory.latent_tensor.metadata,
                    },
                    "memory_key_embedding": None
                    if memory.memory_key_embedding is None
                    else memory.memory_key_embedding.detach().cpu(),
                    "memory_summary": memory.memory_summary,
                    "source_agent": memory.source_agent,
                    "timestamp": memory.timestamp,
                    "utility_score": float(memory.utility_score),
                    "metadata": memory.metadata,
                }
                for memory in cluster.memories
            ],
        }
        output_path = self.cluster_path(cluster.cluster_id)
        torch.save(payload, output_path)
        return output_path

    def load_cluster(
        self,
        cluster_id: str,
        *,
        map_location: Optional[str | torch.device] = "cpu",
    ) -> MemoryCluster:
        payload = torch.load(self.cluster_path(cluster_id), map_location=map_location)
        memories = []
        for memory_payload in payload.get("memories", []):
            latent_payload = memory_payload["latent_tensor"]
            latent = LatentTensor(
                tensor_data=latent_payload["tensor_data"],
                key_vector=latent_payload["key_vector"],
                utility_score=float(latent_payload["utility_score"]),
                storage_id=latent_payload["storage_id"],
                pointer=latent_payload.get("pointer"),
                metadata=latent_payload.get("metadata", {}),
            )
            memories.append(
                ClusterMemory(
                    local_index=int(memory_payload["local_index"]),
                    latent_tensor=latent,
                    memory_key_embedding=memory_payload.get("memory_key_embedding"),
                    memory_summary=str(memory_payload.get("memory_summary", "")),
                    source_agent=memory_payload.get("source_agent"),
                    timestamp=memory_payload.get("timestamp"),
                    utility_score=float(memory_payload.get("utility_score", latent.utility_score)),
                    metadata=dict(memory_payload.get("metadata") or {}),
                )
            )
        return MemoryCluster(
            cluster_id=payload["cluster_id"],
            pointer_id=payload.get("pointer_id"),
            summary_key_text=str(payload.get("summary_key_text", "")),
            summary_key_embedding=payload.get("summary_key_embedding"),
            centroid_embedding=payload.get("centroid_embedding"),
            memories=memories,
            utility_score=float(payload.get("utility_score", 1.0)),
            version=int(payload.get("version", 1)),
            metadata=dict(payload.get("metadata") or {}),
        )

    def load_all(
        self,
        *,
        map_location: Optional[str | torch.device] = "cpu",
    ) -> dict[str, MemoryCluster]:
        loaded: dict[str, MemoryCluster] = {}
        for cluster_path in sorted(self.clusters_dir.glob("*.pt")):
            cluster = self.load_cluster(cluster_path.stem, map_location=map_location)
            loaded[cluster.cluster_id] = cluster
        return loaded

    def save_all(self, memory_clusters: dict[str, MemoryCluster]) -> None:
        for cluster in memory_clusters.values():
            self.save_cluster(cluster)


class PersistentMemoryBackend:
    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
        self.page_table_store = DiskPageTableStore(self.root_dir)
        self.latent_store = DiskLatentStore(self.root_dir)
        self.cluster_store = DiskClusterStore(self.root_dir)

    def save(
        self,
        page_table: PageTable,
        memory_store: dict[str, LatentTensor],
        memory_clusters: Optional[dict[str, MemoryCluster]] = None,
    ) -> None:
        self.page_table_store.save(page_table)
        if memory_clusters:
            self.cluster_store.save_all(memory_clusters)
        self.latent_store.save_all(memory_store)

    def save_page_table(self, page_table: PageTable) -> None:
        self.page_table_store.save(page_table)

    def save_latent(self, latent: LatentTensor) -> None:
        self.latent_store.save_latent(latent)

    def save_cluster(self, cluster: MemoryCluster) -> None:
        self.cluster_store.save_cluster(cluster)

    def load(
        self,
        *,
        map_location: Optional[str | torch.device] = "cpu",
    ) -> tuple[PageTable, dict[str, LatentTensor], dict[str, MemoryCluster]]:
        page_table = self.page_table_store.load()
        memory_clusters = self.cluster_store.load_all(map_location=map_location)
        memory_store: dict[str, LatentTensor] = {}
        for cluster in memory_clusters.values():
            for memory in cluster.memories:
                memory_store[memory.storage_id] = memory.latent_tensor
        legacy_memory_store = self.latent_store.load_all(page_table, map_location=map_location)
        for storage_id, latent in legacy_memory_store.items():
            memory_store.setdefault(storage_id, latent)
        return page_table, memory_store, memory_clusters
