from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch

from latent_os.data_models import LatentTensor, PageTable


class DiskPageTableStore:
    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.page_table_path = self.root_dir / "page_table.json"

    def save(self, page_table: PageTable) -> None:
        payload = {
            "mapping": page_table.mapping,
            "next_index": page_table._next_index,
        }
        self.page_table_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    def load(self) -> PageTable:
        if not self.page_table_path.exists():
            return PageTable()
        payload = json.loads(self.page_table_path.read_text())
        return PageTable(
            mapping=payload.get("mapping", {}),
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
        for storage_id in page_table.mapping.values():
            latent_path = self.latent_path(storage_id)
            if latent_path.exists():
                loaded[storage_id] = self.load_latent(storage_id, map_location=map_location)
        return loaded

    def save_all(self, memory_store: dict[str, LatentTensor]) -> None:
        for latent in memory_store.values():
            self.save_latent(latent)


class PersistentMemoryBackend:
    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
        self.page_table_store = DiskPageTableStore(self.root_dir)
        self.latent_store = DiskLatentStore(self.root_dir)

    def save(self, page_table: PageTable, memory_store: dict[str, LatentTensor]) -> None:
        self.page_table_store.save(page_table)
        self.latent_store.save_all(memory_store)

    def save_page_table(self, page_table: PageTable) -> None:
        self.page_table_store.save(page_table)

    def save_latent(self, latent: LatentTensor) -> None:
        self.latent_store.save_latent(latent)

    def load(
        self,
        *,
        map_location: Optional[str | torch.device] = "cpu",
    ) -> tuple[PageTable, dict[str, LatentTensor]]:
        page_table = self.page_table_store.load()
        memory_store = self.latent_store.load_all(page_table, map_location=map_location)
        return page_table, memory_store
