"""Retrieval-key alignment training for SSAMem."""

from ssamem.retrieval_training.data import RetrievalBatch, RetrievalManifestDataset, retrieval_collate
from ssamem.retrieval_training.model import (
    RetrievalAlignmentModel,
    RetrievalAlignmentOutput,
    RetrievalModelConfig,
)
from ssamem.retrieval_training.trainer import (
    RetrievalEvalResult,
    evaluate_retrieval_alignment,
    load_retrieval_checkpoint,
    save_retrieval_checkpoint,
    train_retrieval_alignment,
)

__all__ = [
    "RetrievalBatch",
    "RetrievalManifestDataset",
    "retrieval_collate",
    "RetrievalAlignmentModel",
    "RetrievalAlignmentOutput",
    "RetrievalModelConfig",
    "RetrievalEvalResult",
    "evaluate_retrieval_alignment",
    "load_retrieval_checkpoint",
    "save_retrieval_checkpoint",
    "train_retrieval_alignment",
]
