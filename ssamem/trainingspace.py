"""Backward-compatible imports for training utilities.

New code should prefer `ssamem.training.space`.
"""

from ssamem.training.space import *  # noqa: F401,F403
from ssamem.training.space import _jsonl_rows, _resolve_latent_path  # noqa: F401

