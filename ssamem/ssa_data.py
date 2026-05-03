"""Backward-compatible imports for SSA data utilities.

New code should prefer `ssamem.training.data`.
"""

from ssamem.training.data import *  # noqa: F401,F403
from ssamem.training.data import (  # noqa: F401
    _record_from_agent_trajectory,
    _record_from_kodcode,
    _record_from_popqa,
)
