from __future__ import annotations

import logging


def configure_runtime(args) -> None:
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper()), format="%(levelname)s %(message)s")
