from __future__ import annotations

from ssamem.cli import build_parser
from ssamem.commands.runtime_helpers import configure_runtime
from ssamem.commands.runtime import CORE_COMMAND_HANDLERS
from ssamem.commands.tooling import TOOLING_COMMAND_HANDLERS, TORCH_FREE_COMMANDS
from ssamem.config import RunConfig


COMMAND_HANDLERS = {**CORE_COMMAND_HANDLERS, **TOOLING_COMMAND_HANDLERS}


def _requires_torch(args) -> bool:
    if args.command == "build-ssa-data" and getattr(args, "no_latents", False):
        return False
    return args.command not in TORCH_FREE_COMMANDS


def _ensure_runtime_dependencies(args) -> None:
    if not _requires_torch(args):
        return
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "This ssamem command requires `torch` and `transformers` in the active Python environment. "
            "Install dependencies from `ssamem/requirements.txt` first."
        ) from exc


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _ensure_runtime_dependencies(args)
    configure_runtime(args)
    if args.output_dir:
        RunConfig(output_dir=args.output_dir).ensure_output_dir()
    COMMAND_HANDLERS[args.command](args)


if __name__ == "__main__":
    main()
