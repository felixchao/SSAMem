from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class RuntimeConfig:
    runtime_mode: str = "tiny-random"
    model_name_or_path: Optional[str] = None
    tokenizer_name_or_path: Optional[str] = None
    trust_remote_code: bool = False
    torch_dtype: str = "float32"
    device: str = "cpu"
    load_in_4bit: bool = False
    bnb_4bit_quant_type: str = "nf4"

    # Tiny random Llama config
    vocab_size: int = 512
    hidden_size: int = 128
    intermediate_size: int = 256
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    num_key_value_heads: int = 4
    max_position_embeddings: int = 512


@dataclass
class KernelConfig:
    key_dim: Optional[int] = None
    retriever_type: str = "hash_mips"
    hash_vocab_size: int = 4096
    reward_threshold: float = 0.0
    latent_window: int = 8
    top_k_prefetch: int = 1
    cluster_assignment_threshold: float = 0.95
    cluster_assignment_top_k: int = 4
    storage_root: Optional[str] = None
    autosave: bool = True
    autoload: bool = True


@dataclass
class MASConfig:
    architecture: str = "camel"
    shared_runtime: bool = True
    task_domain: Optional[str] = None


@dataclass
class RunConfig:
    seed: int = 7
    run_name: Optional[str] = None
    output_dir: Optional[str] = None
    log_level: str = "INFO"

    def resolved_run_name(self, command: str) -> str:
        return self.run_name or command.replace("_", "-")

    def ensure_output_dir(self) -> Optional[Path]:
        if not self.output_dir:
            return None
        path = Path(self.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path


@dataclass
class PipelineConfig:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    kernel: KernelConfig = field(default_factory=KernelConfig)
    mas: MASConfig = field(default_factory=MASConfig)
    run: RunConfig = field(default_factory=RunConfig)
    max_new_tokens: int = 32
    strip_pointer_tokens: bool = True
