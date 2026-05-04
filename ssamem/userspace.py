from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Optional, Sequence

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, LlamaConfig, LlamaForCausalLM

from ssamem.config import PipelineConfig, RuntimeConfig
from ssamem.data_models import AgentMessage, MASExecutionTrace, MASTurn
from ssamem.mas import DEFAULT_MEMORY_CONTENT, build_mas_topology
from ssamem.memory_actions import (
    build_answer_prompt,
    build_memory_request_prompt,
    format_memory_observation,
    parse_memory_request,
)
from ssamem.tokenizer import SimpleTokenizer


def freeze_model(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False


def resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    if dtype_name not in dtype_map:
        raise ValueError(f"Unsupported torch dtype '{dtype_name}'.")
    return dtype_map[dtype_name]


def build_deterministic_generation_config(
    *,
    max_new_tokens: int,
    pad_token_id: int,
    eos_token_id: int,
) -> GenerationConfig:
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
    )
    generation_config.temperature = None
    generation_config.top_k = None
    generation_config.top_p = None
    generation_config.min_p = None
    generation_config.typical_p = None
    return generation_config


@dataclass
class SoftPromptBatch:
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    input_ids: torch.Tensor
    prefix_lengths: list[int]
    text_lengths: list[int]


@dataclass
class UserSpaceGenerationOutput:
    text: str
    sequences: torch.Tensor
    prompt_token_count: int
    mounted_latent_tokens: int


class _RoleRuntime(nn.Module):
    """
    Private low-level execution runtime.

    This is the old single-model execution concept, but it is now intentionally hidden behind
    `UserSpaceMAS` so the public user-space abstraction is MAS-first.
    """

    def __init__(self, model: nn.Module, tokenizer, freeze_backbone: bool = True) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

        if getattr(self.tokenizer, "pad_token", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

        if freeze_backbone:
            freeze_model(self.model)
            self.model.eval()

    @classmethod
    def from_config(cls, config: RuntimeConfig, freeze_backbone: bool = True) -> "_RoleRuntime":
        if config.runtime_mode == "hf":
            if not config.model_name_or_path:
                raise ValueError("`model_name_or_path` is required when runtime_mode='hf'.")
            dtype = resolve_torch_dtype(config.torch_dtype)
            model_kwargs = {
                "torch_dtype": dtype,
                "trust_remote_code": config.trust_remote_code,
            }
            if config.load_in_4bit:
                try:
                    from transformers import BitsAndBytesConfig
                except ModuleNotFoundError as exc:
                    raise RuntimeError("4-bit loading requires `bitsandbytes` support in transformers.") from exc
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type=config.bnb_4bit_quant_type,
                    bnb_4bit_compute_dtype=dtype,
                )
            model = AutoModelForCausalLM.from_pretrained(
                config.model_name_or_path,
                **model_kwargs,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                config.tokenizer_name_or_path or config.model_name_or_path,
                trust_remote_code=config.trust_remote_code,
            )
            if getattr(model, "generation_config", None) is not None:
                model.generation_config.do_sample = False
                model.generation_config.temperature = None
                model.generation_config.top_k = None
                model.generation_config.top_p = None
                model.generation_config.min_p = None
                model.generation_config.typical_p = None
        elif config.runtime_mode == "tiny-random":
            llama_config = LlamaConfig(
                vocab_size=config.vocab_size,
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_hidden_layers=config.num_hidden_layers,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                max_position_embeddings=config.max_position_embeddings,
                bos_token_id=1,
                eos_token_id=2,
                pad_token_id=0,
            )
            model = LlamaForCausalLM(llama_config)
            tokenizer = SimpleTokenizer(vocab_size=config.vocab_size)
        else:
            raise ValueError(f"Unsupported runtime mode: {config.runtime_mode}")

        runtime = cls(model=model, tokenizer=tokenizer, freeze_backbone=freeze_backbone)
        runtime.to(torch.device(config.device))
        return runtime

    @property
    def device(self) -> torch.device:
        return self.model.device

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)

    def _canonicalize_mounted_latents(
        self,
        prompts: Sequence[str],
        mounted_latents: Optional[Sequence[Sequence[torch.Tensor]] | Sequence[torch.Tensor]],
    ) -> list[list[torch.Tensor]]:
        batch_size = len(prompts)
        if mounted_latents is None:
            return [[] for _ in range(batch_size)]
        if len(mounted_latents) == 0:
            return [[] for _ in range(batch_size)]

        if batch_size == 1 and mounted_latents and isinstance(mounted_latents[0], torch.Tensor):
            return [list(mounted_latents)]  # type: ignore[arg-type]

        latent_batches = mounted_latents  # type: ignore[assignment]
        if len(latent_batches) != batch_size:
            raise ValueError(
                "`mounted_latents` must align with the prompt batch size. "
                f"Expected {batch_size}, got {len(latent_batches)}."
            )
        return [list(latents) for latents in latent_batches]

    def _merge_latent_prefix(self, latent_tensors: Sequence[torch.Tensor]) -> torch.Tensor:
        if not latent_tensors:
            return torch.empty(0, self.hidden_size, device=self.device, dtype=self.model.dtype)

        prepared = []
        for latent in latent_tensors:
            if latent.dim() == 3:
                if latent.size(0) != 1:
                    raise ValueError(
                        "Batched latent tensors are not supported inside a single role sample. "
                        f"Received {tuple(latent.shape)}."
                    )
                latent = latent.squeeze(0)
            if latent.dim() != 2:
                raise ValueError(
                    "Each latent tensor must have shape [K, D] or [1, K, D], "
                    f"but received {tuple(latent.shape)}."
                )
            if latent.size(-1) != self.hidden_size:
                raise ValueError(
                    "Latent hidden size must match the user-space hidden size. "
                    f"Expected {self.hidden_size}, got {latent.size(-1)}."
                )
            prepared.append(latent.to(device=self.device, dtype=self.model.dtype))

        return torch.cat(prepared, dim=0)

    def prepare_soft_prompt_batch(
        self,
        prompts: Sequence[str],
        mounted_latents: Optional[Sequence[Sequence[torch.Tensor]] | Sequence[torch.Tensor]] = None,
    ) -> SoftPromptBatch:
        latent_batches = self._canonicalize_mounted_latents(prompts, mounted_latents)
        tokenized = self.tokenizer(
            list(prompts),
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized["attention_mask"].to(self.device)

        embedding_layer = self.model.get_input_embeddings()
        text_embeds = embedding_layer(input_ids).to(self.model.dtype)
        pad_embed = embedding_layer(
            torch.tensor([[self.tokenizer.pad_token_id]], device=self.device)
        ).to(self.model.dtype)

        merged_embeds = []
        merged_masks = []
        prefix_lengths = []
        text_lengths = attention_mask.sum(dim=1).tolist()

        for idx, latent_group in enumerate(latent_batches):
            latent_prefix = self._merge_latent_prefix(latent_group)
            prefix_len = int(latent_prefix.size(0))
            prefix_lengths.append(prefix_len)

            text_len = int(text_lengths[idx])
            sample_embeds = text_embeds[idx, -text_len:, :].unsqueeze(0)
            sample_mask = torch.ones((1, text_len), dtype=attention_mask.dtype, device=self.device)

            if prefix_len > 0:
                prefix_embeds = latent_prefix.unsqueeze(0)
                prefix_mask = torch.ones((1, prefix_len), dtype=sample_mask.dtype, device=self.device)
                sample_embeds = torch.cat([prefix_embeds, sample_embeds], dim=1)
                sample_mask = torch.cat([prefix_mask, sample_mask], dim=1)

            merged_embeds.append(sample_embeds)
            merged_masks.append(sample_mask)

        max_seq_len = max(sample.size(1) for sample in merged_embeds)
        padded_embeds = []
        padded_masks = []
        for sample_embeds, sample_mask in zip(merged_embeds, merged_masks):
            pad_len = max_seq_len - sample_embeds.size(1)
            if pad_len > 0:
                sample_embeds = torch.cat([pad_embed.repeat(1, pad_len, 1), sample_embeds], dim=1)
                mask_pad = torch.zeros((1, pad_len), dtype=sample_mask.dtype, device=self.device)
                sample_mask = torch.cat([mask_pad, sample_mask], dim=1)
            padded_embeds.append(sample_embeds)
            padded_masks.append(sample_mask)

        return SoftPromptBatch(
            inputs_embeds=torch.cat(padded_embeds, dim=0),
            attention_mask=torch.cat(padded_masks, dim=0),
            input_ids=input_ids,
            prefix_lengths=prefix_lengths,
            text_lengths=[int(length) for length in text_lengths],
        )

    @torch.no_grad()
    def encode_text_as_latent(
        self,
        text: str,
        *,
        max_tokens: int = 128,
        strategy: str = "last_hidden",
    ) -> torch.Tensor:
        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=max_tokens,
            return_tensors="pt",
        )
        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized["attention_mask"].to(self.device)

        if strategy == "embeddings":
            embedding_layer = self.model.get_input_embeddings()
            return embedding_layer(input_ids)[0].to(self.model.dtype)

        if strategy != "last_hidden":
            raise ValueError(f"Unsupported latent encoding strategy: {strategy}")

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        return outputs.hidden_states[-1][0].to(self.model.dtype)

    @torch.no_grad()
    def generate_with_mount(
        self,
        prompt: str,
        mounted_latents: Optional[Sequence[torch.Tensor]] = None,
        generation_config: Optional[GenerationConfig] = None,
        **generate_kwargs,
    ) -> UserSpaceGenerationOutput:
        batch = self.prepare_soft_prompt_batch(prompts=[prompt], mounted_latents=mounted_latents)
        generation_output = self.model.generate(
            inputs_embeds=batch.inputs_embeds,
            attention_mask=batch.attention_mask,
            generation_config=generation_config,
            **generate_kwargs,
        )

        max_new_tokens = None
        if generation_config is not None and generation_config.max_new_tokens is not None:
            max_new_tokens = generation_config.max_new_tokens
        elif "max_new_tokens" in generate_kwargs:
            max_new_tokens = int(generate_kwargs["max_new_tokens"])

        if max_new_tokens is not None and generation_output.size(1) > max_new_tokens:
            decode_ids = generation_output[:, -max_new_tokens:]
        else:
            decode_ids = generation_output
        text = self.tokenizer.decode(decode_ids[0], skip_special_tokens=True)

        return UserSpaceGenerationOutput(
            text=text,
            sequences=generation_output,
            prompt_token_count=int(batch.attention_mask[0].sum().item()),
            mounted_latent_tokens=batch.prefix_lengths[0],
        )


class UserSpaceMAS(nn.Module):
    """
    Public user-space abstraction.

    Externally, user-space is a MAS system. Role execution and soft-prompt
    injection are internal details handled by the private `_RoleRuntime`.
    """

    def __init__(
        self,
        runtime: _RoleRuntime,
        *,
        architecture: str = "camel",
        task_domain: str | None = None,
        strip_pointer_tokens: bool = True,
        max_new_tokens: int = 32,
    ) -> None:
        super().__init__()
        self._runtime = runtime
        self.architecture = architecture
        self.task_domain = task_domain
        self.strip_pointer_tokens = strip_pointer_tokens
        self.max_new_tokens = max_new_tokens

    @classmethod
    def from_config(cls, config: PipelineConfig) -> "UserSpaceMAS":
        runtime = _RoleRuntime.from_config(config.runtime)
        return cls(
            runtime=runtime,
            architecture=config.mas.architecture,
            task_domain=config.mas.task_domain,
            strip_pointer_tokens=config.strip_pointer_tokens,
            max_new_tokens=config.max_new_tokens,
        )

    @property
    def device(self) -> torch.device:
        return self._runtime.device

    @property
    def hidden_size(self) -> int:
        return self._runtime.hidden_size

    @property
    def tokenizer(self):
        return self._runtime.tokenizer

    @property
    def model(self):
        return self._runtime.model

    def prepare_soft_prompt_batch(
        self,
        prompts: Sequence[str],
        mounted_latents: Optional[Sequence[Sequence[torch.Tensor]] | Sequence[torch.Tensor]] = None,
    ) -> SoftPromptBatch:
        return self._runtime.prepare_soft_prompt_batch(prompts=prompts, mounted_latents=mounted_latents)

    def encode_text_as_latent(
        self,
        text: str,
        *,
        max_tokens: int = 128,
        strategy: str = "last_hidden",
    ) -> torch.Tensor:
        return self._runtime.encode_text_as_latent(
            text,
            max_tokens=max_tokens,
            strategy=strategy,
        )

    def build_prompt_for_role(self, content: str) -> str:
        if not self.strip_pointer_tokens:
            return content
        return re.sub(r"<PTR_0x[0-9A-Fa-f]+>", "[mounted-memory]", content).strip()

    def run_role_turn(
        self,
        *,
        content: str,
        mounted_latents: Optional[Sequence[torch.Tensor]] = None,
        generation_config: Optional[GenerationConfig] = None,
    ) -> UserSpaceGenerationOutput:
        role_prompt = self.build_prompt_for_role(content)
        if generation_config is None:
            generation_config = build_deterministic_generation_config(
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        return self._runtime.generate_with_mount(
            prompt=role_prompt,
            mounted_latents=mounted_latents,
            generation_config=generation_config,
        )

    def run_multi_agent_task(
        self,
        *,
        task_description: str,
        kernel,
        mas_style: Optional[str] = None,
        task_domain: Optional[str] = None,
        memory_content: str = DEFAULT_MEMORY_CONTENT,
        generation_config: Optional[GenerationConfig] = None,
        top_k_prefetch: int = 0,
    ) -> MASExecutionTrace:
        style = mas_style or self.architecture
        resolved_task_domain = task_domain or self.task_domain
        topology = build_mas_topology(style, task_domain=resolved_task_domain)
        role_outputs: dict[str, str] = {}
        trace = MASExecutionTrace(architecture=topology.architecture, task_description=task_description)

        for role_spec in topology.roles:
            role_prompt = role_spec.render_prompt(
                task_description,
                role_outputs,
                feedback_template=topology.feedback_template,
                memory_content=memory_content or DEFAULT_MEMORY_CONTENT,
            )
            ipc_message = AgentMessage(
                sender_id="kernel" if not role_spec.upstream_roles else ",".join(role_spec.upstream_roles),
                receiver_id=role_spec.role,
                content=role_prompt,
            )
            resolved = kernel.handle_ipc(ipc_message, top_k_prefetch=top_k_prefetch)
            generation = self.run_role_turn(
                content=role_prompt,
                mounted_latents=resolved.mounted_latents,
                generation_config=generation_config,
            )
            role_outputs[role_spec.role] = generation.text
            prefetched_scores = [hit.score for hit in resolved.prefetched_hits]
            mounted_pointer_ids = [hit.exact_address for hit in resolved.explicit_hits + resolved.prefetched_hits]
            explicit_pointer_ids = [hit.exact_address for hit in resolved.explicit_hits]
            prefetched_pointer_ids = [hit.exact_address for hit in resolved.prefetched_hits]
            mounted_latent_count = len(resolved.mounted_latents)
            mounted_latent_tokens = sum(int(latent.size(0)) for latent in resolved.mounted_latents)
            trace.turns.append(
                MASTurn(
                    role=role_spec.role,
                    upstream_roles=list(role_spec.upstream_roles),
                    prompt=self.build_prompt_for_role(role_prompt),
                    response=generation.text,
                    mounted_pointer_ids=mounted_pointer_ids,
                    explicit_pointer_ids=explicit_pointer_ids,
                    prefetched_pointer_ids=prefetched_pointer_ids,
                    prefetched_scores=prefetched_scores,
                    retrieval_query=ipc_message.content,
                    mounted_latent_count=mounted_latent_count,
                    mounted_latent_tokens=mounted_latent_tokens,
                )
            )

        if topology.final_role in role_outputs:
            trace.final_output = role_outputs[topology.final_role]
        elif trace.turns:
            trace.final_output = trace.turns[-1].response
        return trace

    def run_multi_agent_task_with_memory_actions(
        self,
        *,
        task_description: str,
        kernel,
        mas_style: Optional[str] = None,
        task_domain: Optional[str] = None,
        memory_content: str = DEFAULT_MEMORY_CONTENT,
        pointer_table_context: str = "",
        generation_config: Optional[GenerationConfig] = None,
        request_generation_config: Optional[GenerationConfig] = None,
        default_top_k: int = 1,
        memory_request_policy: str = "auto",
    ) -> MASExecutionTrace:
        style = mas_style or self.architecture
        resolved_task_domain = task_domain or self.task_domain
        topology = build_mas_topology(style, task_domain=resolved_task_domain)
        role_outputs: dict[str, str] = {}
        trace = MASExecutionTrace(architecture=topology.architecture, task_description=task_description)

        for role_spec in topology.roles:
            role_prompt = role_spec.render_prompt(
                task_description,
                role_outputs,
                feedback_template=topology.feedback_template,
                memory_content=memory_content or DEFAULT_MEMORY_CONTENT,
            )
            request_prompt = build_memory_request_prompt(
                role_prompt,
                pointer_table_context,
                task_query=task_description,
                default_top_k=default_top_k,
                policy=memory_request_policy,
            )
            request_generation = self.run_role_turn(
                content=request_prompt,
                mounted_latents=[],
                generation_config=request_generation_config or generation_config,
            )
            memory_request = parse_memory_request(
                request_generation.text,
                default_query=task_description,
                default_top_k=default_top_k,
                policy=memory_request_policy,
            )

            hits = []
            if memory_request.mode == "GET" and memory_request.address:
                try:
                    memory = kernel.get_memory(memory_request.address)
                    pointer = memory.latent_tensor.pointer or memory_request.address.split(":", 1)[0]
                    hits = [
                        kernel.memory_agent.resolve_explicit_pointers(
                            [f"{pointer}:{memory.local_index:04d}"]
                        )[0]
                    ]
                except Exception:
                    hits = []
            elif memory_request.mode == "SEARCH":
                hits = kernel.search_memory(memory_request.query or task_description, top_k=memory_request.top_k)

            memory_observation = format_memory_observation(hits)
            answer_prompt = build_answer_prompt(role_prompt, memory_observation)
            generation = self.run_role_turn(
                content=answer_prompt,
                mounted_latents=[hit.latent.tensor_data for hit in hits],
                generation_config=generation_config,
            )
            role_outputs[role_spec.role] = generation.text
            mounted_pointer_ids = [hit.exact_address for hit in hits]
            mounted_latent_tokens = sum(int(hit.latent.tensor_data.size(0)) for hit in hits)
            trace.turns.append(
                MASTurn(
                    role=role_spec.role,
                    upstream_roles=list(role_spec.upstream_roles),
                    prompt=self.build_prompt_for_role(answer_prompt),
                    response=generation.text,
                    mounted_pointer_ids=mounted_pointer_ids,
                    prefetched_pointer_ids=mounted_pointer_ids if memory_request.mode == "SEARCH" else [],
                    explicit_pointer_ids=mounted_pointer_ids if memory_request.mode == "GET" else [],
                    prefetched_scores=[hit.score for hit in hits],
                    retrieval_query=memory_request.query or memory_request.address or "",
                    mounted_latent_count=len(hits),
                    mounted_latent_tokens=mounted_latent_tokens,
                    memory_request={
                        "mode": memory_request.mode,
                        "query": memory_request.query,
                        "address": memory_request.address,
                        "top_k": memory_request.top_k,
                        "raw_text": memory_request.raw_text,
                    },
                    memory_observation=memory_observation,
                    request_response=request_generation.text,
                )
            )

        if topology.final_role in role_outputs:
            trace.final_output = role_outputs[topology.final_role]
        elif trace.turns:
            trace.final_output = trace.turns[-1].response
        return trace
