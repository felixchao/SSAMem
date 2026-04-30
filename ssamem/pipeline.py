from __future__ import annotations

from typing import Optional

import torch
from transformers import GenerationConfig

from ssamem.config import PipelineConfig
from ssamem.data_models import AgentMessage, MASExecutionTrace, PipelineRunResult
from ssamem.kernelspace import MemoryAgent, OSKernel
from ssamem.retrieval import build_retriever
from ssamem.storage import PersistentMemoryBackend
from ssamem.userspace import UserSpaceMAS, build_deterministic_generation_config


class PointerDrivenSSAMemPipeline:
    def __init__(
        self,
        userspace: UserSpaceMAS,
        kernel: OSKernel,
        *,
        top_k_prefetch: int = 1,
        max_new_tokens: int = 32,
        strip_pointer_tokens: bool = True,
    ) -> None:
        self.userspace = userspace
        self.kernel = kernel
        self.top_k_prefetch = top_k_prefetch
        self.max_new_tokens = max_new_tokens
        self.strip_pointer_tokens = strip_pointer_tokens

    @classmethod
    def from_config(cls, config: PipelineConfig) -> "PointerDrivenSSAMemPipeline":
        userspace = UserSpaceMAS.from_config(config)
        persistence = None
        if config.kernel.storage_root:
            persistence = PersistentMemoryBackend(config.kernel.storage_root)
        key_dim = config.kernel.key_dim or userspace.hidden_size
        memory_agent = MemoryAgent(
            hidden_size=userspace.hidden_size,
            key_dim=key_dim,
            reward_threshold=config.kernel.reward_threshold,
            latent_window=config.kernel.latent_window,
            persistence=persistence,
            autosave=config.kernel.autosave,
        )
        retriever = build_retriever(
            config.kernel.retriever_type,
            key_dim=memory_agent.key_dim,
            hash_vocab_size=config.kernel.hash_vocab_size,
        )
        if persistence is not None and config.kernel.autoload:
            memory_agent.load_from_disk(map_location=userspace.device)
        kernel = OSKernel(memory_agent=memory_agent, retriever=retriever)
        return cls(
            userspace=userspace,
            kernel=kernel,
            top_k_prefetch=config.kernel.top_k_prefetch,
            max_new_tokens=config.max_new_tokens,
            strip_pointer_tokens=config.strip_pointer_tokens,
        )

    def register_memory_tensor(
        self,
        tensor_data: torch.Tensor,
        *,
        utility_score: float = 1.0,
        metadata: Optional[dict] = None,
    ) -> str:
        return self.kernel.memory_agent.add_tensor(
            tensor_data=tensor_data,
            utility_score=utility_score,
            metadata=metadata,
        )

    def build_prompt_for_userspace(self, content: str) -> str:
        return self.userspace.build_prompt_for_role(content)

    def run_message(
        self,
        *,
        sender_id: str,
        receiver_id: str,
        content: str,
        generation_config: Optional[GenerationConfig] = None,
        top_k_prefetch: Optional[int] = None,
    ) -> PipelineRunResult:
        ipc_message = AgentMessage(sender_id=sender_id, receiver_id=receiver_id, content=content)
        resolved = self.kernel.handle_ipc(
            ipc_message,
            top_k_prefetch=self.top_k_prefetch if top_k_prefetch is None else top_k_prefetch,
        )
        userspace_prompt = self.build_prompt_for_userspace(content)

        if generation_config is None:
            generation_config = build_deterministic_generation_config(
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.userspace.tokenizer.pad_token_id,
                eos_token_id=self.userspace.tokenizer.eos_token_id,
            )

        generation = self.userspace.run_role_turn(
            content=userspace_prompt,
            mounted_latents=resolved.mounted_latents,
            generation_config=generation_config,
        )
        explicit_pointer_ids = [hit.pointer for hit in resolved.explicit_hits]
        prefetched_pointer_ids = [hit.pointer for hit in resolved.prefetched_hits]
        prefetched_scores = [hit.score for hit in resolved.prefetched_hits]
        mounted_pointer_ids = [hit.pointer for hit in resolved.explicit_hits + resolved.prefetched_hits]
        mounted_latent_count = len(resolved.mounted_latents)
        mounted_latent_tokens = sum(int(latent.size(0)) for latent in resolved.mounted_latents)
        return PipelineRunResult(
            resolved_message=resolved,
            prompt=userspace_prompt,
            output_text=generation.text,
            mounted_pointer_ids=mounted_pointer_ids,
            explicit_pointer_ids=explicit_pointer_ids,
            prefetched_pointer_ids=prefetched_pointer_ids,
            prefetched_scores=prefetched_scores,
            retrieval_query=ipc_message.content,
            mounted_latent_count=mounted_latent_count,
            mounted_latent_tokens=mounted_latent_tokens,
        )

    def consolidate_episode(
        self,
        hidden_states: torch.Tensor,
        global_reward: float,
        *,
        metadata: Optional[dict] = None,
    ):
        return self.kernel.consolidate(
            hidden_states=hidden_states,
            global_reward=global_reward,
            metadata=metadata,
        )

    def save_memory_store(self) -> None:
        self.kernel.memory_agent.save_to_disk()

    def load_memory_store(self) -> None:
        self.kernel.memory_agent.load_from_disk(map_location=self.userspace.device)

    def run_multi_agent_task(
        self,
        task_description: str,
        *,
        mas_style: str = "camel",
        task_domain: Optional[str] = None,
        memory_content: str = "[mounted-latent-memory]",
        generation_config: Optional[GenerationConfig] = None,
        top_k_prefetch: Optional[int] = None,
    ) -> MASExecutionTrace:
        return self.userspace.run_multi_agent_task(
            task_description=task_description,
            kernel=self.kernel,
            mas_style=mas_style,
            task_domain=task_domain,
            memory_content=memory_content,
            generation_config=generation_config,
            top_k_prefetch=self.top_k_prefetch if top_k_prefetch is None else top_k_prefetch,
        )
