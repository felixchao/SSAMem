from __future__ import annotations

from typing import Optional

import torch
from transformers import GenerationConfig

from latent_os.config import PipelineConfig
from latent_os.data_models import AgentMessage, MASExecutionTrace, PipelineRunResult
from latent_os.kernelspace import MemoryAgent, OSKernel
from latent_os.storage import PersistentMemoryBackend
from latent_os.userspace import UserSpaceMAS


class PointerDrivenLatentOSPipeline:
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
    def from_config(cls, config: PipelineConfig) -> "PointerDrivenLatentOSPipeline":
        userspace = UserSpaceMAS.from_config(config)
        persistence = None
        if config.kernel.storage_root:
            persistence = PersistentMemoryBackend(config.kernel.storage_root)
        memory_agent = MemoryAgent(
            hidden_size=userspace.hidden_size,
            key_dim=config.kernel.key_dim,
            reward_threshold=config.kernel.reward_threshold,
            latent_window=config.kernel.latent_window,
            persistence=persistence,
            autosave=config.kernel.autosave,
        )
        if persistence is not None and config.kernel.autoload:
            memory_agent.load_from_disk(map_location=userspace.device)
        kernel = OSKernel(memory_agent=memory_agent)
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
            generation_config = GenerationConfig(
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.userspace.tokenizer.pad_token_id,
                eos_token_id=self.userspace.tokenizer.eos_token_id,
            )

        generation = self.userspace.run_role_turn(
            content=userspace_prompt,
            mounted_latents=resolved.mounted_latents,
            generation_config=generation_config,
        )
        mounted_pointer_ids = [hit.pointer for hit in resolved.explicit_hits + resolved.prefetched_hits]
        return PipelineRunResult(
            resolved_message=resolved,
            prompt=userspace_prompt,
            output_text=generation.text,
            mounted_pointer_ids=mounted_pointer_ids,
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
        generation_config: Optional[GenerationConfig] = None,
        top_k_prefetch: Optional[int] = None,
    ) -> MASExecutionTrace:
        return self.userspace.run_multi_agent_task(
            task_description=task_description,
            kernel=self.kernel,
            mas_style=mas_style,
            task_domain=task_domain,
            generation_config=generation_config,
            top_k_prefetch=self.top_k_prefetch if top_k_prefetch is None else top_k_prefetch,
        )
