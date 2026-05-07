from __future__ import annotations

"""Top-level SSAMem pipeline orchestration.

This module is the main entry for understanding how the system is wired:

- userspace: MAS role execution and latent prompt mounting
- kernel: pointer table, memory clusters, SEARCH / GET
- retriever: query-to-memory retrieval backend

Lower-level implementation details still live in their own modules, but the
end-to-end control flow is centralized here.
"""

from typing import Optional

import torch
from transformers import GenerationConfig

from ssamem.config import PipelineConfig
from ssamem.core.kernelspace import MemoryAgent, OSKernel
from ssamem.core.userspace import UserSpaceMAS, build_deterministic_generation_config
from ssamem.data_models import AgentMessage, ClusterMemory, MASExecutionTrace, PipelineRunResult, PointerSearchHit
from ssamem.retrieval import DenseInnerProductRetriever, RandomHyperplaneLSHIndex, build_retriever
from ssamem.storage import PersistentMemoryBackend


class PointerDrivenSSAMemPipeline:
    """High-level orchestrator for the SSAMem runtime.

    The pipeline intentionally keeps the system-level story in one place:

    1. build userspace and kernel components
    2. resolve pointer-based memory access
    3. mount retrieved latent memories into MAS role prompts
    4. run either direct role turns or full MAS loops
    """

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
    def _build_memory_agent_from_config(
        cls,
        *,
        config: PipelineConfig,
        userspace: UserSpaceMAS,
        persistence: Optional[PersistentMemoryBackend],
    ) -> MemoryAgent:
        key_dim = config.kernel.key_dim or userspace.hidden_size
        return MemoryAgent(
            hidden_size=userspace.hidden_size,
            key_dim=key_dim,
            reward_threshold=config.kernel.reward_threshold,
            latent_window=config.kernel.latent_window,
            persistence=persistence,
            autosave=config.kernel.autosave,
            cluster_assignment_threshold=config.kernel.cluster_assignment_threshold,
            cluster_assignment_top_k=config.kernel.cluster_assignment_top_k,
        )

    @classmethod
    def _build_kernel_from_config(
        cls,
        *,
        config: PipelineConfig,
        memory_agent: MemoryAgent,
    ) -> OSKernel:
        retriever = build_retriever(
            config.kernel.retriever_type,
            key_dim=memory_agent.key_dim,
            hash_vocab_size=config.kernel.hash_vocab_size,
        )
        return OSKernel(memory_agent=memory_agent, retriever=retriever)

    @classmethod
    def from_config(cls, config: PipelineConfig) -> "PointerDrivenSSAMemPipeline":
        userspace = UserSpaceMAS.from_config(config)
        persistence = None
        if config.kernel.storage_root:
            persistence = PersistentMemoryBackend(config.kernel.storage_root)
        memory_agent = cls._build_memory_agent_from_config(
            config=config,
            userspace=userspace,
            persistence=persistence,
        )
        if persistence is not None and config.kernel.autoload:
            memory_agent.load_from_disk(map_location=userspace.device)
        kernel = cls._build_kernel_from_config(config=config, memory_agent=memory_agent)
        return cls(
            userspace=userspace,
            kernel=kernel,
            top_k_prefetch=config.kernel.top_k_prefetch,
            max_new_tokens=config.max_new_tokens,
            strip_pointer_tokens=config.strip_pointer_tokens,
        )

    def attach_trained_retriever(self, retrieval_model) -> None:
        """Swap in a trained query encoder + retrieval key space.

        This is used when the pipeline should SEARCH with the learned
        query-to-latent retriever instead of the default hash retriever.
        """
        key_dim = retrieval_model.config.key_dim
        self.kernel.memory_agent.key_dim = key_dim
        self.kernel.memory_agent.cluster_assignment_lsh = RandomHyperplaneLSHIndex(key_dim)
        self.kernel.retriever = DenseInnerProductRetriever(
            query_encoder=retrieval_model.query_encoder,
            lsh_index=RandomHyperplaneLSHIndex(key_dim),
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

    def _resolve_top_k_prefetch(self, top_k_prefetch: Optional[int]) -> int:
        return self.top_k_prefetch if top_k_prefetch is None else top_k_prefetch

    def _resolve_generation_config(
        self,
        generation_config: Optional[GenerationConfig],
    ) -> GenerationConfig:
        if generation_config is not None:
            return generation_config
        return build_deterministic_generation_config(
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.userspace.tokenizer.pad_token_id,
            eos_token_id=self.userspace.tokenizer.eos_token_id,
        )

    def resolve_message(
        self,
        *,
        sender_id: str,
        receiver_id: str,
        content: str,
        top_k_prefetch: Optional[int] = None,
    ):
        ipc_message = AgentMessage(sender_id=sender_id, receiver_id=receiver_id, content=content)
        resolved = self.kernel.handle_ipc(
            ipc_message,
            top_k_prefetch=self._resolve_top_k_prefetch(top_k_prefetch),
        )
        return ipc_message, resolved

    def run_message(
        self,
        *,
        sender_id: str,
        receiver_id: str,
        content: str,
        generation_config: Optional[GenerationConfig] = None,
        top_k_prefetch: Optional[int] = None,
    ) -> PipelineRunResult:
        ipc_message, resolved = self.resolve_message(
            sender_id=sender_id,
            receiver_id=receiver_id,
            content=content,
            top_k_prefetch=top_k_prefetch,
        )
        userspace_prompt = self.build_prompt_for_userspace(content)
        generation_config = self._resolve_generation_config(generation_config)

        generation = self.userspace.run_role_turn(
            content=userspace_prompt,
            mounted_latents=resolved.mounted_latents,
            generation_config=generation_config,
        )
        explicit_pointer_ids = [hit.exact_address for hit in resolved.explicit_hits]
        prefetched_pointer_ids = [hit.exact_address for hit in resolved.prefetched_hits]
        prefetched_scores = [hit.score for hit in resolved.prefetched_hits]
        mounted_pointer_ids = [hit.exact_address for hit in resolved.explicit_hits + resolved.prefetched_hits]
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

    def pointer_table_context(self, *, max_entries: Optional[int] = None) -> str:
        return self.kernel.pointer_table_context(max_entries=max_entries)

    def search_memory(self, intent_text: str, *, top_k: int = 1) -> list[PointerSearchHit]:
        return self.kernel.search_memory(intent_text, top_k=top_k)

    def get_memory(self, address: str) -> ClusterMemory:
        return self.kernel.get_memory(address)

    def build_memory_protocol(
        self,
        *,
        mode: str = "standard",
    ) -> str:
        if mode == "search-only":
            return (
                "The MAS agent must explicitly request memory before answering. "
                "Only SEARCH is allowed in this evaluation; do not use GET."
            )
        return (
            "The MAS agent must explicitly request memory before answering. "
            "Use SEARCH when it only knows a summary key; use GET for exact addresses."
        )

    def build_memory_context(
        self,
        *,
        memory_content: str,
        include_pointer_table: bool = False,
        pointer_table_max_entries: Optional[int] = None,
        protocol_mode: str = "standard",
    ) -> str:
        if not include_pointer_table:
            return memory_content
        return (
            f"{self.pointer_table_context(max_entries=pointer_table_max_entries)}\n\n"
            "[Memory Agent Protocol]\n"
            f"{self.build_memory_protocol(mode=protocol_mode)}\n\n"
            f"{memory_content}"
        ).strip()

    def run_multi_agent_task(
        self,
        task_description: str,
        *,
        mas_style: str = "camel",
        task_domain: Optional[str] = None,
        memory_content: str = "[mounted-latent-memory]",
        include_pointer_table: bool = False,
        pointer_table_max_entries: Optional[int] = None,
        generation_config: Optional[GenerationConfig] = None,
        top_k_prefetch: Optional[int] = None,
    ) -> MASExecutionTrace:
        memory_content = self.build_memory_context(
            memory_content=memory_content,
            include_pointer_table=include_pointer_table,
            pointer_table_max_entries=pointer_table_max_entries,
            protocol_mode="standard",
        )
        return self.userspace.run_multi_agent_task(
            task_description=task_description,
            kernel=self.kernel,
            mas_style=mas_style,
            task_domain=task_domain,
            memory_content=memory_content,
            generation_config=generation_config,
            top_k_prefetch=self._resolve_top_k_prefetch(top_k_prefetch),
        )

    def run_multi_agent_task_with_memory_actions(
        self,
        task_description: str,
        *,
        mas_style: str = "camel",
        task_domain: Optional[str] = None,
        memory_content: str = "[mounted-latent-memory]",
        pointer_table_max_entries: Optional[int] = None,
        generation_config: Optional[GenerationConfig] = None,
        request_generation_config: Optional[GenerationConfig] = None,
        default_top_k: int = 1,
        memory_request_policy: str = "auto",
    ) -> MASExecutionTrace:
        pointer_table_context = (
            f"{self.pointer_table_context(max_entries=pointer_table_max_entries)}\n\n"
            "[Memory Agent Protocol]\n"
            f"{self.build_memory_protocol(mode='search-only' if memory_request_policy in {'require-search', 'search-only'} else 'standard')}"
        )
        return self.userspace.run_multi_agent_task_with_memory_actions(
            task_description=task_description,
            kernel=self.kernel,
            mas_style=mas_style,
            task_domain=task_domain,
            memory_content=memory_content,
            pointer_table_context=pointer_table_context,
            generation_config=generation_config,
            request_generation_config=request_generation_config,
            default_top_k=default_top_k,
            memory_request_policy=memory_request_policy,
        )
