from __future__ import annotations

from dataclasses import dataclass, field
from string import Formatter

from ssamem.mas_prompts import get_domain_prompts


DEFAULT_MEMORY_CONTENT = "[mounted-latent-memory]"


@dataclass(frozen=True)
class MASPromptTemplate:
    system_template: str
    user_template: str
    output_schema: str = ""

    @property
    def required_fields(self) -> tuple[str, ...]:
        formatter = Formatter()
        fields = []
        for _, field_name, _, _ in formatter.parse(self.user_template):
            if field_name:
                fields.append(field_name)
        return tuple(dict.fromkeys(fields))


@dataclass
class MASRoleSpec:
    role: str
    prompt: MASPromptTemplate
    upstream_roles: list[str] = field(default_factory=list)
    field_bindings: dict[str, str] = field(default_factory=dict)
    memory_policy: str = "mounted"

    def _resolve_binding(
        self,
        binding: str,
        *,
        task_description: str,
        role_outputs: dict[str, str],
        feedback_template: str | None,
        memory_content: str,
    ) -> str:
        if binding == "task_description":
            return task_description
        if binding == "memory_content":
            return memory_content
        if binding.startswith("role:"):
            role_name = binding.split(":", 1)[1]
            return role_outputs.get(role_name, "")
        if binding.startswith("feedback:"):
            if not feedback_template:
                raise ValueError(f"Role '{self.role}' requested feedback pages without a template.")
            payload = binding.split(":", 1)[1]
            actor_role, critic_role = payload.split("|", 1)
            return feedback_template.format(
                actor_output=role_outputs.get(actor_role, ""),
                critic_output=role_outputs.get(critic_role, ""),
            ).strip()
        raise ValueError(f"Unsupported MAS field binding: {binding}")

    def render_prompt(
        self,
        task_description: str,
        role_outputs: dict[str, str],
        *,
        feedback_template: str | None = None,
        memory_content: str = DEFAULT_MEMORY_CONTENT,
    ) -> str:
        prompt_fields: dict[str, str] = {}
        for field_name in self.prompt.required_fields:
            binding = self.field_bindings.get(field_name, field_name)
            prompt_fields[field_name] = self._resolve_binding(
                binding,
                task_description=task_description,
                role_outputs=role_outputs,
                feedback_template=feedback_template,
                memory_content=memory_content,
            )

        return (
            f"[Role]\n{self.role}\n\n"
            f"[System Instruction]\n{self.prompt.system_template.strip()}\n\n"
            f"{self.prompt.user_template.format(**prompt_fields).strip()}"
        ).strip()


@dataclass(frozen=True)
class MASTopologySpec:
    architecture: str
    roles: tuple[MASRoleSpec, ...]
    final_role: str
    task_domain: str | None = None
    feedback_template: str | None = None


def _template_from_bundle(bundle: dict[str, str], prefix: str) -> MASPromptTemplate:
    return MASPromptTemplate(
        system_template=bundle[f"{prefix}_system"].strip(),
        user_template=bundle[f"{prefix}_user"].strip(),
    )


def _generic_templates(style: str) -> tuple[dict[str, MASPromptTemplate], str | None]:
    if style == "autogen":
        return (
            {
                "assistant": MASPromptTemplate(
                    system_template="Produce the best direct answer for the task.",
                    user_template=(
                        "[Task]\n{task_description}\n\n"
                        "[Mounted Memory]\n{memory_content}\n\n"
                        "[Output Requirement]\nProvide the current best answer."
                    ),
                ),
                "user_proxy": MASPromptTemplate(
                    system_template="Review the assistant answer from a user perspective and refine it.",
                    user_template=(
                        "[Task]\n{task_description}\n\n"
                        "[Mounted Memory]\n{memory_content}\n\n"
                        "[Assistant Output]\n{assistant_output}\n\n"
                        "[Output Requirement]\nReturn a refined final answer."
                    ),
                ),
            },
            None,
        )

    if style == "camel":
        return (
            {
                "user_proxy": MASPromptTemplate(
                    system_template="Rewrite the task into a clear operating brief for downstream agents.",
                    user_template=(
                        "[Task]\n{task_description}\n\n"
                        "[Mounted Memory]\n{memory_content}\n\n"
                        "[Output Requirement]\nProvide a concise task brief."
                    ),
                ),
                "actor": MASPromptTemplate(
                    system_template="Solve the task directly and use mounted latent memory when useful.",
                    user_template=(
                        "[Task]\n{task_description}\n\n"
                        "[Mounted Memory]\n{memory_content}\n\n"
                        "[User Proxy Answer]\n{userproxy_output}\n\n"
                        "[Output Requirement]\nDraft the best candidate answer."
                    ),
                ),
                "critic": MASPromptTemplate(
                    system_template="Critique the current draft and identify factual or reasoning gaps.",
                    user_template=(
                        "[Task]\n{task_description}\n\n"
                        "[Mounted Memory]\n{memory_content}\n\n"
                        "[Actor Output]\n{actor_output}\n\n"
                        "[Output Requirement]\nProvide specific critique and missing points."
                    ),
                ),
                "summarizer": MASPromptTemplate(
                    system_template="Synthesize the draft and critique into the final answer.",
                    user_template=(
                        "[Task]\n{task_description}\n\n"
                        "[Mounted Memory]\n{memory_content}\n\n"
                        "[Actor Output]\n{actor_output}\n\n"
                        "[Critic Feedback]\n{critic_output}\n\n"
                        "[Output Requirement]\nReturn the final answer only."
                    ),
                ),
            },
            None,
        )

    if style in {"debate", "macnet"}:
        return (
            {
                "actor": MASPromptTemplate(
                    system_template="Produce one plausible solution path.",
                    user_template=(
                        "[Task]\n{task_description}\n\n"
                        "[Mounted Memory]\n{memory_content}\n\n"
                        "[Output Requirement]\nReturn your proposal."
                    ),
                ),
                "critic": MASPromptTemplate(
                    system_template="Critique the current actor proposal.",
                    user_template=(
                        "[Task]\n{task_description}\n\n"
                        "[Mounted Memory]\n{memory_content}\n\n"
                        "[Actor Output]\n{actor_output}\n\n"
                        "[Output Requirement]\nReturn your critique."
                    ),
                ),
                "summarizer": MASPromptTemplate(
                    system_template="Fuse both proposals and critiques into a single final answer.",
                    user_template=(
                        "[Task]\n{task_description}\n\n"
                        "[Mounted Memory]\n{memory_content}\n\n"
                        "[Feedback Page 1]\n{feedback_page1}\n\n"
                        "[Feedback Page 2]\n{feedback_page2}\n\n"
                        "[Output Requirement]\nReturn the final merged answer."
                    ),
                ),
            },
            "## Actor Output\n{actor_output}\n\n## Critic Feedback\n{critic_output}",
        )

    raise ValueError(f"Unsupported MAS style: {style}")


def _domain_templates(style: str, task_domain: str) -> tuple[dict[str, MASPromptTemplate], str | None]:
    prompt_library = get_domain_prompts(task_domain)
    style_key = "macnet" if style == "debate" else style
    if style_key not in prompt_library:
        raise ValueError(f"Task domain '{task_domain}' does not support MAS style '{style}'.")
    bundle = prompt_library[style_key]
    templates: dict[str, MASPromptTemplate] = {}
    if style_key == "autogen":
        templates["assistant"] = _template_from_bundle(bundle, "assistant")
        templates["user_proxy"] = _template_from_bundle(bundle, "user_proxy")
    elif style_key == "camel":
        templates["user_proxy"] = _template_from_bundle(bundle, "user_proxy")
        templates["actor"] = _template_from_bundle(bundle, "actor")
        templates["critic"] = _template_from_bundle(bundle, "critic")
        templates["summarizer"] = _template_from_bundle(bundle, "summarizer")
    else:
        templates["actor"] = _template_from_bundle(bundle, "actor")
        templates["critic"] = _template_from_bundle(bundle, "critic")
        templates["summarizer"] = _template_from_bundle(bundle, "summarizer")
    return templates, bundle.get("feedback_page")


def _resolve_templates(style: str, task_domain: str | None) -> tuple[dict[str, MASPromptTemplate], str | None]:
    if task_domain:
        return _domain_templates(style, task_domain)
    return _generic_templates(style)


def build_mas_topology(style: str, task_domain: str | None = None) -> MASTopologySpec:
    normalized = style.lower()
    templates, feedback_template = _resolve_templates(normalized, task_domain)

    if normalized == "camel":
        roles = (
            MASRoleSpec(
                role="user_proxy_agent",
                prompt=templates["user_proxy"],
                field_bindings={"task_description": "task_description", "memory_content": "memory_content"},
            ),
            MASRoleSpec(
                role="actor_agent",
                prompt=templates["actor"],
                upstream_roles=["user_proxy_agent"],
                field_bindings={
                    "task_description": "task_description",
                    "memory_content": "memory_content",
                    "userproxy_output": "role:user_proxy_agent",
                },
            ),
            MASRoleSpec(
                role="critic_agent",
                prompt=templates["critic"],
                upstream_roles=["actor_agent"],
                field_bindings={
                    "task_description": "task_description",
                    "memory_content": "memory_content",
                    "actor_output": "role:actor_agent",
                },
            ),
            MASRoleSpec(
                role="summarizer_agent",
                prompt=templates["summarizer"],
                upstream_roles=["actor_agent", "critic_agent"],
                field_bindings={
                    "task_description": "task_description",
                    "memory_content": "memory_content",
                    "actor_output": "role:actor_agent",
                    "critic_output": "role:critic_agent",
                },
            ),
        )
        return MASTopologySpec(
            architecture=normalized,
            roles=roles,
            final_role="summarizer_agent",
            task_domain=task_domain,
            feedback_template=feedback_template,
        )

    if normalized == "autogen":
        roles = (
            MASRoleSpec(
                role="assistant_agent",
                prompt=templates["assistant"],
                field_bindings={"task_description": "task_description", "memory_content": "memory_content"},
            ),
            MASRoleSpec(
                role="user_proxy_agent",
                prompt=templates["user_proxy"],
                upstream_roles=["assistant_agent"],
                field_bindings={
                    "task_description": "task_description",
                    "memory_content": "memory_content",
                    "assistant_output": "role:assistant_agent",
                },
            ),
        )
        return MASTopologySpec(
            architecture=normalized,
            roles=roles,
            final_role="user_proxy_agent",
            task_domain=task_domain,
            feedback_template=feedback_template,
        )

    if normalized in {"debate", "macnet"}:
        roles = (
            MASRoleSpec(
                role="actor_agent_1",
                prompt=templates["actor"],
                field_bindings={"task_description": "task_description", "memory_content": "memory_content"},
            ),
            MASRoleSpec(
                role="critic_agent_1",
                prompt=templates["critic"],
                upstream_roles=["actor_agent_1"],
                field_bindings={
                    "task_description": "task_description",
                    "memory_content": "memory_content",
                    "actor_output": "role:actor_agent_1",
                },
            ),
            MASRoleSpec(
                role="actor_agent_2",
                prompt=templates["actor"],
                field_bindings={"task_description": "task_description", "memory_content": "memory_content"},
            ),
            MASRoleSpec(
                role="critic_agent_2",
                prompt=templates["critic"],
                upstream_roles=["actor_agent_2"],
                field_bindings={
                    "task_description": "task_description",
                    "memory_content": "memory_content",
                    "actor_output": "role:actor_agent_2",
                },
            ),
            MASRoleSpec(
                role="summarizer_agent",
                prompt=templates["summarizer"],
                upstream_roles=["critic_agent_1", "critic_agent_2"],
                field_bindings={
                    "task_description": "task_description",
                    "memory_content": "memory_content",
                    "feedback_page1": "feedback:actor_agent_1|critic_agent_1",
                    "feedback_page2": "feedback:actor_agent_2|critic_agent_2",
                },
            ),
        )
        return MASTopologySpec(
            architecture=normalized,
            roles=roles,
            final_role="summarizer_agent",
            task_domain=task_domain,
            feedback_template=feedback_template,
        )

    raise ValueError(f"Unsupported MAS style: {style}")


def build_mas_role_specs(style: str, task_domain: str | None = None) -> list[MASRoleSpec]:
    return list(build_mas_topology(style, task_domain=task_domain).roles)
