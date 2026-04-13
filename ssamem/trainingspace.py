from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import torch
from torch import nn

from ssamem.userspace import UserSpaceMAS


def masked_mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    masked_hidden = hidden_states * mask
    denom = mask.sum(dim=1).clamp(min=1.0)
    return masked_hidden.sum(dim=1) / denom


@dataclass
class CODiBatch:
    latent_tensors: Sequence[torch.Tensor]
    explicit_cot_texts: Sequence[str]
    student_prompts: Optional[Sequence[str]] = None


@dataclass
class DistillationStepOutput:
    loss: torch.Tensor
    student_hidden: torch.Tensor
    teacher_hidden: torch.Tensor


class ExplicitTeacher(nn.Module):
    def __init__(self, userspace: UserSpaceMAS) -> None:
        super().__init__()
        self.userspace = userspace

    @property
    def hidden_size(self) -> int:
        return self.userspace.hidden_size

    def forward(self, texts: Sequence[str]) -> torch.Tensor:
        tokenized = self.userspace.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokenized["input_ids"].to(self.userspace.device)
        attention_mask = tokenized["attention_mask"].to(self.userspace.device)
        outputs = self.userspace.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        return masked_mean_pool(outputs.hidden_states[-1], attention_mask)


class LatentStudent(nn.Module):
    def __init__(self, userspace: UserSpaceMAS) -> None:
        super().__init__()
        self.userspace = userspace

    @property
    def hidden_size(self) -> int:
        return self.userspace.hidden_size

    def forward(self, prompts: Sequence[str], latent_tensors: Sequence[torch.Tensor]) -> torch.Tensor:
        latent_batches = [[latent] for latent in latent_tensors]
        batch = self.userspace.prepare_soft_prompt_batch(prompts=prompts, mounted_latents=latent_batches)
        outputs = self.userspace.model(
            inputs_embeds=batch.inputs_embeds,
            attention_mask=batch.attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        return masked_mean_pool(outputs.hidden_states[-1], batch.attention_mask)


class CODiDistiller(nn.Module):
    def __init__(
        self,
        student: LatentStudent,
        teacher: ExplicitTeacher,
        *,
        temperature: float = 1.0,
        train_teacher: bool = False,
    ) -> None:
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.temperature = temperature
        self.student_head = nn.Linear(student.hidden_size, teacher.hidden_size)

        if not train_teacher:
            for param in self.teacher.parameters():
                param.requires_grad = False
            self.teacher.eval()

    def soft_cross_entropy(self, student_logits: torch.Tensor, teacher_targets: torch.Tensor) -> torch.Tensor:
        teacher_probs = torch.softmax(teacher_targets / self.temperature, dim=-1)
        student_log_probs = torch.log_softmax(student_logits / self.temperature, dim=-1)
        loss = -(teacher_probs * student_log_probs).sum(dim=-1).mean()
        return loss * (self.temperature ** 2)

    def compute_step(self, batch: CODiBatch) -> DistillationStepOutput:
        if len(batch.latent_tensors) != len(batch.explicit_cot_texts):
            raise ValueError("Latent tensor batch size must match teacher text batch size.")

        prompts = batch.student_prompts or ["Summarize the mounted latent memory."] * len(batch.latent_tensors)
        student_hidden = self.student(prompts=prompts, latent_tensors=batch.latent_tensors)
        with torch.set_grad_enabled(any(param.requires_grad for param in self.teacher.parameters())):
            teacher_hidden = self.teacher(batch.explicit_cot_texts)

        student_logits = self.student_head(student_hidden)
        loss = self.soft_cross_entropy(student_logits, teacher_hidden.detach())
        return DistillationStepOutput(
            loss=loss,
            student_hidden=student_hidden,
            teacher_hidden=teacher_hidden.detach(),
        )

    def fit(
        self,
        dataloader: Iterable[CODiBatch],
        optimizer: torch.optim.Optimizer,
        *,
        epochs: int = 1,
        grad_clip_norm: Optional[float] = None,
        device: Optional[torch.device | str] = None,
    ) -> list[float]:
        if device is not None:
            self.to(device)

        history = []
        self.train()
        self.teacher.eval()

        for _ in range(epochs):
            for batch in dataloader:
                optimizer.zero_grad(set_to_none=True)
                step_output = self.compute_step(batch)
                step_output.loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip_norm)
                optimizer.step()
                history.append(float(step_output.loss.detach().cpu().item()))

        return history
