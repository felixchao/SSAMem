from __future__ import annotations

from ssamem.config import KernelConfig, MASConfig, PipelineConfig, RunConfig, RuntimeConfig


def build_pipeline_from_args(args) -> PointerDrivenSSAMemPipeline:
    from ssamem.core.pipeline import PointerDrivenSSAMemPipeline

    config = PipelineConfig(
        runtime=RuntimeConfig(
            runtime_mode=args.runtime_mode,
            model_name_or_path=args.model_name_or_path,
            trust_remote_code=getattr(args, "trust_remote_code", False),
            torch_dtype=getattr(args, "torch_dtype", None) or ("float16" if args.runtime_mode == "hf" else "float32"),
            device=args.device,
            load_in_4bit=getattr(args, "load_in_4bit", False),
        ),
        kernel=KernelConfig(
            top_k_prefetch=getattr(args, "top_k_prefetch", 1),
            storage_root=getattr(args, "storage_root", None),
            cluster_assignment_threshold=getattr(args, "cluster_assignment_threshold", 0.95),
            cluster_assignment_top_k=getattr(args, "cluster_assignment_top_k", 4),
        ),
        mas=MASConfig(
            architecture=getattr(args, "mas_style", "camel"),
            task_domain=getattr(args, "task_domain", None),
        ),
        run=RunConfig(
            seed=getattr(args, "seed", 7),
            run_name=getattr(args, "run_name", None),
            output_dir=getattr(args, "output_dir", None),
            log_level=getattr(args, "log_level", "INFO"),
        ),
        max_new_tokens=getattr(args, "max_new_tokens", 32),
    )
    return PointerDrivenSSAMemPipeline.from_config(config)


def build_synthetic_ssa_batches(hidden_size: int, steps: int) -> list[SSABatch]:
    import torch
    from ssamem.training.space import SSABatch

    batches = []
    for step in range(steps):
        latent_tensor = torch.randn(8, hidden_size)
        explicit_text = (
            f"Thought {step}: read the evidence, extract the answer span, "
            f"then provide the final short answer."
        )
        batches.append(
            SSABatch(
                latent_tensors=[latent_tensor],
                explicit_cot_texts=[explicit_text],
                student_prompts=["Decode the mounted latent memory into a compact reasoning state."],
            )
        )
    return batches


def build_ssa_distiller_for_eval(args):
    from ssamem.training.space import SSADistiller, ExplicitTeacher, LatentStudent

    pipeline = build_pipeline_from_args(args)
    alignment_cfg = getattr(args, "alignment", {})
    distiller = SSADistiller(
        student=LatentStudent(pipeline.userspace),
        teacher=ExplicitTeacher(pipeline.userspace),
        loss_type=alignment_cfg.get("loss_type", "smooth_l1"),
        distill_loss_div_std=bool(alignment_cfg.get("distill_loss_div_std", True)),
        answer_loss_weight=float(alignment_cfg.get("answer_loss_weight", 0.0)),
        preference_loss_weight=float(alignment_cfg.get("preference_loss_weight", 0.0)),
        preference_beta=float(alignment_cfg.get("preference_beta", 0.1)),
        utility_loss_weight=float(alignment_cfg.get("utility_loss_weight", 0.0)),
        utility_margin=float(alignment_cfg.get("utility_margin", 0.2)),
    )
    distiller.to(args.device)
    return pipeline, distiller
