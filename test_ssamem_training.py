from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from ssamem.config import KernelConfig, PipelineConfig, RuntimeConfig
from ssamem.data_models import AgentMessage
from ssamem.core.kernelspace import IPCBus, MemoryAgent
from ssamem.core.memory_actions import parse_memory_request
from ssamem.core.pipeline import PointerDrivenSSAMemPipeline
from ssamem.retrieval import HashEmbeddingEncoder, RandomHyperplaneLSHIndex, DenseInnerProductRetriever
from ssamem.training.space import (
    SSABatch,
    SSADistiller,
    SSAManifestDataset,
    SSASample,
    ExplicitTeacher,
    LatentStudent,
    PointerPreferenceSample,
    build_ssa_data_from_traces,
    evaluate_pointer_routing,
    extract_probe_keywords,
    generate_synthetic_api_ssa_records,
    initialize_pointer_vocabulary,
    load_ssa_manifest,
    make_multimemory_ssa_manifest,
    pointer_preferences_from_rollouts,
    pointer_tokens,
    ssa_collate,
    split_ssa_manifest,
)
from ssamem.training.data import _record_from_agent_trajectory, _record_from_kodcode, _record_from_popqa


def build_tiny_pipeline() -> PointerDrivenSSAMemPipeline:
    return PointerDrivenSSAMemPipeline.from_config(
        PipelineConfig(
            runtime=RuntimeConfig(
                runtime_mode="tiny-random",
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=4,
                vocab_size=128,
            )
        )
    )


class SSAMemTrainingTests(unittest.TestCase):
    def test_ssa_hidden_alignment_backward_smoke(self) -> None:
        torch.manual_seed(7)
        pipeline = build_tiny_pipeline()
        distiller = SSADistiller(
            student=LatentStudent(pipeline.userspace),
            teacher=ExplicitTeacher(pipeline.userspace),
        )
        batch = SSABatch(
            latent_tensors=[torch.randn(4, pipeline.userspace.hidden_size)],
            explicit_cot_texts=["Task: answer with the tax API rule.\nContext: tax rate is 5%.\nThe answer is:"],
            student_prompts=["Task: answer with the tax API rule.\nThe answer is:"],
        )
        output = distiller.compute_step(batch)
        self.assertEqual(output.student_hidden.shape, output.teacher_hidden.shape)
        output.loss.backward()
        self.assertIsNotNone(distiller.student_projection.weight.grad)
        projector_grads = [
            parameter.grad
            for parameter in distiller.student.memory_projector.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(any(grad is not None for grad in projector_grads))
        composer_grads = [
            parameter.grad
            for parameter in distiller.student.memory_composer.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(any(grad is not None for grad in composer_grads))

    def test_projector_answer_loss_backward_smoke(self) -> None:
        torch.manual_seed(7)
        pipeline = build_tiny_pipeline()
        distiller = SSADistiller(
            student=LatentStudent(pipeline.userspace),
            teacher=ExplicitTeacher(pipeline.userspace),
            answer_loss_weight=0.5,
        )
        batch = SSABatch(
            latent_tensors=[torch.randn(4, pipeline.userspace.hidden_size)],
            explicit_cot_texts=["Task: answer with the tax API rule.\nContext: tax rate is 5%.\nThe answer is:"],
            student_prompts=["Task: answer with the tax API rule.\nThe answer is:"],
            target_texts=[" 5%"],
        )
        output = distiller.compute_step(batch)
        self.assertIsNotNone(output.answer_loss)
        output.loss.backward()
        projector_grads = [
            parameter.grad
            for parameter in distiller.student.memory_projector.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(any(grad is not None for grad in projector_grads))
        composer_grads = [
            parameter.grad
            for parameter in distiller.student.memory_composer.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(any(grad is not None for grad in composer_grads))

    def test_memory_utility_preference_loss_backward_smoke(self) -> None:
        torch.manual_seed(7)
        pipeline = build_tiny_pipeline()
        distiller = SSADistiller(
            student=LatentStudent(pipeline.userspace),
            teacher=ExplicitTeacher(pipeline.userspace),
            answer_loss_weight=0.1,
            preference_loss_weight=0.1,
            preference_beta=0.2,
        )
        batch = SSABatch(
            latent_tensors=[
                torch.randn(4, pipeline.userspace.hidden_size),
                torch.randn(4, pipeline.userspace.hidden_size),
            ],
            explicit_cot_texts=[
                "Task: answer with the tax API rule.\nContext: tax rate is 5%.\nThe answer is:",
                "Task: answer with the payroll API rule.\nContext: payroll rate is 7%.\nThe answer is:",
            ],
            student_prompts=[
                "Task: answer with the tax API rule.\nThe answer is:",
                "Task: answer with the payroll API rule.\nThe answer is:",
            ],
            target_texts=[" 5%", " 7%"],
        )
        output = distiller.compute_step(batch)
        self.assertIsNotNone(output.preference_loss)
        self.assertIsNotNone(output.preference_margin)
        output.loss.backward()
        projector_grads = [
            parameter.grad
            for parameter in distiller.student.memory_projector.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(any(grad is not None for grad in projector_grads))

    def test_pointer_vocabulary_and_preference_validation(self) -> None:
        pipeline = build_tiny_pipeline()
        added = initialize_pointer_vocabulary(pipeline.userspace.tokenizer, pipeline.userspace.model, max_index=3)
        self.assertEqual(added, 3)
        self.assertEqual(pointer_tokens(2), ["<PTR_0x001>", "<PTR_0x002>"])
        encoded = pipeline.userspace.tokenizer("<PTR_0x001>", add_special_tokens=False, padding=False)["input_ids"][0]
        self.assertEqual(len(encoded), 1)
        PointerPreferenceSample(
            prompt="Use the tax spec.",
            chosen="<PTR_0x001>",
            rejected="<PTR_0x002>",
        ).validate()
        with self.assertRaises(ValueError):
            PointerPreferenceSample(prompt="bad", chosen="PTR_1", rejected="<PTR_0x002>").validate()

    def test_memory_action_parser_and_exact_get(self) -> None:
        search = parse_memory_request("SEARCH: tax API rule; top_k=2")
        self.assertEqual(search.mode, "SEARCH")
        self.assertEqual(search.query, "tax API rule")
        self.assertEqual(search.top_k, 2)
        get = parse_memory_request("GET: <PTR_0x001>:0000")
        self.assertEqual(get.mode, "GET")
        self.assertEqual(get.address, "<PTR_0x001>:0000")

        pipeline = build_tiny_pipeline()
        pointer = pipeline.register_memory_tensor(
            torch.randn(4, pipeline.userspace.hidden_size),
            metadata={"topic": "tax API rule"},
        )
        memory = pipeline.get_memory(f"{pointer}:0000")
        self.assertEqual(memory.local_index, 0)
        self.assertEqual(memory.latent_tensor.pointer, pointer)

    def test_build_ssa_manifest_with_latent_shards(self) -> None:
        import tempfile
        from pathlib import Path

        pipeline = build_tiny_pipeline()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "traces.jsonl"
            input_path.write_text(
                json.dumps(
                    {
                        "task_prompt": "Implement tax API.",
                        "context_text": "The tax rate is 5 percent.",
                        "answer_prefix": "The answer is:",
                        "trace_id": "trace-a",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            samples = build_ssa_data_from_traces(
                input_path=input_path,
                output_dir=tmp_path / "ssa",
                userspace=pipeline.userspace,
                latent_max_tokens=16,
            )
            self.assertEqual(len(samples), 1)
            loaded = load_ssa_manifest(tmp_path / "ssa" / "manifest.jsonl")
            self.assertIsInstance(loaded[0], SSASample)
            self.assertTrue((tmp_path / "ssa" / loaded[0].latent_tensor_path).exists())
            self.assertEqual(loaded[0].metadata["source_trace_id"], "trace-a")

    def test_build_ssa_manifest_preserves_target_text(self) -> None:
        pipeline = build_tiny_pipeline()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "traces.jsonl"
            input_path.write_text(
                json.dumps(
                    {
                        "task_prompt": "Who founded ExampleCo?",
                        "context_text": "ExampleCo was founded by Ada Lovelace.",
                        "answer_prefix": "The answer is:",
                        "target_text": " Ada Lovelace",
                        "trace_id": "trace-target",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            build_ssa_data_from_traces(
                input_path=input_path,
                output_dir=tmp_path / "ssa",
                userspace=pipeline.userspace,
                latent_max_tokens=16,
            )
            dataset = SSAManifestDataset(tmp_path / "ssa" / "manifest.jsonl")
            batch = ssa_collate([dataset[0]])
            self.assertEqual(batch.target_texts, [" Ada Lovelace"])

    def test_manifest_dataset_loads_multiple_latent_paths_as_one_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            latent_dir = tmp_path / "latents"
            latent_dir.mkdir()
            torch.save(torch.ones(2, 32), latent_dir / "a.pt")
            torch.save(torch.zeros(3, 32), latent_dir / "b.pt")
            manifest = tmp_path / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "task_prompt": "What is the answer?",
                        "context_text": "Two memories are mounted.",
                        "latent_tensor_paths": ["latents/a.pt", "latents/b.pt"],
                        "target_text": " answer",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            dataset = SSAManifestDataset(manifest)
            batch = ssa_collate([dataset[0]])

            self.assertEqual(len(batch.latent_tensors), 1)
            self.assertIsInstance(batch.latent_tensors[0], list)
            self.assertEqual(len(batch.latent_tensors[0]), 2)
            self.assertEqual(batch.target_texts, [" answer"])

    def test_latent_student_projects_multiple_memories_to_fixed_prefix(self) -> None:
        pipeline = build_tiny_pipeline()
        student = LatentStudent(pipeline.userspace, composer_prefix_length=4)
        memories = [torch.randn(2, pipeline.userspace.hidden_size), torch.randn(3, pipeline.userspace.hidden_size)]

        projected = student.project_latents(["Question?\n\nThe answer is:"], [memories])

        self.assertEqual(len(projected), 1)
        self.assertEqual(tuple(projected[0].shape), (4, pipeline.userspace.hidden_size))

    def test_make_multimemory_manifest_adds_distractors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            latent_dir = tmp_path / "latents"
            latent_dir.mkdir()
            rows = []
            for idx, target in enumerate([" Ada", " Bob", " Cara"]):
                latent_path = latent_dir / f"{idx}.pt"
                torch.save(torch.full((2, 32), float(idx)), latent_path)
                rows.append(
                    {
                        "task_prompt": f"Question {idx}?",
                        "context_text": f"Memory {idx}",
                        "latent_tensor_path": f"latents/{idx}.pt",
                        "target_text": target,
                    }
                )
            manifest = tmp_path / "manifest.jsonl"
            manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            output = tmp_path / "multi" / "manifest.jsonl"

            payload = make_multimemory_ssa_manifest(manifest, output_path=output, distractors=2, seed=3)
            samples = load_ssa_manifest(output)

            self.assertEqual(payload["output_count"], 3)
            self.assertEqual(len(samples), 3)
            self.assertEqual(len(samples[0].latent_tensor_paths), 3)
            self.assertEqual(samples[0].metadata["distractor_count"], 2)
            self.assertIn(samples[0].metadata["positive_memory_index"], [0, 1, 2])

    def test_popqa_and_kodcode_records_include_target_text(self) -> None:
        popqa = _record_from_popqa(
            {
                "question": "Who founded ExampleCo?",
                "answer": ["Ada Lovelace", "A. Lovelace"],
                "context_text": "ExampleCo was founded by Ada Lovelace.",
            },
            0,
        )
        self.assertEqual(popqa.target_text, " Ada Lovelace")
        self.assertEqual(popqa.metadata["source"], "popqa")

        kodcode = _record_from_kodcode(
            {
                "prompt": "Return x plus one.",
                "solution": "def add_one(x):\n    return x + 1",
                "test": "assert add_one(1) == 2",
                "test_info": [{"function_name": "add_one"}],
            },
            1,
        )
        self.assertIn("def add_one", kodcode.target_text)
        self.assertEqual(kodcode.metadata["source"], "kodcode")

    def test_agent_trajectory_records_use_trajectory_as_teacher_memory(self) -> None:
        record = _record_from_agent_trajectory(
            {
                "task_prompt": "Answer the user using prior MAS experience.",
                "agent": "solver",
                "trajectory": [
                    {
                        "agent": "planner",
                        "prompt": "Question: Who directed The Driver?",
                        "output": "We need a film director fact.",
                    },
                    {
                        "agent": "solver",
                        "prompt": "Use the retrieved memory.",
                        "output": "The answer is Walter Hill.",
                    },
                ],
                "target_agent_output": " Walter Hill",
                "reward": 1.0,
            },
            3,
        )
        self.assertIn("[planner]", record.context_text)
        self.assertIn("Output: The answer is Walter Hill.", record.context_text)
        self.assertEqual(record.answer_prefix, "Next agent output:")
        self.assertEqual(record.target_text, "Walter Hill")
        self.assertEqual(record.metadata["agent_role"], "solver")

    def test_pointer_preferences_from_rollout_rewards(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rollouts.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "prompt": "Implement tax API.",
                        "candidates": [
                            {"pointer": "<PTR_0x001>", "reward": 0.0},
                            {"pointer": "<PTR_0x042>", "reward": 1.0},
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            prefs = pointer_preferences_from_rollouts(path)
            self.assertEqual(prefs[0].chosen, "<PTR_0x042>")
            self.assertEqual(prefs[0].rejected, "<PTR_0x001>")

    def test_generate_synthetic_api_ssa_records(self) -> None:
        records = generate_synthetic_api_ssa_records(limit=5, seed=7)
        self.assertEqual(len(records), 5)
        self.assertTrue(records[0].task_prompt)
        self.assertIn("API contract", records[0].context_text)
        self.assertEqual(records[0].metadata["source"], "synthetic_api_seed")

    def test_split_ssa_manifest_rewrites_latent_paths(self) -> None:
        import tempfile
        from pathlib import Path

        pipeline = build_tiny_pipeline()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "traces.jsonl"
            input_path.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "task_prompt": f"Implement tax API {idx}.",
                            "context_text": f"The tax rate for region {idx} is 5 percent.",
                            "answer_prefix": "The answer is:",
                            "trace_id": f"trace-{idx}",
                        }
                    )
                    for idx in range(10)
                )
                + "\n",
                encoding="utf-8",
            )
            build_ssa_data_from_traces(
                input_path=input_path,
                output_dir=tmp_path / "ssa",
                userspace=pipeline.userspace,
                latent_max_tokens=16,
            )
            split_payload = split_ssa_manifest(tmp_path / "ssa" / "manifest.jsonl", output_dir=tmp_path / "suite" / "splits")
            self.assertEqual(split_payload["total"], 10)
            train_manifest = Path(split_payload["splits"]["train"]["manifest"])
            loaded = load_ssa_manifest(train_manifest)
            self.assertTrue(loaded)
            latent_path = (train_manifest.parent / loaded[0].latent_tensor_path).resolve()
            self.assertTrue(latent_path.exists())

    def test_evaluate_pointer_routing_and_probe_keywords(self) -> None:
        prefs = [
            PointerPreferenceSample(prompt="Use the tax spec.", chosen="<PTR_0x001>", rejected="<PTR_0x002>"),
            PointerPreferenceSample(prompt="Use the tax spec.", chosen="<PTR_0x001>", rejected="<PTR_0x003>"),
            PointerPreferenceSample(prompt="Use the shipping spec.", chosen="<PTR_0x004>", rejected="<PTR_0x001>"),
        ]
        routing = evaluate_pointer_routing(prefs, seed=7)
        self.assertEqual(routing["count"], 3)
        self.assertIn("random_accuracy", routing)
        self.assertIn("majority_accuracy", routing)
        keywords = extract_probe_keywords("API contract for /tax/calculate. Tax rate is 5%. Return JSON.")
        self.assertTrue(keywords)

    def test_cluster_backed_registration_and_exact_address_fetch(self) -> None:
        pipeline = build_tiny_pipeline()
        pointer = pipeline.register_memory_tensor(
            torch.randn(4, pipeline.userspace.hidden_size),
            metadata={"topic": "cluster-a"},
        )
        cluster = pipeline.kernel.memory_agent.get_cluster_by_pointer(pointer)
        self.assertEqual(cluster.pointer_id, pointer)
        self.assertEqual(cluster.cluster_size, 1)
        appended = pipeline.kernel.memory_agent.append_memory_to_cluster(
            pointer,
            torch.randn(5, pipeline.userspace.hidden_size),
            metadata={"topic": "cluster-a", "kind": "followup"},
            memory_summary="followup memory",
        )
        self.assertEqual(appended.local_index, 1)
        exact = pipeline.kernel.memory_agent.get_memory_by_address(pointer, 1)
        self.assertEqual(exact.local_index, 1)
        self.assertEqual(exact.memory_summary, "followup memory")
        self.assertEqual(
            pipeline.kernel.memory_agent.get_cluster_by_pointer(pointer).cluster_size,
            2,
        )

    def test_cluster_persistence_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = PointerDrivenSSAMemPipeline.from_config(
                PipelineConfig(
                    runtime=RuntimeConfig(
                        runtime_mode="tiny-random",
                        hidden_size=32,
                        intermediate_size=64,
                        num_hidden_layers=1,
                        num_attention_heads=4,
                        num_key_value_heads=4,
                        vocab_size=128,
                    ),
                    kernel=KernelConfig(storage_root=str(Path(tmpdir) / "store"), autoload=False),
                )
            )
            pointer = pipeline.register_memory_tensor(
                torch.randn(4, pipeline.userspace.hidden_size),
                metadata={"topic": "persistent-cluster"},
            )
            pipeline.kernel.memory_agent.append_memory_to_cluster(
                pointer,
                torch.randn(3, pipeline.userspace.hidden_size),
                metadata={"topic": "persistent-cluster", "kind": "extra"},
            )
            pipeline.save_memory_store()

            restored = PointerDrivenSSAMemPipeline.from_config(
                PipelineConfig(
                    runtime=RuntimeConfig(
                        runtime_mode="tiny-random",
                        hidden_size=32,
                        intermediate_size=64,
                        num_hidden_layers=1,
                        num_attention_heads=4,
                        num_key_value_heads=4,
                        vocab_size=128,
                    ),
                    kernel=KernelConfig(storage_root=str(Path(tmpdir) / "store"), autoload=True),
                )
            )
            cluster = restored.kernel.memory_agent.get_cluster_by_pointer(pointer)
            self.assertEqual(cluster.cluster_size, 2)
            fetched = restored.kernel.memory_agent.get_memory_by_address(pointer, 1)
            self.assertEqual(fetched.local_index, 1)

    def test_two_stage_search_prefers_matching_cluster(self) -> None:
        torch.manual_seed(7)
        hidden_size = 8
        memory_agent = MemoryAgent(hidden_size=hidden_size, key_dim=hidden_size)

        alpha_pointer = memory_agent.add_tensor(
            torch.ones(4, hidden_size),
            metadata={"topic": "alpha"},
        )
        beta_pointer = memory_agent.add_tensor(
            -torch.ones(4, hidden_size),
            metadata={"topic": "beta"},
        )

        alpha_cluster = memory_agent.get_cluster_by_pointer(alpha_pointer)
        beta_cluster = memory_agent.get_cluster_by_pointer(beta_pointer)
        alpha_cluster.summary_key_text = "alpha memory"
        beta_cluster.summary_key_text = "beta memory"

        encoder = HashEmbeddingEncoder(key_dim=hidden_size, vocab_size=256)
        alpha_embedding = encoder("alpha memory").squeeze(0).detach()
        beta_embedding = encoder("beta memory").squeeze(0).detach()
        alpha_cluster.summary_key_embedding = alpha_embedding
        alpha_cluster.centroid_embedding = alpha_embedding
        beta_cluster.summary_key_embedding = beta_embedding
        beta_cluster.centroid_embedding = beta_embedding
        memory_agent.page_table.resolve_entry(alpha_pointer).summary_key_embedding = alpha_embedding.clone()
        memory_agent.page_table.resolve_entry(beta_pointer).summary_key_embedding = beta_embedding.clone()

        retriever = DenseInnerProductRetriever(
            query_encoder=encoder,
            lsh_index=RandomHyperplaneLSHIndex(hidden_size, num_tables=6, num_planes=10, seed=7),
            cluster_rerank_k=2,
        )
        bus = IPCBus(memory_agent=memory_agent, retriever=retriever)
        resolved = bus.intercept(
            AgentMessage(sender_id="a", receiver_id="b", content="please use alpha memory"),
            top_k_prefetch=1,
        )
        self.assertEqual(len(resolved.prefetched_hits), 1)
        self.assertEqual(resolved.prefetched_hits[0].pointer, alpha_pointer)
        self.assertEqual(resolved.prefetched_hits[0].cluster_id, alpha_cluster.cluster_id)

    def test_write_time_lsh_assignment_appends_similar_and_splits_dissimilar(self) -> None:
        hidden_size = 8
        memory_agent = MemoryAgent(
            hidden_size=hidden_size,
            key_dim=hidden_size,
            cluster_assignment_threshold=0.7,
        )

        pointer_a = memory_agent.add_tensor(
            torch.ones(4, hidden_size),
            metadata={"topic": "shared-topic"},
        )
        pointer_b = memory_agent.add_tensor(
            torch.ones(4, hidden_size) * 0.9,
            metadata={"topic": "shared-topic-variant"},
        )
        pointer_c = memory_agent.add_tensor(
            -torch.ones(4, hidden_size),
            metadata={"topic": "opposite-topic"},
        )

        self.assertEqual(pointer_a, pointer_b)
        cluster_a = memory_agent.get_cluster_by_pointer(pointer_a)
        self.assertEqual(cluster_a.cluster_size, 2)
        self.assertNotEqual(pointer_a, pointer_c)
        self.assertEqual(len(memory_agent.memory_clusters), 2)


if __name__ == "__main__":
    unittest.main()
