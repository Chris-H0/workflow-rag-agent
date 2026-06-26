from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from placement_compiler.core.artifacts import build_artifact, write_artifact
from placement_compiler.generation.candidates import CandidateGenerationError, CandidateGenerator
from placement_compiler.cli import pipeline_from_config
from placement_compiler.core.metadata import build_workflow_metadata
from placement_compiler.core.models import (
    Candidate,
    CandidateSetDraft,
    ModelEndpoint,
    NodeRegistryMetadata,
    WorkflowEdge,
)
from placement_compiler.pipeline.config import load_pipeline_config
from placement_compiler.adapters.repository_workflows import load_existing_workflow_metadata


class FakeCompilerLLM:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.messages = []

    def generate(self, messages, output_schema):
        self.messages.append(messages)
        if not self.responses:
            raise AssertionError("fake compiler LLM has no response left")
        return self.responses.pop(0)


def sample_workflow():
    return build_workflow_metadata(
        workflow_id="sample",
        node_ids=["__start__", "plan", "act", "__end__"],
        edges=[
            WorkflowEdge(source="__start__", target="plan"),
            WorkflowEdge(source="plan", target="act"),
            WorkflowEdge(source="act", target="__end__"),
        ],
        registry={
            "plan": NodeRegistryMetadata(
                is_llm_placement_unit=True,
                structured_output_required=True,
            ),
            "act": NodeRegistryMetadata(
                is_llm_placement_unit=True,
                tool_use=True,
            ),
        },
    )


def sample_endpoints():
    return [
        ModelEndpoint(
            id="local",
            model="qwen3.5:2b",
            location="local",
            provider="ollama",
            tool_calling=True,
            structured_output=False,
        ),
        ModelEndpoint(
            id="cloud",
            model="gpt-4.1-mini",
            location="cloud",
            provider="openai",
            tool_calling=True,
            structured_output=True,
        ),
    ]


class PlacementCompilerTests(unittest.TestCase):
    def test_existing_qa_workflow_metadata_extraction(self):
        metadata = load_existing_workflow_metadata("qa")

        self.assertEqual(metadata.workflow_id, "qa-workflow")
        self.assertIn("generate_query_or_respond", metadata.entry_nodes)
        self.assertIn("generate_answer", metadata.terminal_nodes)
        self.assertIn("generate_query_or_respond", metadata.terminal_nodes)
        self.assertTrue(metadata.conditional_edges)

        nodes = {node.id: node for node in metadata.nodes}
        self.assertTrue(nodes["generate_query_or_respond"].is_llm_placement_unit)
        self.assertTrue(nodes["generate_query_or_respond"].tool_use)
        self.assertTrue(nodes["decide_after_retrieval"].structured_output_required)
        self.assertFalse(nodes["generate_followup_query"].is_llm_placement_unit)
        self.assertTrue(nodes["retrieve"].in_cycle)

    def test_metadata_registry_merging(self):
        metadata = build_workflow_metadata(
            workflow_id="tiny",
            node_ids=["__start__", "a", "b", "__end__"],
            edges=[
                WorkflowEdge(source="__start__", target="a"),
                WorkflowEdge(source="a", target="b", conditional=True, label="ok"),
                WorkflowEdge(source="b", target="__end__"),
            ],
            registry={
                "a": {
                    "is_llm_placement_unit": True,
                    "semantic_role": "router",
                    "branch_control": True,
                }
            },
        )

        nodes = {node.id: node for node in metadata.nodes}
        self.assertTrue(nodes["a"].is_llm_placement_unit)
        self.assertEqual(nodes["a"].semantic_role, "router")
        self.assertEqual(nodes["a"].stage, "entry")
        self.assertEqual(nodes["b"].stage, "terminal")
        self.assertEqual(metadata.conditional_edges[0].label, "ok")

    def test_generation_of_exactly_n_complete_candidates(self):
        response = CandidateSetDraft(
            candidates=[
                Candidate(
                    id="quality",
                    description="Cloud for every placement unit.",
                    assignments={"plan": "cloud", "act": "cloud"},
                ),
                Candidate(
                    id="balanced",
                    description="Cloud for structured planning, local for action.",
                    assignments={"plan": "cloud", "act": "local"},
                ),
            ]
        )
        llm = FakeCompilerLLM(response)
        candidates = CandidateGenerator(llm, max_attempts=1).generate(
            workflow=sample_workflow(),
            model_endpoints=sample_endpoints(),
            candidate_count=2,
            priorities=["quality", "cloud-cost"],
        )

        self.assertEqual(len(candidates), 2)
        self.assertEqual(set(candidates[0].assignments), {"plan", "act"})
        self.assertIn("candidate_count", llm.messages[0][1]["content"])

    def test_invalid_model_and_node_rejection(self):
        response = CandidateSetDraft(
            candidates=[
                Candidate(
                    id="bad",
                    description="Invalid candidate.",
                    assignments={"plan": "missing-endpoint", "extra": "cloud"},
                )
            ]
        )
        with self.assertRaisesRegex(CandidateGenerationError, "unknown"):
            CandidateGenerator(FakeCompilerLLM(response), max_attempts=1).generate(
                workflow=sample_workflow(),
                model_endpoints=sample_endpoints(),
                candidate_count=1,
            )

    def test_duplicate_candidate_detection(self):
        response = CandidateSetDraft(
            candidates=[
                Candidate(
                    id="one",
                    description="First.",
                    assignments={"plan": "cloud", "act": "local"},
                ),
                Candidate(
                    id="two",
                    description="Duplicate assignments.",
                    assignments={"plan": "cloud", "act": "local"},
                ),
            ]
        )
        with self.assertRaisesRegex(CandidateGenerationError, "duplicates"):
            CandidateGenerator(FakeCompilerLLM(response), max_attempts=1).generate(
                workflow=sample_workflow(),
                model_endpoints=sample_endpoints(),
                candidate_count=2,
            )

    def test_capability_incompatibility_rejection(self):
        response = CandidateSetDraft(
            candidates=[
                Candidate(
                    id="bad-capabilities",
                    description="Local endpoint cannot handle structured planning.",
                    assignments={"plan": "local", "act": "local"},
                )
            ]
        )
        with self.assertRaisesRegex(CandidateGenerationError, "structured output"):
            CandidateGenerator(FakeCompilerLLM(response), max_attempts=1).generate(
                workflow=sample_workflow(),
                model_endpoints=sample_endpoints(),
                candidate_count=1,
            )

    def test_context_window_incompatibility_rejection(self):
        workflow = build_workflow_metadata(
            workflow_id="context",
            node_ids=["__start__", "plan", "__end__"],
            edges=[
                WorkflowEdge(source="__start__", target="plan"),
                WorkflowEdge(source="plan", target="__end__"),
            ],
            registry={
                "plan": NodeRegistryMetadata(
                    is_llm_placement_unit=True,
                    required_context_window=1000,
                )
            },
        )
        response = CandidateSetDraft(
            candidates=[
                Candidate(
                    id="too-small",
                    description="Endpoint context window is too small.",
                    assignments={"plan": "small"},
                )
            ]
        )
        endpoints = [
            ModelEndpoint(
                id="small",
                model="small",
                location="local",
                provider="mock",
                context_window=10,
            )
        ]

        with self.assertRaisesRegex(CandidateGenerationError, "context window"):
            CandidateGenerator(FakeCompilerLLM(response), max_attempts=1).generate(
                workflow=workflow,
                model_endpoints=endpoints,
                candidate_count=1,
            )

    def test_valid_json_serialization(self):
        candidates = [
            Candidate(
                id="balanced",
                description="Valid.",
                assignments={"plan": "cloud", "act": "local"},
            )
        ]
        artifact = build_artifact(
            workflow=sample_workflow(),
            model_endpoints=sample_endpoints(),
            candidates=candidates,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output = write_artifact(artifact, Path(tmpdir) / "candidates.json")
            data = json.loads(output.read_text())

        self.assertEqual(data["schema_version"], "1.0")
        self.assertEqual(data["candidate_count"], 1)
        self.assertEqual(data["candidates"][0]["assignments"]["plan"], "cloud")

    def test_example_run_config_loads(self):
        config = load_pipeline_config("placement_compiler/examples/qa_pipeline_run.yaml")

        self.assertEqual(config.workflow, "qa")
        self.assertEqual(config.compile.candidates, 3)
        self.assertEqual(config.compile.compiler.provider, "openai")
        self.assertEqual(config.compile.compiler.model, "gpt-5.5")
        self.assertTrue(config.models.exists())
        self.assertTrue(str(config.candidate_artifact).endswith("placement_candidates/qa/candidates.json"))
        self.assertIsNotNone(config.profile)
        self.assertEqual(config.profile.sample_size, 10)
        self.assertTrue(str(config.profile.output).endswith("placement_profiles/qa"))

    def test_pipeline_compile_phase_against_existing_qa_workflow(self):
        response = CandidateSetDraft(
            candidates=[
                Candidate(
                    id="quality",
                    description="Cloud endpoints for all QA placement units.",
                    assignments={
                        "generate_query_or_respond": "gpt-4.1-mini-cloud",
                        "decide_after_retrieval": "gpt-4.1-mini-cloud",
                        "rewrite_question": "gpt-4.1-mini-cloud",
                        "generate_answer": "gpt-4.1-mini-cloud",
                    },
                )
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "qa-candidates.json"
            config = Path(tmpdir) / "qa-pipeline.yaml"
            models = (
                Path.cwd()
                / "placement_compiler"
                / "examples"
                / "model_endpoints.yaml"
            )
            config.write_text(
                "\n".join(
                    [
                        "workflow: qa",
                        f"models: {models}",
                        f"candidate_artifact: {output}",
                        "default_phase: compile",
                        "compile:",
                        "  candidates: 1",
                        "  priorities: [quality]",
                        "  compiler:",
                        "    provider: openai",
                        "    model: gpt-5.5",
                        "    temperature: 0",
                        "  max_attempts: 1",
                    ]
                )
                + "\n"
            )
            paths = pipeline_from_config(
                config,
                phase="compile",
                compiler_llm=FakeCompilerLLM(response),
            )
            data = json.loads(Path(paths["candidate_artifact"]).read_text())

        self.assertEqual(data["workflow_id"], "qa-workflow")
        self.assertEqual(data["candidate_count"], 1)


if __name__ == "__main__":
    unittest.main()
