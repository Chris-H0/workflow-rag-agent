from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from placement_compiler.artifacts import build_artifact, write_artifact
from placement_compiler.candidates import CandidateGenerationError, CandidateGenerator
from placement_compiler.cli import generate_command
from placement_compiler.config import load_run_config
from placement_compiler.metadata import build_workflow_metadata
from placement_compiler.models import (
    Candidate,
    CandidateSetDraft,
    ModelEndpoint,
    NodeRegistryMetadata,
    WorkflowEdge,
)
from placement_compiler.workflows import load_existing_workflow_metadata


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

    def test_generate_command_against_existing_qa_workflow(self):
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
                ),
                Candidate(
                    id="balanced",
                    description="Local query generation with cloud routing and answer.",
                    assignments={
                        "generate_query_or_respond": "qwen-local",
                        "decide_after_retrieval": "gpt-4.1-mini-cloud",
                        "rewrite_question": "qwen-local",
                        "generate_answer": "gpt-4.1-mini-cloud",
                    },
                ),
                Candidate(
                    id="local-first",
                    description="Local where capabilities allow it.",
                    assignments={
                        "generate_query_or_respond": "qwen-local",
                        "decide_after_retrieval": "gpt-4.1-mini-cloud",
                        "rewrite_question": "qwen-local",
                        "generate_answer": "qwen-local",
                    },
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "qa-candidates.json"
            config = Path(tmpdir) / "qa-run.yaml"
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
                        "candidates: 3",
                        f"output: {output}",
                        "priorities:",
                        "  - quality",
                        "compiler:",
                        "  provider: openai",
                        "  model: gpt-5.5",
                        "  temperature: 0",
                        "max_attempts: 1",
                    ]
                )
                + "\n"
            )
            args = argparse.Namespace(
                config=str(config),
            )
            path = generate_command(args, compiler_llm=FakeCompilerLLM(response))
            data = json.loads(path.read_text())

        self.assertEqual(data["workflow_id"], "qa-workflow")
        self.assertEqual(data["candidate_count"], 3)
        self.assertEqual(len(data["workflow"]["nodes"]), 6)

    def test_example_run_config_loads(self):
        config = load_run_config("placement_compiler/examples/qa_candidate_run.yaml")

        self.assertEqual(config.workflow, "qa")
        self.assertEqual(config.candidates, 3)
        self.assertEqual(config.compiler.provider, "openai")
        self.assertEqual(config.compiler.model, "gpt-5.5")
        self.assertTrue(config.models.exists())
        self.assertTrue(str(config.output).endswith("placement_candidates/qa/candidates.json"))


if __name__ == "__main__":
    unittest.main()
