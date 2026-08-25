from __future__ import annotations

import json
import unittest

from placement_compiler.adapters.workflows import load_workflow_driver
from placement_compiler.core.catalogue import load_model_catalogue
from placement_compiler.core.metadata import build_workflow_metadata
from placement_compiler.core.models import (
    ModelEndpoint,
    PlacementProposal,
    PlacementUnitSpec,
    WorkflowEdge,
)
from placement_compiler.generation.candidates import (
    CandidateGenerationError,
    CandidateGenerator,
)
from placement_compiler.pipeline.config import load_pipeline_config
from workflows.code.main import NODE_MODEL_CONFIG


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
        placement_units={
            "plan": PlacementUnitSpec(structured_output_required=True),
            "act": PlacementUnitSpec(tool_use=True),
        },
    )


def sample_endpoints():
    return [
        ModelEndpoint(
            id="local",
            model="qwen",
            location="local",
            provider="ollama",
            tool_calling=True,
            structured_output=False,
        ),
        ModelEndpoint(
            id="strong-cloud",
            model="gpt-5.5",
            location="cloud",
            provider="openai",
            tool_calling=True,
            structured_output=True,
        ),
        ModelEndpoint(
            id="mid-cloud",
            model="gpt-mini",
            location="cloud",
            provider="openai",
            tool_calling=True,
            structured_output=True,
        ),
    ]


def result(assignments, quality=1.0):
    return {
        "plan": {"id": "baseline", "source": "baseline", "assignments": assignments},
        "metrics": {"quality": quality, "cloud_api_cost": 1.0},
    }


class PlacementCompilerTests(unittest.TestCase):
    def test_existing_workflow_metadata_is_manifest_first(self):
        qa_metadata = load_workflow_driver("qa").metadata()
        units = {unit.id: unit for unit in qa_metadata.placement_units}

        self.assertEqual(qa_metadata.workflow_id, "qa-workflow")
        self.assertEqual(
            set(units),
            {
                "generate_query_or_respond",
                "decide_after_retrieval",
                "rewrite_question",
                "generate_answer",
            },
        )
        self.assertTrue(units["generate_query_or_respond"].tool_use)
        self.assertTrue(units["decide_after_retrieval"].structured_output_required)
        self.assertEqual(
            load_workflow_driver("qa").primary_metric.name,
            "answer_token_f1",
        )

        code_metadata = load_workflow_driver("code").metadata()
        units = {unit.id: unit for unit in code_metadata.placement_units}
        self.assertTrue(units["implement_solution"].user_facing_output)
        self.assertFalse(units["review_solution"].user_facing_output)
        self.assertFalse(units["review_solution"].structured_output_required)
        self.assertTrue(units["review_solution"].branch_control)

    def test_proposal_receives_all_previous_evaluation_results(self):
        history = [
            result({"plan": "strong-cloud", "act": "strong-cloud"}),
            {
                "plan": {
                    "id": "candidate-1",
                    "source": "candidate",
                    "assignments": {"plan": "strong-cloud", "act": "local"},
                },
                "metrics": {"quality": 0.8, "cloud_api_cost": 0.2},
            },
        ]
        llm = FakeCompilerLLM(
            PlacementProposal(
                assignments={"plan": "strong-cloud", "act": "strong-cloud"},
            ),
            PlacementProposal(
                assignments={"plan": "strong-cloud", "act": "local"},
            ),
        )

        with self.assertRaises(CandidateGenerationError):
            CandidateGenerator(llm, max_attempts=2).propose(
                workflow=sample_workflow(),
                model_endpoints=sample_endpoints(),
                iteration=2,
                evaluated_results=history,
            )

        prompt = llm.messages[0][1]["content"]
        self.assertIn("evaluated_results", prompt)
        self.assertIn('"quality": 0.8', prompt)
        self.assertIn("candidate-1", prompt)

    def test_complete_candidate_can_change_multiple_modules(self):
        llm = FakeCompilerLLM(
            PlacementProposal(
                assignments={"plan": "mid-cloud", "act": "local"},
                rationale={"plan": "reduce cost", "act": "run locally"},
            )
        )
        candidate = CandidateGenerator(llm, max_attempts=1).propose(
            workflow=sample_workflow(),
            model_endpoints=sample_endpoints(),
            iteration=1,
            evaluated_results=[
                result({"plan": "strong-cloud", "act": "strong-cloud"})
            ],
        )

        self.assertEqual(candidate.id, "candidate-1")
        self.assertEqual(candidate.assignments["plan"], "mid-cloud")
        self.assertEqual(candidate.assignments["act"], "local")
        self.assertIn("any number of placement units", llm.messages[0][0]["content"])

    def test_invalid_capabilities_and_unknown_assignments_are_rejected(self):
        invalid = PlacementProposal(assignments={"plan": "local", "extra": "local"})
        with self.assertRaisesRegex(CandidateGenerationError, "structured output|unknown"):
            CandidateGenerator(FakeCompilerLLM(invalid), max_attempts=1).propose(
                workflow=sample_workflow(),
                model_endpoints=sample_endpoints(),
                iteration=1,
                evaluated_results=[
                    result({"plan": "strong-cloud", "act": "strong-cloud"})
                ],
            )

    def test_example_configs_use_ten_samples_and_strongest_cloud_baseline(self):
        for workflow in ("qa", "code"):
            config = load_pipeline_config(
                f"placement_compiler/examples/{workflow}_pipeline_run.yaml"
            )
            self.assertEqual(config.sample_size, 10)
            self.assertEqual(config.strongest_cloud_endpoint, "gpt-5.5-cloud")
            self.assertEqual(config.compiler.model, "gpt-5.5")
            self.assertTrue(config.models.exists())
            self.assertNotIn(
                "max_drop_from_baseline",
                json.dumps(config.model_dump(mode="json")),
            )

        endpoints = load_model_catalogue(
            "placement_compiler/examples/model_endpoints.yaml"
        )
        qwen = next(endpoint for endpoint in endpoints if endpoint.id == "qwen-local")
        self.assertTrue(qwen.structured_output)
        self.assertEqual(qwen.model_kwargs["num_predict"], 768)
        qwen_configs = [
            config
            for config in NODE_MODEL_CONFIG.values()
            if config["model"] == "qwen3.5:2b"
        ]
        self.assertTrue(qwen_configs)
        self.assertTrue(
            all(config["num_predict"] == 768 for config in qwen_configs)
        )

    def test_initial_code_config_matches_initial_qa_search_scale(self):
        config = load_pipeline_config(
            "placement_compiler/examples/code_pipeline_initial.yaml"
        )

        self.assertEqual(config.workflow, "code")
        self.assertEqual(config.sample_size, 50)
        self.assertEqual(config.iterations, 12)
        self.assertEqual(config.seed, 42)
        self.assertEqual(config.repeats, 1)
        self.assertEqual(config.strongest_cloud_endpoint, "gpt-5.5-cloud")
        self.assertEqual(config.compiler.model, "gpt-5.5")
        self.assertEqual(config.max_attempts, 3)

    def test_structured_qa_configs_are_three_independent_full_searches(self):
        outputs = set()
        for sequence in range(1, 4):
            config = load_pipeline_config(
                "placement_compiler/examples/"
                f"qa_pipeline_structured_v2_{sequence:02d}.yaml"
            )
            self.assertEqual(config.workflow, "qa")
            self.assertEqual(config.sample_size, 50)
            self.assertEqual(config.iterations, 12)
            self.assertEqual(config.seed, 42)
            self.assertEqual(config.repeats, 1)
            outputs.add(config.output)

        self.assertEqual(len(outputs), 3)
        self.assertEqual(
            {output.name for output in outputs},
            {"compiler-sequence-01", "compiler-sequence-02", "compiler-sequence-03"},
        )
        self.assertTrue(all(output.parent.name == "qa" for output in outputs))


if __name__ == "__main__":
    unittest.main()
