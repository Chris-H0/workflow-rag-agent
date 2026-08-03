from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage
from pydantic import create_model

from placement_compiler.adapters.workflows import load_workflow_driver
from placement_compiler.core.metadata import build_workflow_metadata
from placement_compiler.core.models import (
    ModelEndpoint,
    PlacementPlan,
    PlacementProposal,
    PlacementUnitSpec,
    WorkflowEdge,
)
from placement_compiler.pipeline.config import CompilerConfig, PipelineConfig
from placement_compiler.pipeline.runner import run_pipeline
from placement_compiler.profiling.models import (
    MetricDefinition,
    EvaluationSettings,
    RunRecord,
    RunTrace,
    SearchArtifact,
    WorkflowResult,
)
from placement_compiler.profiling.runner import (
    PlanEvaluator,
    aggregate_plan_result,
    build_cloud_baseline,
    deterministic_sample,
    write_search_artifacts,
)
from placement_compiler.runtime.endpoint_registry import (
    EndpointRegistry,
    ModelResolver,
    TraceCollector,
)


class FakeCompilerLLM:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.messages = []

    def generate(self, messages, output_schema):
        self.messages.append(messages)
        return self.responses.pop(0)


class MockPlacementModel:
    def __init__(self, model_kwargs=None):
        self.model_kwargs = model_kwargs or {}
        self.structured_schema = None
        self.include_raw = False

    def bind_tools(self, tools):
        return self

    def with_structured_output(self, schema, *, include_raw=False, **kwargs):
        clone = MockPlacementModel(self.model_kwargs)
        clone.structured_schema = schema
        clone.include_raw = include_raw
        return clone

    def invoke(self, messages):
        text = "\n".join(str(getattr(message, "content", message)) for message in messages)
        content = self._content(text)
        usage = {
            "input_tokens": max(1, len(text.split())),
            "output_tokens": max(1, len(content.split())),
        }
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
        if self.structured_schema is not None:
            data = {"decision": "answer"}
            parsed = self.structured_schema.model_validate(data)
            if self.include_raw:
                return {
                    "raw": AIMessage(content=json.dumps(data), usage_metadata=usage),
                    "parsed": parsed,
                    "parsing_error": None,
                }
            return parsed
        return AIMessage(content=content, usage_metadata=usage)

    def _content(self, text):
        if "Review" in text or "generated_test_result" in text:
            return json.dumps({"decision": "final", "comments": "mock review"})
        match = re.search(r"Entry point:\s*([A-Za-z_][A-Za-z0-9_]*)", text)
        entry_point = match.group(1) if match else "add"
        if "Generate tests" in text or "test" in text.lower():
            return f"assert {entry_point}(1, 2) == 3"
        if "Entry point:" in text or "Solve this MBPP" in text:
            return f"def {entry_point}(a, b):\n    return a + b"
        if "plan" in text.lower() or "programming task" in text.lower():
            return "Use a direct implementation."
        return "Paris"


def mock_model_factory(endpoint):
    return MockPlacementModel(dict(endpoint.model_kwargs))


def tiny_workflow():
    return build_workflow_metadata(
        workflow_id="tiny",
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


def endpoint(endpoint_id="strong", location="cloud", cost=2.0):
    return ModelEndpoint(
        id=endpoint_id,
        model="mock",
        location=location,
        provider="mock",
        tool_calling=True,
        structured_output=True,
        input_cost_per_million_tokens=cost,
        output_cost_per_million_tokens=cost,
    )


class FakeDriver:
    primary_metric = MetricDefinition(name="quality", direction="maximise")

    def __init__(self, examples):
        self.examples = examples
        self.calls = []

    def load_examples(self, settings):
        return list(self.examples)

    def example_id(self, example):
        return str(example["id"])

    def prepare(self, examples, runtime=None):
        self.prepared = list(examples)

    def run_example(self, resolver, example, **kwargs):
        self.calls.append((resolver.plan.id, example["id"]))
        return WorkflowResult(
            output={"quality": 1.0},
            metrics={"quality": 1.0},
            latency_seconds=0.01,
        )


class PlacementProfilerTests(unittest.TestCase):
    def test_strongest_cloud_baseline_is_used_everywhere(self):
        workflow = tiny_workflow()
        endpoints = [endpoint("strong"), endpoint("local", location="local")]
        baseline = build_cloud_baseline(workflow, endpoints, "strong")

        self.assertEqual(baseline.source, "baseline")
        self.assertEqual(set(baseline.assignments.values()), {"strong"})
        with self.assertRaisesRegex(ValueError, "not cloud-hosted"):
            build_cloud_baseline(workflow, endpoints, "local")

    def test_evaluator_reuses_one_deterministic_sample_for_every_plan(self):
        examples = [{"id": str(index)} for index in range(20)]
        workflow = tiny_workflow()
        endpoints = [endpoint()]
        driver = FakeDriver(examples)
        evaluator = PlanEvaluator(
            workflow=workflow,
            endpoint_registry=EndpointRegistry(
                endpoints, model_factory=mock_model_factory
            ),
            driver=driver,
            settings=EvaluationSettings(sample_size=10, seed=7),
        )
        selected = evaluator.prepare()
        baseline = build_cloud_baseline(workflow, endpoints, "strong")
        candidate = baseline.model_copy(
            update={"id": "candidate-1", "source": "candidate"}
        )
        _, baseline_runs = evaluator.evaluate(baseline)
        _, candidate_runs = evaluator.evaluate(candidate)

        self.assertEqual(len(selected), 10)
        self.assertEqual(
            [run.example_id for run in baseline_runs],
            [run.example_id for run in candidate_runs],
        )

    def test_trace_cost_is_aggregated_into_plan_result(self):
        workflow = tiny_workflow()
        cloud = endpoint(cost=2.0)
        collector = TraceCollector()
        resolver = ModelResolver(
            plan=PlacementPlan(
                id="candidate-1",
                assignments={"plan": "strong", "act": "strong"},
            ),
            endpoint_registry=EndpointRegistry(
                [cloud], model_factory=mock_model_factory
            ),
            workflow=workflow,
            trace_collector=collector,
        )
        resolver.get_model("plan").invoke([{"role": "user", "content": "hello"}])
        schema = create_model("Decision", decision=(str, ...))
        resolver.get_model("plan").with_structured_output(schema).invoke(
            [{"role": "user", "content": "choose"}]
        )
        plan = resolver.plan
        result = aggregate_plan_result(
            plan,
            [
                RunRecord(
                    plan_id=plan.id,
                    plan_source=plan.source,
                    example_id="1",
                    repeat=0,
                    metrics={"quality": 1.0},
                    latency_seconds=0.01,
                    trace=RunTrace(invocations=collector.invocations),
                )
            ],
        )

        self.assertGreater(result.metrics["cloud_api_cost"], 0)
        self.assertEqual(result.metrics["quality"], 1.0)

    def test_results_writer_does_not_select_or_rank(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            plan = PlacementPlan(
                id="baseline", source="baseline", assignments={"plan": "strong"}
            )
            artifact = SearchArtifact(
                workflow_id="tiny",
                workflow="tiny",
                search_config={},
                primary_metric=MetricDefinition(name="quality", direction="maximise"),
                selected_example_ids=["1"],
                results=[aggregate_plan_result(plan, [])],
            )
            paths = write_search_artifacts(
                artifact=artifact, runs=[], output_dir=tmpdir
            )
            data = json.loads(paths["results"].read_text())

        self.assertEqual(set(paths), {"results", "runs"})
        self.assertNotIn("selected_candidate_id", data)
        self.assertNotIn("ranking", data)

    def test_qa_and_code_sequential_search_smoke(self):
        models = Path("placement_compiler/examples/model_endpoints.yaml").resolve()
        compiler = CompilerConfig(provider="openai", model="gpt-5.5")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            qa_workflow = load_workflow_driver("qa").metadata()
            qa_candidate_1 = {
                unit.id: "gpt-4.1-mini-cloud" for unit in qa_workflow.placement_units
            }
            qa_candidate_2 = dict(qa_candidate_1)
            qa_candidate_2["rewrite_question"] = "qwen-local"
            qa_llm = FakeCompilerLLM(
                PlacementProposal(assignments=qa_candidate_1),
                PlacementProposal(assignments=qa_candidate_2),
            )
            qa_config = PipelineConfig(
                workflow="qa",
                models=models,
                strongest_cloud_endpoint="gpt-5.5-cloud",
                iterations=2,
                sample_size=1,
                compiler=compiler,
                output=tmp / "qa",
                examples=[
                    {
                        "id": "qa-smoke",
                        "level": "hard",
                        "question": "What is the answer?",
                        "answer": "Paris",
                        "supporting_facts": {"title": ["Paris"]},
                        "context": {
                            "title": ["Paris"],
                            "sentences": [["Paris is the answer."]],
                        },
                    }
                ],
            )
            qa_paths = run_pipeline(
                qa_config,
                compiler_llm=qa_llm,
                model_factory=mock_model_factory,
            )
            qa_results = json.loads(qa_paths["results"].read_text())

            self.assertEqual(len(qa_results["results"]), 3)
            self.assertEqual(
                set(qa_results["results"][0]["plan"]["assignments"].values()),
                {"gpt-5.5-cloud"},
            )
            second_prompt = qa_llm.messages[1][1]["content"]
            self.assertIn("candidate-1", second_prompt)
            self.assertIn("exact_match", second_prompt)
            self.assertIn("qa-smoke", second_prompt)

            code_workflow = load_workflow_driver("code").metadata()
            code_candidate = {
                unit.id: "gpt-4.1-mini-cloud"
                for unit in code_workflow.placement_units
            }
            code_config = PipelineConfig(
                workflow="code",
                models=models,
                strongest_cloud_endpoint="gpt-5.5-cloud",
                iterations=1,
                sample_size=1,
                compiler=compiler,
                output=tmp / "code",
                examples=[
                    {
                        "task_id": "Mbpp/999",
                        "entry_point": "add",
                        "prompt": "def add(a, b):\n    pass\n",
                        "canonical_solution": "\ndef add(a, b):\n    return a + b\n",
                        "base_input": [[1, 2], [3, 4]],
                        "plus_input": [[0, 0], [-1, 2]],
                        "atol": 0,
                        "contract": "",
                    }
                ],
            )
            code_paths = run_pipeline(
                code_config,
                compiler_llm=FakeCompilerLLM(
                    PlacementProposal(assignments=code_candidate)
                ),
                model_factory=mock_model_factory,
            )
            code_results = json.loads(code_paths["results"].read_text())
            self.assertEqual(len(code_results["results"]), 2)


if __name__ == "__main__":
    unittest.main()
