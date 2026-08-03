from __future__ import annotations

import json
import re
import tempfile
import time
import unittest
from pathlib import Path

from pydantic import create_model
from langchain_core.messages import AIMessage

from placement_compiler.core.artifacts import build_artifact
from placement_compiler.pipeline.config import load_pipeline_config
from placement_compiler.pipeline.runner import profile_candidates
from placement_compiler.runtime.endpoint_registry import EndpointRegistry, ModelResolver, TraceCollector
from placement_compiler.core.metadata import build_workflow_metadata
from placement_compiler.core.models import (
    ModelEndpoint,
    PlacementPlan,
    PlacementUnitSpec,
    WorkflowEdge,
)
from placement_compiler.profiling.models import (
    CandidateProfile,
    MetricDefinition,
    ProfileArtifact,
    ProfileSettings,
    QualityConstraint,
    RankingConfig,
    RankingObjective,
    RunRecord,
    RunTrace,
    WorkflowResult,
)
from placement_compiler.profiling import (
    CandidateProfiler,
    aggregate_candidate_profile,
    apply_quality_constraints,
    build_baseline_plan,
    deterministic_sample,
    load_candidate_artifact,
    rank_candidates,
    write_profile_artifacts,
)
from placement_compiler.adapters.workflows import load_workflow_driver


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


def mock_endpoint(endpoint_id="mock-cloud", location="cloud", cost=1.0):
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


class MockPlacementModel:
    def __init__(self, model_kwargs=None):
        self.model_kwargs = model_kwargs or {}
        self.structured_schema = None
        self.include_raw = False

    def bind_tools(self, tools):
        return self

    def with_structured_output(self, schema, *, include_raw=False):
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
            data = self.model_kwargs.get("structured_payload") or {"decision": "answer"}
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
        entry_point_match = re.search(
            r"Entry point:\s*([A-Za-z_][A-Za-z0-9_]*)", text
        )
        entry_point = entry_point_match.group(1) if entry_point_match else "add"
        if "Generate tests" in text or "test" in text.lower():
            return f"assert {entry_point}(1, 2) == 3"
        if "Entry point:" in text or "Solve this MBPP" in text:
            return f"def {entry_point}(a, b):\n    return a + b"
        if "plan" in text.lower() or "programming task" in text.lower():
            return "Use a direct implementation."
        return str(self.model_kwargs.get("default_answer", "Paris"))


def mock_model_factory(endpoint):
    return MockPlacementModel(dict(endpoint.model_kwargs))


def write_artifact(path: Path, workflow, endpoints, candidates):
    artifact = build_artifact(
        workflow=workflow,
        model_endpoints=endpoints,
        candidates=candidates,
    )
    path.write_text(json.dumps(artifact.model_dump(mode="json"), indent=2) + "\n")
    return artifact


class FakeDriver:
    primary_metric = MetricDefinition(name="quality", direction="maximise")

    def __init__(self, examples, qualities, delay=0.0, fail_candidate=None):
        self.examples = examples
        self.qualities = qualities
        self.delay = delay
        self.fail_candidate = fail_candidate
        self.calls = []

    def load_examples(self, profile):
        return list(self.examples)

    def example_id(self, example):
        return str(example["id"])

    def prepare(self, examples):
        self.prepared_ids = [example["id"] for example in examples]

    def run_example(self, model_resolver, example, **kwargs):
        plan = model_resolver.plan
        self.calls.append((plan.id, example["id"]))
        if self.delay:
            time.sleep(self.delay)
        if plan.id == self.fail_candidate:
            raise RuntimeError("forced failure")
        quality = self.qualities.get(plan.id, 0.0)
        return WorkflowResult(
            output={"quality": self.qualities.get(plan.id, 0.0)},
            metrics={"quality": quality},
            latency_seconds=0.01,
            trace=RunTrace(),
        )


class PlacementProfilerTests(unittest.TestCase):
    def test_candidate_artifact_loading_and_baseline_plan(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "candidates.json"
            artifact = write_artifact(
                path,
                tiny_workflow(),
                [mock_endpoint()],
                [
                    PlacementPlan(
                        id="candidate-a",
                        description="A",
                        assignments={"plan": "mock-cloud", "act": "mock-cloud"},
                    )
                ],
            )

            loaded = load_candidate_artifact(path)
            baseline = build_baseline_plan(
                artifact=loaded,
                baseline_type="all_endpoint",
                endpoint_id="mock-cloud",
            )

        self.assertEqual(loaded.workflow_id, artifact.workflow_id)
        self.assertEqual(baseline.assignments, {"plan": "mock-cloud", "act": "mock-cloud"})

    def test_stable_plan_to_placement_unit_resolution(self):
        workflow = tiny_workflow()
        resolver = ModelResolver(
            plan=PlacementPlan(
                id="ok",
                assignments={"plan": "mock-cloud", "act": "mock-cloud"},
            ),
            endpoint_registry=EndpointRegistry(
                [mock_endpoint()], model_factory=mock_model_factory
            ),
            workflow=workflow,
            trace_collector=TraceCollector(),
        )

        self.assertIsNotNone(resolver.get_model("plan"))

        with self.assertRaisesRegex(Exception, "missing assignments"):
            ModelResolver(
                plan=PlacementPlan(id="bad", assignments={"plan": "mock-cloud"}),
                endpoint_registry=EndpointRegistry(
                    [mock_endpoint()], model_factory=mock_model_factory
                ),
                workflow=workflow,
                trace_collector=TraceCollector(),
            )

        context_workflow = build_workflow_metadata(
            workflow_id="context",
            node_ids=["__start__", "plan", "__end__"],
            edges=[
                WorkflowEdge(source="__start__", target="plan"),
                WorkflowEdge(source="plan", target="__end__"),
            ],
            placement_units={
                "plan": PlacementUnitSpec(required_context_window=1000)
            },
        )
        with self.assertRaisesRegex(Exception, "context window"):
            ModelResolver(
                plan=PlacementPlan(id="too-small", assignments={"plan": "small"}),
                endpoint_registry=EndpointRegistry(
                    [mock_endpoint(endpoint_id="small").model_copy(update={"context_window": 10})],
                    model_factory=mock_model_factory,
                ),
                workflow=context_workflow,
                trace_collector=TraceCollector(),
            )

    def test_deterministic_sample_selection(self):
        examples = [{"id": str(index)} for index in range(10)]
        first = deterministic_sample(examples, sample_size=4, seed=7, example_id=lambda ex: ex["id"])
        second = deterministic_sample(examples, sample_size=4, seed=7, example_id=lambda ex: ex["id"])

        self.assertEqual([ex["id"] for ex in first], [ex["id"] for ex in second])
        self.assertEqual(len(first), 4)

    def test_every_candidate_receives_same_examples(self):
        workflow = tiny_workflow()
        artifact = build_artifact(
            workflow=workflow,
            model_endpoints=[mock_endpoint()],
            candidates=[
                PlacementPlan(id="a", description="A", assignments={"plan": "mock-cloud", "act": "mock-cloud"}),
                PlacementPlan(id="b", description="B", assignments={"plan": "mock-cloud", "act": "mock-cloud"}),
            ],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "candidates.json"
            path.write_text(json.dumps(artifact.model_dump(mode="json")) + "\n")
            loaded = load_candidate_artifact(path)
            driver = FakeDriver(
                [{"id": "1"}, {"id": "2"}, {"id": "3"}],
                {"baseline-all-mock-cloud": 1.0, "a": 0.9, "b": 0.8},
            )
            profiler = CandidateProfiler(
                candidate_artifact=loaded,
                source_candidate_artifact=path,
                endpoint_registry=EndpointRegistry(
                    artifact.model_endpoints, model_factory=mock_model_factory
                ),
                driver=driver,
                profile=ProfileSettings(sample_size=2, seed=1, repeats=1),
                baseline_plan=build_baseline_plan(
                    artifact=artifact,
                    baseline_type="all_endpoint",
                    endpoint_id="mock-cloud",
                ),
                quality_constraint=QualityConstraint(metric="quality", max_drop_from_baseline=0.5),
                ranking=RankingConfig(),
                workflow_name="fake",
            )
            profile, runs = profiler.run()

        candidate_to_examples = {}
        for run in runs:
            candidate_to_examples.setdefault(run.candidate_id, []).append(run.example_id)
        self.assertEqual(candidate_to_examples["a"], candidate_to_examples["b"])
        self.assertIn(profile.selected_candidate_id, {"a", "b"})

    def test_quality_constraints_higher_and_lower(self):
        baseline = CandidateProfile(
            candidate_id="baseline",
            candidate_source="baseline",
            assignments={},
            metrics={"accuracy": 0.9, "error_rate": 0.1},
        )
        high_candidates = [
            CandidateProfile(candidate_id="ok", candidate_source="candidate", assignments={}, metrics={"accuracy": 0.86}),
            CandidateProfile(candidate_id="bad", candidate_source="candidate", assignments={}, metrics={"accuracy": 0.7}),
        ]
        apply_quality_constraints(
            baseline=baseline,
            candidates=high_candidates,
            primary_metric=MetricDefinition(name="accuracy", direction="maximise"),
            quality_constraint=QualityConstraint(metric="accuracy", max_drop_from_baseline=0.05),
        )
        self.assertTrue(high_candidates[0].feasible)
        self.assertFalse(high_candidates[1].feasible)

        low_candidates = [
            CandidateProfile(candidate_id="ok", candidate_source="candidate", assignments={}, metrics={"error_rate": 0.12}),
            CandidateProfile(candidate_id="bad", candidate_source="candidate", assignments={}, metrics={"error_rate": 0.2}),
        ]
        apply_quality_constraints(
            baseline=baseline,
            candidates=low_candidates,
            primary_metric=MetricDefinition(name="error_rate", direction="minimise"),
            quality_constraint=QualityConstraint(metric="error_rate", max_drop_from_baseline=0.05),
        )
        self.assertTrue(low_candidates[0].feasible)
        self.assertFalse(low_candidates[1].feasible)

    def test_cost_latency_ranking_and_no_selection(self):
        candidates = [
            CandidateProfile(
                candidate_id="slow-cheap",
                candidate_source="candidate",
                assignments={},
                feasible=True,
                metrics={"quality": 1.0, "cloud_api_cost": 0.1, "mean_latency_seconds": 2.0},
            ),
            CandidateProfile(
                candidate_id="fast-expensive",
                candidate_source="candidate",
                assignments={},
                feasible=True,
                metrics={"quality": 1.0, "cloud_api_cost": 0.2, "mean_latency_seconds": 1.0},
            ),
        ]
        selected = rank_candidates(
            candidates,
            ranking=RankingConfig(
                objectives=[
                    RankingObjective(metric="cloud_api_cost", direction="minimise"),
                    RankingObjective(metric="mean_latency_seconds", direction="minimise"),
                ]
            ),
            primary_metric=MetricDefinition(name="quality", direction="maximise"),
        )
        self.assertEqual(selected, "slow-cheap")
        self.assertEqual(candidates[0].rank, 1)

        for candidate in candidates:
            candidate.feasible = False
            candidate.rank = None
        self.assertIsNone(
            rank_candidates(
                candidates,
                ranking=RankingConfig(),
                primary_metric=MetricDefinition(name="quality", direction="maximise"),
            )
        )
        self.assertIsNotNone(candidates[0].diagnostic_rank)

    def test_failed_run_and_cost_accounting(self):
        workflow = tiny_workflow()
        endpoint = mock_endpoint(cost=2.0)
        trace_collector = TraceCollector()
        resolver = ModelResolver(
            plan=PlacementPlan(id="cost", assignments={"plan": "mock-cloud", "act": "mock-cloud"}),
            endpoint_registry=EndpointRegistry(
                [endpoint], model_factory=mock_model_factory
            ),
            workflow=workflow,
            trace_collector=trace_collector,
        )
        resolver.get_model("plan").invoke([{"role": "user", "content": "hello"}])
        decision_schema = create_model("Decision", decision=(str, ...))
        resolver.get_model("plan").with_structured_output(decision_schema).invoke(
            [{"role": "user", "content": "choose"}]
        )
        profile = aggregate_candidate_profile(
            PlacementPlan(id="cost", assignments={"plan": "mock-cloud", "act": "mock-cloud"}),
            [
                RunRecord(
                    candidate_id="cost",
                    candidate_source="candidate",
                    example_id="1",
                    repeat=0,
                    metrics={"quality": 1.0},
                    output={"quality": 1.0},
                    trace=RunTrace(invocations=trace_collector.invocations),
                    latency_seconds=0.01,
                )
            ],
        )

        self.assertGreater(profile.metrics["cloud_api_cost"], 0)
        self.assertEqual(profile.metrics["failed_run_count"], 0)

        artifact = build_artifact(
            workflow=workflow, model_endpoints=[endpoint], candidates=[]
        )
        driver = FakeDriver([{"id": "1"}], {}, fail_candidate="baseline")
        profiler = CandidateProfiler(
            candidate_artifact=artifact,
            source_candidate_artifact="x",
            endpoint_registry=EndpointRegistry(
                [endpoint], model_factory=mock_model_factory
            ),
            driver=driver,
            profile=ProfileSettings(sample_size=1),
            baseline_plan=PlacementPlan(id="baseline", source="baseline", assignments={"plan": "mock-cloud", "act": "mock-cloud"}),
            quality_constraint=QualityConstraint(metric="quality"),
            ranking=RankingConfig(),
            workflow_name="fake",
        )
        _, runs = profiler.run()
        self.assertIn("forced failure", runs[0].error)

    def test_json_jsonl_csv_artifact_generation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            profile = ProfileArtifact(
                source_candidate_artifact="candidates.json",
                workflow_id="tiny",
                workflow="fake",
                profile_config={},
                primary_metric=MetricDefinition(name="quality", direction="maximise"),
                selected_example_ids=["1"],
                baseline=CandidateProfile(
                    candidate_id="baseline",
                    candidate_source="baseline",
                    assignments={},
                    metrics={"quality": 1.0},
                ),
                quality_constraint=QualityConstraint(metric="quality"),
                ranking=RankingConfig(),
                candidates=[
                    CandidateProfile(
                        candidate_id="a",
                        candidate_source="candidate",
                        assignments={"plan": "mock-cloud"},
                        feasible=True,
                        rank=1,
                        metrics={"quality": 1.0, "cloud_api_cost": 0.1},
                    )
                ],
                selected_candidate_id="a",
            )
            paths = write_profile_artifacts(profile=profile, runs=[], output_dir=tmpdir)

            self.assertTrue(paths["profile"].exists())
            self.assertTrue(paths["runs"].exists())
            self.assertTrue(paths["candidates"].exists())
            selected = json.loads(paths["selected_plan"].read_text())
            self.assertEqual(selected["selected_plan"], {"plan": "mock-cloud"})

    def test_qa_and_code_profile_smoke_with_mock_models(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            qa_artifact = tmp / "qa-candidates.json"
            qa_workflow = load_workflow_driver("qa").metadata()
            qa_assignments = {
                unit.id: "mock-cloud" for unit in qa_workflow.placement_units
            }
            write_artifact(
                qa_artifact,
                qa_workflow,
                [mock_endpoint()],
                [PlacementPlan(id="mock-qa", description="mock", assignments=qa_assignments)],
            )
            qa_config = tmp / "qa-profile.yaml"
            models = Path.cwd() / "placement_compiler" / "examples" / "model_endpoints.yaml"
            qa_config.write_text(
                f"""
workflow: qa
models: {models}
candidate_artifact: {qa_artifact}
compile:
  candidates: 1
  compiler:
    provider: openai
    model: gpt-5.5
profile:
  sample_size: 1
  seed: 1
  repeats: 1
  examples:
    - id: qa-smoke
      level: hard
      question: What is the answer?
      answer: Paris
      supporting_facts:
        title: [Paris]
      context:
        title: [Paris]
        sentences:
          - [Paris is the answer.]
  baseline:
    type: all_endpoint
    endpoint_id: mock-cloud
  quality_constraint:
    metric: exact_match
    max_drop_from_baseline: 0
  ranking:
    objectives:
      - metric: cloud_api_cost
        direction: minimise
  output: {tmp / "qa-profile"}
"""
            )
            qa_paths = profile_candidates(
                load_pipeline_config(qa_config), model_factory=mock_model_factory
            )
            qa_profile = json.loads(qa_paths["profile"].read_text())
            self.assertEqual(qa_profile["selected_candidate_id"], "mock-qa")

            code_artifact = tmp / "code-candidates.json"
            code_workflow = load_workflow_driver("code").metadata()
            code_assignments = {
                unit.id: "mock-cloud" for unit in code_workflow.placement_units
            }
            write_artifact(
                code_artifact,
                code_workflow,
                [mock_endpoint()],
                [PlacementPlan(id="mock-code", description="mock", assignments=code_assignments)],
            )
            code_config = tmp / "code-profile.yaml"
            code_config.write_text(
                f"""
workflow: code
models: {models}
candidate_artifact: {code_artifact}
compile:
  candidates: 1
  compiler:
    provider: openai
    model: gpt-5.5
profile:
  sample_size: 1
  seed: 1
  repeats: 1
  examples:
    - task_id: Mbpp/999
      entry_point: add
      prompt: "def add(a, b):\\n    pass\\n"
      canonical_solution: "\\ndef add(a, b):\\n    return a + b\\n"
      base_input:
        - [1, 2]
        - [3, 4]
      plus_input:
        - [0, 0]
        - [-1, 2]
      atol: 0
      contract: ""
  baseline:
    type: all_endpoint
    endpoint_id: mock-cloud
  quality_constraint:
    metric: mbpp_plus_pass
    max_drop_from_baseline: 0
  ranking:
    objectives:
      - metric: cloud_api_cost
        direction: minimise
  output: {tmp / "code-profile"}
"""
            )
            code_paths = profile_candidates(
                load_pipeline_config(code_config), model_factory=mock_model_factory
            )
            code_profile = json.loads(code_paths["profile"].read_text())
            self.assertEqual(code_profile["selected_candidate_id"], "mock-code")


if __name__ == "__main__":
    unittest.main()
