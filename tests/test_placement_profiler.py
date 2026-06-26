from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

from pydantic import create_model

from placement_compiler.artifacts import build_artifact
from placement_compiler.cli import profile_from_config
from placement_compiler.endpoint_registry import EndpointRegistry, ModelResolver, TraceCollector
from placement_compiler.metadata import build_workflow_metadata
from placement_compiler.models import Candidate, ModelEndpoint, NodeRegistryMetadata, WorkflowEdge
from placement_compiler.profile_models import (
    CandidateProfile,
    MetricDefinition,
    PlacementPlan,
    ProfileArtifact,
    ProfileSettings,
    QualityConstraint,
    RankingConfig,
    RankingObjective,
    RunTrace,
    WorkflowExecution,
)
from placement_compiler.profiling import (
    CandidateProfiler,
    aggregate_candidate_profile,
    apply_quality_constraints,
    build_baseline_plan,
    deterministic_sample,
    load_candidate_artifact,
    rank_candidates,
    run_record_from_execution,
    write_profile_artifacts,
)
from placement_compiler.workflows import load_existing_workflow_metadata


def tiny_workflow():
    return build_workflow_metadata(
        workflow_id="tiny",
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


def write_artifact(path: Path, workflow, endpoints, candidates):
    artifact = build_artifact(
        workflow=workflow,
        model_endpoints=endpoints,
        candidates=candidates,
    )
    path.write_text(json.dumps(artifact.model_dump(mode="json"), indent=2) + "\n")
    return artifact


class FakeEvaluationAdapter:
    primary_metric = MetricDefinition(name="quality", direction="maximise")

    def __init__(self, examples):
        self.examples = examples

    def load_examples(self, profile):
        return list(self.examples)

    def example_id(self, example):
        return str(example["id"])

    def score(self, example, output):
        return {"quality": output["quality"]}


class FakeRuntimeAdapter:
    def __init__(self, qualities, delay=0.0, fail_candidate=None):
        self.qualities = qualities
        self.delay = delay
        self.fail_candidate = fail_candidate
        self.calls = []

    def prepare(self, examples):
        self.prepared_ids = [example["id"] for example in examples]

    def run_example(self, plan, example, repeat):
        self.calls.append((plan.id, example["id"], repeat))
        if self.delay:
            time.sleep(self.delay)
        if plan.id == self.fail_candidate:
            return WorkflowExecution(
                output={},
                latency_seconds=0.01,
                error="forced failure",
            )
        return WorkflowExecution(
            output={"quality": self.qualities.get(plan.id, 0.0)},
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
                    Candidate(
                        id="candidate-a",
                        description="A",
                        assignments={"plan": "mock-cloud", "act": "mock-cloud"},
                    )
                ],
            )

            loaded = load_candidate_artifact(path)
            baseline = build_baseline_plan(
                artifact=loaded.artifact,
                baseline_type="all_endpoint",
                endpoint_id="mock-cloud",
            )

        self.assertEqual(loaded.artifact.workflow_id, artifact.workflow_id)
        self.assertEqual(baseline.assignments, {"plan": "mock-cloud", "act": "mock-cloud"})

    def test_stable_plan_to_placement_unit_resolution(self):
        workflow = tiny_workflow()
        resolver = ModelResolver(
            plan=PlacementPlan(
                id="ok",
                assignments={"plan": "mock-cloud", "act": "mock-cloud"},
            ),
            endpoint_registry=EndpointRegistry([mock_endpoint()]),
            workflow=workflow,
            trace_collector=TraceCollector(),
        )

        self.assertIsNotNone(resolver.get_model("plan"))

        with self.assertRaisesRegex(Exception, "missing placement units"):
            ModelResolver(
                plan=PlacementPlan(id="bad", assignments={"plan": "mock-cloud"}),
                endpoint_registry=EndpointRegistry([mock_endpoint()]),
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
            registry={
                "plan": NodeRegistryMetadata(
                    is_llm_placement_unit=True,
                    required_context_window=1000,
                )
            },
        )
        with self.assertRaisesRegex(Exception, "context window"):
            ModelResolver(
                plan=PlacementPlan(id="too-small", assignments={"plan": "small"}),
                endpoint_registry=EndpointRegistry(
                    [mock_endpoint(endpoint_id="small").model_copy(update={"context_window": 10})]
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
                Candidate(id="a", description="A", assignments={"plan": "mock-cloud", "act": "mock-cloud"}),
                Candidate(id="b", description="B", assignments={"plan": "mock-cloud", "act": "mock-cloud"}),
            ],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "candidates.json"
            path.write_text(json.dumps(artifact.model_dump(mode="json")) + "\n")
            loaded = load_candidate_artifact(path)
            runtime = FakeRuntimeAdapter({"baseline-all-mock-cloud": 1.0, "a": 0.9, "b": 0.8})
            profiler = CandidateProfiler(
                candidate_artifact=loaded,
                evaluation_adapter=FakeEvaluationAdapter([{"id": "1"}, {"id": "2"}, {"id": "3"}]),
                runtime_adapter=runtime,
                profile=ProfileSettings(sample_size=2, seed=1, repeats=1, timeout_seconds=5),
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

    def test_timeout_failed_run_and_cost_accounting(self):
        workflow = tiny_workflow()
        endpoint = mock_endpoint(cost=2.0)
        trace_collector = TraceCollector()
        resolver = ModelResolver(
            plan=PlacementPlan(id="cost", assignments={"plan": "mock-cloud", "act": "mock-cloud"}),
            endpoint_registry=EndpointRegistry([endpoint]),
            workflow=workflow,
            trace_collector=trace_collector,
        )
        resolver.get_model("plan").invoke([{"role": "user", "content": "hello"}])
        decision_schema = create_model("Decision", decision=(str, ...))
        resolver.get_model("plan").with_structured_output(decision_schema).invoke(
            [{"role": "user", "content": "choose"}]
        )
        execution = WorkflowExecution(
            output={"quality": 1.0},
            trace=RunTrace(invocations=trace_collector.invocations),
            latency_seconds=0.01,
        )
        profile = aggregate_candidate_profile(
            PlacementPlan(id="cost", assignments={"plan": "mock-cloud", "act": "mock-cloud"}),
            [
                run_record_from_execution(
                    plan=PlacementPlan(id="cost", assignments={"plan": "mock-cloud", "act": "mock-cloud"}),
                    example_id="1",
                    repeat=0,
                    execution=execution,
                    metrics={"quality": 1.0},
                )
            ],
        )

        self.assertGreater(profile.metrics["cloud_api_cost"], 0)
        self.assertEqual(profile.metrics["failed_run_count"], 0)

        runtime = FakeRuntimeAdapter({}, delay=0.2)
        profiler = CandidateProfiler(
            candidate_artifact=type("Loaded", (), {"artifact": build_artifact(workflow=workflow, model_endpoints=[endpoint], candidates=[]), "path": Path("x")})(),
            evaluation_adapter=FakeEvaluationAdapter([{"id": "1"}]),
            runtime_adapter=runtime,
            profile=ProfileSettings(sample_size=1, timeout_seconds=0.01),
            baseline_plan=PlacementPlan(id="baseline", source="baseline", assignments={"plan": "mock-cloud", "act": "mock-cloud"}),
            quality_constraint=QualityConstraint(metric="quality"),
            ranking=RankingConfig(),
            workflow_name="fake",
        )
        _, runs = profiler.run()
        self.assertTrue(runs[0].timed_out)

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
            qa_workflow = load_existing_workflow_metadata("qa")
            qa_assignments = {
                node.id: "mock-cloud" for node in qa_workflow.placement_units()
            }
            write_artifact(
                qa_artifact,
                qa_workflow,
                [mock_endpoint()],
                [Candidate(id="mock-qa", description="mock", assignments=qa_assignments)],
            )
            qa_config = tmp / "qa-profile.yaml"
            qa_config.write_text(
                f"""
workflow: qa
candidate_artifact: {qa_artifact}
profile:
  sample_size: 1
  seed: 1
  repeats: 1
  timeout_seconds: 20
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
            qa_paths = profile_from_config(qa_config)
            qa_profile = json.loads(qa_paths["profile"].read_text())
            self.assertEqual(qa_profile["selected_candidate_id"], "mock-qa")

            code_artifact = tmp / "code-candidates.json"
            code_workflow = load_existing_workflow_metadata("code")
            code_assignments = {
                node.id: "mock-cloud" for node in code_workflow.placement_units()
            }
            write_artifact(
                code_artifact,
                code_workflow,
                [mock_endpoint()],
                [Candidate(id="mock-code", description="mock", assignments=code_assignments)],
            )
            code_config = tmp / "code-profile.yaml"
            code_config.write_text(
                f"""
workflow: code
candidate_artifact: {code_artifact}
profile:
  sample_size: 1
  seed: 1
  repeats: 1
  timeout_seconds: 30
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
            code_paths = profile_from_config(code_config)
            code_profile = json.loads(code_paths["profile"].read_text())
            self.assertEqual(code_profile["selected_candidate_id"], "mock-code")


if __name__ == "__main__":
    unittest.main()
