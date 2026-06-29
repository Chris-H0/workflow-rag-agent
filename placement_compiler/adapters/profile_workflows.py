"""Generic profiling adapters backed by workflow-owned evaluation hooks."""

from __future__ import annotations

import importlib
import time
from typing import Any

from placement_compiler.runtime.endpoint_registry import EndpointRegistry, ModelResolver, TraceCollector
from placement_compiler.profiling.models import (
    MetricDefinition,
    PlacementPlan,
    ProfileSettings,
    RunTrace,
    WorkflowExecution,
)
from placement_compiler.profiling.runner import EvaluationAdapter, RuntimeAdapter
from placement_compiler.core.models import PlacementArtifact
from placement_compiler.adapters.repository_workflows import WORKFLOW_SPECS, WorkflowSpec, _workflow_import_context


def build_repository_profile_adapters(
    *,
    workflow: str,
    candidate_artifact: PlacementArtifact,
    endpoint_registry: EndpointRegistry,
) -> tuple[EvaluationAdapter, RuntimeAdapter]:
    spec = WORKFLOW_SPECS.get(workflow)
    if spec is None:
        raise ValueError(f"unknown workflow {workflow!r}")

    evaluation = WorkflowEvaluationAdapter(spec)
    runtime = LangGraphWorkflowRuntimeAdapter(spec, candidate_artifact, endpoint_registry)
    return evaluation, runtime


class WorkflowEvaluationAdapter:
    def __init__(self, spec: WorkflowSpec) -> None:
        self.spec = spec

    @property
    def primary_metric(self) -> MetricDefinition:
        with _workflow_import_context(self.spec.source_dir):
            module = importlib.import_module(self.spec.evaluation_module)
            return MetricDefinition.model_validate(module.PRIMARY_METRIC)

    def load_examples(self, profile: ProfileSettings) -> list[dict[str, Any]]:
        with _workflow_import_context(self.spec.source_dir):
            module = importlib.import_module(self.spec.evaluation_module)
            return list(module.load_examples(profile))

    def example_id(self, example: dict[str, Any]) -> str:
        with _workflow_import_context(self.spec.source_dir):
            module = importlib.import_module(self.spec.evaluation_module)
            return str(module.example_id(example))

    def score(self, example: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
        with _workflow_import_context(self.spec.source_dir):
            module = importlib.import_module(self.spec.evaluation_module)
            return dict(module.score(example, output))


class LangGraphWorkflowRuntimeAdapter:
    def __init__(
        self,
        spec: WorkflowSpec,
        candidate_artifact: PlacementArtifact,
        endpoint_registry: EndpointRegistry,
    ) -> None:
        self.spec = spec
        self.candidate_artifact = candidate_artifact
        self.endpoint_registry = endpoint_registry
        self.runtime: Any = None

    def prepare(self, examples: list[Any]) -> None:
        with _workflow_import_context(self.spec.source_dir):
            module = importlib.import_module(self.spec.evaluation_module)
            self.runtime = module.prepare_runtime(list(examples))

    def run_example(
        self,
        plan: PlacementPlan,
        example: dict[str, Any],
        repeat: int,
    ) -> WorkflowExecution:
        started = time.perf_counter()
        trace_collector = TraceCollector()
        resolver = ModelResolver(
            plan=plan,
            endpoint_registry=self.endpoint_registry,
            workflow=self.candidate_artifact.workflow,
            trace_collector=trace_collector,
        )

        with _workflow_import_context(self.spec.source_dir):
            graph_module = importlib.import_module(self.spec.graph_module)
            graph_factory = getattr(graph_module, self.spec.graph_factory)
            evaluation_module = importlib.import_module(self.spec.evaluation_module)
            graph = graph_factory(resolver, self.runtime)
            chunks = list(
                graph.stream(
                    evaluation_module.make_input(example),
                    config={"metadata": {"profile_candidate_id": plan.id}},
                )
            )
            output = dict(evaluation_module.extract_output(chunks))
            system_metrics = _system_metrics(evaluation_module, output)

        return WorkflowExecution(
            output=output,
            trace=RunTrace(
                invocations=trace_collector.invocations,
                system_metrics=system_metrics,
            ),
            latency_seconds=time.perf_counter() - started,
        )


def _system_metrics(evaluation_module: Any, output: dict[str, Any]) -> dict[str, Any]:
    if hasattr(evaluation_module, "system_metrics"):
        return dict(evaluation_module.system_metrics(output))
    return {}
