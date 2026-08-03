"""Load repository workflows through one small driver interface."""

from __future__ import annotations

import importlib
import time
from pathlib import Path
from typing import Any, Protocol

from placement_compiler.adapters.langgraph import extract_langgraph_metadata
from placement_compiler.core.catalogue import load_node_registry
from placement_compiler.core.models import PlacementArtifact, PlacementPlan, WorkflowMetadata
from placement_compiler.profiling.models import (
    MetricDefinition,
    ProfileSettings,
    RunTrace,
    WorkflowExecution,
)
from placement_compiler.runtime.endpoint_registry import (
    EndpointRegistry,
    ModelResolver,
    TraceCollector,
)


WORKFLOW_PACKAGES = {
    "qa": "workflows.qa",
    "qa-workflow": "workflows.qa",
    "code": "workflows.code",
    "code-workflow": "workflows.code",
}


class WorkflowDriver(Protocol):
    """Everything the compiler and profiler need from a workflow."""

    root: Path

    @property
    def primary_metric(self) -> MetricDefinition: ...

    def metadata(self) -> WorkflowMetadata: ...

    def load_examples(self, profile: ProfileSettings) -> list[Any]: ...

    def example_id(self, example: Any) -> str: ...

    def prepare(self, examples: list[Any]) -> None: ...

    def run_example(
        self,
        plan: PlacementPlan,
        example: Any,
    ) -> WorkflowExecution: ...

    def score(self, example: Any, output: dict[str, Any]) -> dict[str, Any]: ...


class LangGraphWorkflowDriver:
    def __init__(
        self,
        workflow: str,
        *,
        candidate_artifact: PlacementArtifact | None = None,
        endpoint_registry: EndpointRegistry | None = None,
    ) -> None:
        try:
            package_name = WORKFLOW_PACKAGES[workflow]
        except KeyError as exc:
            raise ValueError(
                f"unknown workflow {workflow!r}; expected one of {available_workflows()}"
            ) from exc

        package = importlib.import_module(package_name)
        self.workflow_id = package.WORKFLOW_ID
        self.root = Path(package.__file__).resolve().parent
        self._graph = importlib.import_module(f"{package_name}.agent.graph")
        self._evaluation = importlib.import_module(f"{package_name}.evaluation")
        self._candidate_artifact = candidate_artifact
        self._endpoint_registry = endpoint_registry
        self._runtime: Any = None
        self._metadata: WorkflowMetadata | None = None

    @property
    def primary_metric(self) -> MetricDefinition:
        return MetricDefinition.model_validate(self._evaluation.PRIMARY_METRIC)

    def metadata(self) -> WorkflowMetadata:
        if self._metadata is None:
            registry = load_node_registry(self.root / "placement.yaml")
            compiled = self._graph.build_graph(_NoopModelResolver(), None)
            self._metadata = extract_langgraph_metadata(
                compiled,
                workflow_id=self.workflow_id,
                registry=registry,
            )
        return self._metadata

    def load_examples(self, profile: ProfileSettings) -> list[Any]:
        return list(self._evaluation.load_examples(profile))

    def example_id(self, example: Any) -> str:
        return str(self._evaluation.example_id(example))

    def prepare(self, examples: list[Any]) -> None:
        self._runtime = self._evaluation.prepare_runtime(list(examples))

    def run_example(
        self,
        plan: PlacementPlan,
        example: Any,
    ) -> WorkflowExecution:
        if self._candidate_artifact is None or self._endpoint_registry is None:
            raise RuntimeError("profiling requires a candidate artifact and endpoint registry")

        started = time.perf_counter()
        trace_collector = TraceCollector()
        resolver = ModelResolver(
            plan=plan,
            endpoint_registry=self._endpoint_registry,
            workflow=self._candidate_artifact.workflow,
            trace_collector=trace_collector,
        )
        graph = self._graph.build_graph(resolver, self._runtime)
        chunks = list(
            graph.stream(
                self._evaluation.make_input(example),
                config={"metadata": {"profile_candidate_id": plan.id}},
            )
        )
        output = dict(self._evaluation.extract_output(chunks))
        system_metrics = (
            dict(self._evaluation.system_metrics(output))
            if hasattr(self._evaluation, "system_metrics")
            else {}
        )
        return WorkflowExecution(
            output=output,
            trace=RunTrace(
                invocations=trace_collector.invocations,
                system_metrics=system_metrics,
            ),
            latency_seconds=time.perf_counter() - started,
        )

    def score(self, example: Any, output: dict[str, Any]) -> dict[str, Any]:
        return dict(self._evaluation.score(example, output))


class _NoopModel:
    def bind_tools(self, tools: list[Any]) -> _NoopModel:
        return self

    def with_structured_output(self, schema: Any) -> _NoopModel:
        return self

    def invoke(self, messages: list[Any]) -> None:
        return None


class _NoopModelResolver:
    def get_model(self, node_name: str) -> _NoopModel:
        return _NoopModel()


def available_workflows() -> list[str]:
    return ["code", "qa"]


def load_workflow_driver(
    workflow: str,
    *,
    candidate_artifact: PlacementArtifact | None = None,
    endpoint_registry: EndpointRegistry | None = None,
) -> WorkflowDriver:
    return LangGraphWorkflowDriver(
        workflow,
        candidate_artifact=candidate_artifact,
        endpoint_registry=endpoint_registry,
    )
