"""Load repository workflows through one small driver interface."""

from __future__ import annotations

import importlib
import time
from pathlib import Path
from typing import Any, Protocol

from placement_compiler.adapters.langgraph import enrich_with_langgraph
from placement_compiler.core.catalogue import load_placement_manifest
from placement_compiler.core.models import WorkflowMetadata
from placement_compiler.profiling.models import (
    MetricDefinition,
    ProfileSettings,
    RunTrace,
    WorkflowResult,
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

    def prepare(self, examples: list[Any], runtime: Any | None = None) -> None: ...

    def run_example(
        self,
        model_resolver: Any,
        example: Any,
        *,
        metadata: dict[str, Any] | None = None,
        on_chunk: Any | None = None,
    ) -> WorkflowResult: ...


class LangGraphWorkflowDriver:
    def __init__(self, workflow: str) -> None:
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
        self._runtime: Any = None
        self._metadata: WorkflowMetadata | None = None

    @property
    def primary_metric(self) -> MetricDefinition:
        return MetricDefinition.model_validate(self._evaluation.PRIMARY_METRIC)

    def metadata(self) -> WorkflowMetadata:
        if self._metadata is None:
            placement_units = load_placement_manifest(self.root / "placement.yaml")
            compiled = self._graph.build_graph(_NoopModelResolver(), None)
            self._metadata = enrich_with_langgraph(
                compiled,
                workflow_id=self.workflow_id,
                placement_units=placement_units,
            )
        return self._metadata

    def load_examples(self, profile: ProfileSettings) -> list[Any]:
        return list(self._evaluation.load_examples(profile))

    def example_id(self, example: Any) -> str:
        return str(self._evaluation.example_id(example))

    def prepare(self, examples: list[Any], runtime: Any | None = None) -> None:
        self._runtime = (
            runtime
            if runtime is not None
            else self._evaluation.prepare_runtime(list(examples))
        )

    def run_example(
        self,
        model_resolver: Any,
        example: Any,
        *,
        metadata: dict[str, Any] | None = None,
        on_chunk: Any | None = None,
    ) -> WorkflowResult:
        started = time.perf_counter()
        graph = self._graph.build_graph(model_resolver, self._runtime)
        chunks = []
        for chunk in graph.stream(
            self._evaluation.make_input(example),
            config={"metadata": metadata or {}},
        ):
            chunks.append(chunk)
            if on_chunk is not None:
                on_chunk(chunk)
        output = dict(self._evaluation.extract_output(chunks))
        system_metrics = (
            dict(self._evaluation.system_metrics(output))
            if hasattr(self._evaluation, "system_metrics")
            else {}
        )
        trace_collector = getattr(model_resolver, "trace_collector", None)
        return WorkflowResult(
            output=output,
            metrics=dict(self._evaluation.score(example, output)),
            trace=RunTrace(
                invocations=(trace_collector.invocations if trace_collector else []),
                system_metrics=system_metrics,
            ),
            latency_seconds=time.perf_counter() - started,
        )


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


def load_workflow_driver(workflow: str) -> WorkflowDriver:
    return LangGraphWorkflowDriver(workflow)
