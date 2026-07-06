"""Load repository workflows and workflow-owned placement metadata manifests."""

from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from placement_compiler.adapters.langgraph import extract_langgraph_metadata
from placement_compiler.core.catalogue import load_node_registry
from placement_compiler.core.models import WorkflowMetadata


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class WorkflowSpec:
    workflow_id: str
    source_dir: Path
    metadata_path: Path
    graph_module: str = "agent.graph"
    graph_factory: str = "build_graph"
    evaluation_module: str = "evaluation"


class _NoopModel:
    def bind_tools(self, tools: list[Any]) -> "_NoopModel":
        return self

    def with_structured_output(self, schema: Any) -> "_NoopModel":
        return self

    def invoke(self, messages: list[Any]) -> None:
        return None


class _NoopModelRouter:
    def get_model(self, node_name: str) -> _NoopModel:
        return _NoopModel()


WORKFLOW_SPECS = {
    "qa": WorkflowSpec(
        workflow_id="qa-workflow",
        source_dir=REPO_ROOT / "QA-WORKFLOW",
        metadata_path=REPO_ROOT / "QA-WORKFLOW" / "placement.yaml",
    ),
    "qa-workflow": WorkflowSpec(
        workflow_id="qa-workflow",
        source_dir=REPO_ROOT / "QA-WORKFLOW",
        metadata_path=REPO_ROOT / "QA-WORKFLOW" / "placement.yaml",
    ),
    "code": WorkflowSpec(
        workflow_id="code-workflow",
        source_dir=REPO_ROOT / "CODE-WORKFLOW",
        metadata_path=REPO_ROOT / "CODE-WORKFLOW" / "placement.yaml",
    ),
    "code-workflow": WorkflowSpec(
        workflow_id="code-workflow",
        source_dir=REPO_ROOT / "CODE-WORKFLOW",
        metadata_path=REPO_ROOT / "CODE-WORKFLOW" / "placement.yaml",
    ),
}


def available_workflows() -> list[str]:
    return sorted({"qa", "code"})


def load_existing_workflow_metadata(workflow: str) -> WorkflowMetadata:
    spec = _get_workflow_spec(workflow)
    registry = load_node_registry(spec.metadata_path)

    compiled = _build_existing_workflow(spec)
    return extract_langgraph_metadata(
        compiled,
        workflow_id=spec.workflow_id,
        registry=registry,
    )


def _get_workflow_spec(workflow: str) -> WorkflowSpec:
    try:
        return WORKFLOW_SPECS[workflow]
    except KeyError as exc:
        raise ValueError(
            f"unknown workflow {workflow!r}; expected one of {available_workflows()}"
        ) from exc


def _build_existing_workflow(spec: WorkflowSpec) -> Any:
    with _workflow_import_context(spec.source_dir):
        graph_module = importlib.import_module(spec.graph_module)
        graph_factory = getattr(graph_module, spec.graph_factory)
        return graph_factory(_NoopModelRouter(), None)


@contextmanager
def _workflow_import_context(source_dir: Path):
    _clear_workflow_modules()
    sys.path.insert(0, str(source_dir))
    try:
        yield
    finally:
        try:
            sys.path.remove(str(source_dir))
        except ValueError:
            pass


def _clear_workflow_modules() -> None:
    prefixes = (
        "agent",
        "analysis",
        "benchmarks",
        "evals",
        "execution",
        "evaluation",
        "paths",
        "rag",
    )
    for module_name in list(sys.modules):
        if module_name in prefixes or module_name.startswith(
            tuple(f"{prefix}." for prefix in prefixes)
        ):
            sys.modules.pop(module_name, None)
