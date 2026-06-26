from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from placement_compiler.adapters.langgraph import extract_langgraph_metadata
from placement_compiler.core.models import NodeRegistryMetadata, WorkflowMetadata


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class WorkflowSpec:
    workflow_id: str
    source_dir: Path
    registry: dict[str, NodeRegistryMetadata]
    builder: str


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


QA_REGISTRY = {
    "generate_query_or_respond": NodeRegistryMetadata(
        is_llm_placement_unit=True,
        semantic_role="initial retrieval controller",
        description=(
            "Decides whether to answer directly or call the retriever for the first query."
        ),
        user_facing_output=True,
        tool_use=True,
        branch_control=True,
    ),
    "decide_after_retrieval": NodeRegistryMetadata(
        is_llm_placement_unit=True,
        semantic_role="retrieval sufficiency router",
        description="Routes after retrieval based on relevance, completeness, and budget.",
        structured_output_required=True,
        branch_control=True,
    ),
    "rewrite_question": NodeRegistryMetadata(
        is_llm_placement_unit=True,
        semantic_role="question rewrite",
        description="Rewrites the original question before another retrieval attempt.",
    ),
    "generate_followup_query": NodeRegistryMetadata(
        semantic_role="follow-up query prompt builder",
        description="Builds a follow-up retrieval prompt without invoking a model.",
    ),
    "generate_answer": NodeRegistryMetadata(
        is_llm_placement_unit=True,
        semantic_role="final answer generation",
        description="Generates the final answer from the question and retrieved context.",
        user_facing_output=True,
    ),
    "retrieve": NodeRegistryMetadata(
        semantic_role="retrieval tool",
        description="Tool node that calls the configured document retriever.",
    ),
}


CODE_REGISTRY = {
    "understand_and_plan": NodeRegistryMetadata(
        is_llm_placement_unit=True,
        semantic_role="task understanding and planning",
        description="Creates a plan for solving the programming task.",
    ),
    "implement_solution": NodeRegistryMetadata(
        is_llm_placement_unit=True,
        semantic_role="solution implementation",
        description="Generates or revises the Python solution.",
    ),
    "generate_tests": NodeRegistryMetadata(
        is_llm_placement_unit=True,
        semantic_role="test generation",
        description="Generates tests for the current solution.",
    ),
    "run_tests": NodeRegistryMetadata(
        semantic_role="test execution",
        description="Executes generated tests in the local sandbox.",
    ),
    "review_solution": NodeRegistryMetadata(
        is_llm_placement_unit=True,
        semantic_role="solution review and routing",
        description="Reviews test results and decides whether to revise or finish.",
        structured_output_required=True,
        branch_control=True,
        user_facing_output=True,
    ),
}


WORKFLOW_SPECS = {
    "qa": WorkflowSpec(
        workflow_id="qa-workflow",
        source_dir=REPO_ROOT / "QA-WORKFLOW",
        registry=QA_REGISTRY,
        builder="qa",
    ),
    "qa-workflow": WorkflowSpec(
        workflow_id="qa-workflow",
        source_dir=REPO_ROOT / "QA-WORKFLOW",
        registry=QA_REGISTRY,
        builder="qa",
    ),
    "code": WorkflowSpec(
        workflow_id="code-workflow",
        source_dir=REPO_ROOT / "CODE-WORKFLOW",
        registry=CODE_REGISTRY,
        builder="code",
    ),
    "code-workflow": WorkflowSpec(
        workflow_id="code-workflow",
        source_dir=REPO_ROOT / "CODE-WORKFLOW",
        registry=CODE_REGISTRY,
        builder="code",
    ),
}


def available_workflows() -> list[str]:
    return sorted({"qa", "code"})


def load_existing_workflow_metadata(
    workflow: str,
    *,
    registry_overrides: dict[str, NodeRegistryMetadata] | None = None,
) -> WorkflowMetadata:
    spec = _get_workflow_spec(workflow)
    registry = dict(spec.registry)
    if registry_overrides:
        registry.update(registry_overrides)

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
        graph_module = importlib.import_module("agent.graph")
        if spec.builder == "qa":
            return graph_module.build_graph(_NoopModelRouter(), _build_placeholder_retriever())
        if spec.builder == "code":
            return graph_module.build_graph(_NoopModelRouter())
    raise ValueError(f"unsupported workflow builder {spec.builder!r}")


def _build_placeholder_retriever() -> Any:
    from langchain.tools import tool

    @tool
    def retrieve(query: str) -> str:
        """Placeholder retriever used only for compile-time metadata extraction."""

        return ""

    return retrieve


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
        "paths",
        "rag",
    )
    for module_name in list(sys.modules):
        if module_name in prefixes or module_name.startswith(
            tuple(f"{prefix}." for prefix in prefixes)
        ):
            sys.modules.pop(module_name, None)
