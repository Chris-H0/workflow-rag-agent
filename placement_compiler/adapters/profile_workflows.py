from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

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
from placement_compiler.adapters.repository_workflows import WORKFLOW_SPECS, _workflow_import_context


def build_repository_profile_adapters(
    *,
    workflow: str,
    candidate_artifact: PlacementArtifact,
    endpoint_registry: EndpointRegistry,
) -> tuple[EvaluationAdapter, RuntimeAdapter]:
    if workflow in {"qa", "qa-workflow"}:
        evaluation = QAEvaluationAdapter()
        runtime = QALangGraphRuntimeAdapter(candidate_artifact, endpoint_registry)
        return evaluation, runtime
    if workflow in {"code", "code-workflow"}:
        evaluation = CodeEvaluationAdapter()
        runtime = CodeLangGraphRuntimeAdapter(candidate_artifact, endpoint_registry)
        return evaluation, runtime
    raise ValueError(f"unknown workflow {workflow!r}")


class SimpleRetriever:
    def __init__(self, documents: list[Any], k: int = 4) -> None:
        self.documents = documents
        self.k = k

    def invoke(self, query: str) -> list[Any]:
        query_tokens = _tokens(query)
        scored = []
        for index, document in enumerate(self.documents):
            text = f"{document.metadata.get('title', '')} {document.page_content}"
            overlap = len(query_tokens.intersection(_tokens(text)))
            scored.append((overlap, -index, document))
        scored.sort(reverse=True)
        return [document for score, _, document in scored[: self.k] if score > 0] or [
            item[2] for item in scored[: self.k]
        ]


class QAEvaluationAdapter:
    @property
    def primary_metric(self) -> MetricDefinition:
        return MetricDefinition(name="exact_match", direction="maximise")

    def load_examples(self, profile: ProfileSettings) -> list[dict[str, Any]]:
        if profile.examples is not None:
            return profile.examples

        spec = WORKFLOW_SPECS["qa"]
        with _workflow_import_context(spec.source_dir):
            from rag.sources import load_hotpotqa_examples

            options = profile.dataset_options
            level = str(options.get("level", "hard"))
            load_limit = int(options.get("load_limit", max(profile.sample_size * 5, profile.sample_size)))
            return list(load_hotpotqa_examples(load_limit, level))

    def example_id(self, example: dict[str, Any]) -> str:
        return str(example["id"])

    def score(self, example: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
        spec = WORKFLOW_SPECS["qa"]
        with _workflow_import_context(spec.source_dir):
            from evals.metrics import get_supporting_titles, score_result

            return score_result(
                str(output.get("agent_answer", "")),
                str(example["answer"]),
                output.get("retrieved_titles", []),
                get_supporting_titles(example),
            )


class QALangGraphRuntimeAdapter:
    def __init__(
        self,
        candidate_artifact: PlacementArtifact,
        endpoint_registry: EndpointRegistry,
    ) -> None:
        self.candidate_artifact = candidate_artifact
        self.endpoint_registry = endpoint_registry
        self.examples: list[dict[str, Any]] = []

    def prepare(self, examples: list[Any]) -> None:
        self.examples = list(examples)

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

        spec = WORKFLOW_SPECS["qa"]
        with _workflow_import_context(spec.source_dir):
            from agent.graph import build_graph
            from agent.tools import build_retriever_tool
            from rag.sources import hotpotqa_example_to_documents

            documents = []
            for loaded_example in self.examples:
                documents.extend(hotpotqa_example_to_documents(loaded_example))
            retriever_tool = build_retriever_tool(SimpleRetriever(documents))
            graph = build_graph(resolver, retriever_tool)

            nodes: list[str] = []
            agent_answer = ""
            retrieved_titles: set[str] = set()
            user_message = HumanMessage(content=str(example["question"]))
            for chunk in graph.stream(
                {"messages": [user_message]},
                config={"metadata": {"profile_candidate_id": plan.id}},
            ):
                for node, update in chunk.items():
                    nodes.append(node)
                    latest_message = update["messages"][-1]
                    if node == "retrieve":
                        retrieved_titles.update(_retrieved_titles(update["messages"]))
                    if isinstance(latest_message, AIMessage) and latest_message.content:
                        agent_answer = str(latest_message.content)

        return WorkflowExecution(
            output={
                "agent_answer": agent_answer,
                "retrieved_titles": sorted(retrieved_titles),
                "nodes": nodes,
                "retrieval_rounds": nodes.count("retrieve"),
            },
            trace=RunTrace(
                invocations=trace_collector.invocations,
                system_metrics={"retrieval_rounds": nodes.count("retrieve")},
            ),
            latency_seconds=time.perf_counter() - started,
        )


class CodeEvaluationAdapter:
    @property
    def primary_metric(self) -> MetricDefinition:
        return MetricDefinition(name="mbpp_plus_pass", direction="maximise")

    def load_examples(self, profile: ProfileSettings) -> list[dict[str, Any]]:
        if profile.examples is not None:
            return profile.examples

        spec = WORKFLOW_SPECS["code"]
        with _workflow_import_context(spec.source_dir):
            from benchmarks.mbpp_plus import load_mbpp_plus_examples

            options = profile.dataset_options
            task_ids = options.get("task_ids")
            load_limit = int(options.get("load_limit", max(profile.sample_size * 5, profile.sample_size)))
            return list(load_mbpp_plus_examples(load_limit, task_ids=task_ids))

    def example_id(self, example: dict[str, Any]) -> str:
        return str(example["task_id"])

    def score(self, example: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
        spec = WORKFLOW_SPECS["code"]
        with _workflow_import_context(spec.source_dir):
            from evals.metrics import score_solution

            return score_solution(example, str(output.get("solution", "")))


class CodeLangGraphRuntimeAdapter:
    def __init__(
        self,
        candidate_artifact: PlacementArtifact,
        endpoint_registry: EndpointRegistry,
    ) -> None:
        self.candidate_artifact = candidate_artifact
        self.endpoint_registry = endpoint_registry

    def prepare(self, examples: list[Any]) -> None:
        return None

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

        spec = WORKFLOW_SPECS["code"]
        with _workflow_import_context(spec.source_dir):
            from agent.graph import build_graph
            from benchmarks.mbpp_plus import make_problem_prompt

            graph = build_graph(resolver)
            nodes: list[str] = []
            plan_text = ""
            solution = ""
            generated_tests = ""
            generated_test_result = {}
            review_comments = ""
            review_decision = ""
            revision_count = 0
            user_message = HumanMessage(content=make_problem_prompt(example))
            for chunk in graph.stream(
                {
                    "messages": [user_message],
                    "task": example,
                    "plan": "",
                    "solution": "",
                    "generated_tests": "",
                    "generated_test_result": {},
                    "review_comments": "",
                    "review_decision": "",
                    "revision_count": 0,
                },
                config={"metadata": {"profile_candidate_id": plan.id}},
            ):
                for node, update in chunk.items():
                    nodes.append(node)
                    plan_text = update.get("plan", plan_text)
                    solution = update.get("solution", solution)
                    generated_tests = update.get("generated_tests", generated_tests)
                    generated_test_result = update.get(
                        "generated_test_result",
                        generated_test_result,
                    )
                    review_comments = update.get("review_comments", review_comments)
                    review_decision = update.get("review_decision", review_decision)
                    revision_count = update.get("revision_count", revision_count)

        return WorkflowExecution(
            output={
                "nodes": nodes,
                "plan": plan_text,
                "solution": solution,
                "generated_tests": generated_tests,
                "generated_test_result": generated_test_result,
                "review_comments": review_comments,
                "review_decision": review_decision,
                "revision_count": revision_count,
            },
            trace=RunTrace(
                invocations=trace_collector.invocations,
                system_metrics={"revision_count": revision_count},
            ),
            latency_seconds=time.perf_counter() - started,
        )


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in str(text).split() if token.strip()}


def _retrieved_titles(messages: list[Any]) -> set[str]:
    titles = set()
    for message in messages:
        for line in str(getattr(message, "content", "")).splitlines():
            if line.startswith("Title: "):
                titles.add(line.removeprefix("Title: ").strip())
    return titles
