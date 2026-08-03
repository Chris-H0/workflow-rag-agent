"""Evaluation and profiling hooks for the QA workflow."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from workflows.qa.agent.tools import build_retriever_tool
from workflows.qa.evals.metrics import get_retrieved_titles, get_supporting_titles, score_result
from workflows.qa.rag.sources import build_hotpotqa_documents, load_hotpotqa_examples


PRIMARY_METRIC = {"name": "exact_match", "direction": "maximise"}


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


def load_examples(profile) -> list[dict[str, Any]]:
    if profile.examples is not None:
        return profile.examples

    return list(load_hotpotqa_examples())


def example_id(example: dict[str, Any]) -> str:
    return str(example["id"])


def prepare_runtime(examples: list[dict[str, Any]]) -> dict[str, Any]:
    documents = build_hotpotqa_documents(examples)
    return {"retriever_tool": build_retriever_tool(SimpleRetriever(documents))}


def make_input(example: dict[str, Any]) -> dict[str, Any]:
    return {"messages": [HumanMessage(content=str(example["question"]))]}


def extract_output(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    nodes: list[str] = []
    agent_answer = ""
    retrieved_titles: set[str] = set()
    for chunk in chunks:
        for node, update in chunk.items():
            nodes.append(node)
            latest_message = update["messages"][-1]
            if node == "retrieve":
                for message in update["messages"]:
                    retrieved_titles.update(
                        get_retrieved_titles(str(getattr(message, "content", "")))
                    )
            if isinstance(latest_message, AIMessage) and latest_message.content:
                agent_answer = str(latest_message.content)

    return {
        "agent_answer": agent_answer,
        "retrieved_titles": sorted(retrieved_titles),
        "nodes": nodes,
        "retrieval_rounds": nodes.count("retrieve"),
    }


def score(example: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
    return score_result(
        str(output.get("agent_answer", "")),
        str(example["answer"]),
        output.get("retrieved_titles", []),
        get_supporting_titles(example),
    )


def system_metrics(output: dict[str, Any]) -> dict[str, Any]:
    return {"retrieval_rounds": output.get("retrieval_rounds", 0)}


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in str(text).split() if token.strip()}
