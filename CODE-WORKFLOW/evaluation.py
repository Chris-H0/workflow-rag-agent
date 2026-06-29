"""Evaluation and profiling hooks for the code-generation workflow."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage

from benchmarks.mbpp_plus import load_mbpp_plus_examples, make_problem_prompt
from evals.metrics import score_solution


PRIMARY_METRIC = {"name": "mbpp_plus_pass", "direction": "maximise"}


def load_examples(profile) -> list[dict[str, Any]]:
    if profile.examples is not None:
        return profile.examples

    options = profile.dataset_options
    task_ids = options.get("task_ids")
    load_limit = int(options.get("load_limit", max(profile.sample_size * 5, profile.sample_size)))
    return list(load_mbpp_plus_examples(load_limit, task_ids=task_ids))


def example_id(example: dict[str, Any]) -> str:
    return str(example["task_id"])


def prepare_runtime(examples: list[dict[str, Any]]) -> None:
    return None


def make_input(example: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": [HumanMessage(content=make_problem_prompt(example))],
        "task": example,
        "plan": "",
        "solution": "",
        "generated_tests": "",
        "generated_test_result": {},
        "review_comments": "",
        "review_decision": "",
        "revision_count": 0,
    }


def extract_output(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    nodes: list[str] = []
    plan_text = ""
    solution = ""
    generated_tests = ""
    generated_test_result = {}
    review_comments = ""
    review_decision = ""
    revision_count = 0

    for chunk in chunks:
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

    return {
        "nodes": nodes,
        "plan": plan_text,
        "solution": solution,
        "generated_tests": generated_tests,
        "generated_test_result": generated_test_result,
        "review_comments": review_comments,
        "review_decision": review_decision,
        "revision_count": revision_count,
    }


def score(example: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
    return score_solution(example, str(output.get("solution", "")))


def system_metrics(output: dict[str, Any]) -> dict[str, Any]:
    return {"revision_count": output.get("revision_count", 0)}
