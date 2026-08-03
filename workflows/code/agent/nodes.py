import json
import re
from typing import Literal

from langchain_core.messages import AIMessage, SystemMessage

from workflows.code.agent.prompts import (
    GENERATE_TESTS_PROMPT,
    IMPLEMENT_PROMPT,
    PLAN_PROMPT,
    REVIEW_PROMPT,
)
from workflows.code.benchmarks.mbpp_plus import make_problem_prompt
from workflows.code.execution.parsing import build_executable_solution, extract_code
from workflows.code.execution.sandbox import run_generated_tests


MAX_REVISIONS = 2


def parse_review_response(content: str):
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        return {
            "decision": "revise",
            "comments": "Review response was not valid JSON. Re-check the solution and return only code.",
        }

    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {
            "decision": "revise",
            "comments": "Review response was not valid JSON. Re-check the solution and return only code.",
        }

    decision = data.get("decision")
    if decision not in {"final", "revise"}:
        decision = "revise"
    return {
        "decision": decision,
        "comments": str(data.get("comments") or ""),
    }


def build_understand_and_plan(response_model):
    def understand_and_plan(state):
        messages = [
            SystemMessage(content=PLAN_PROMPT),
            *state["messages"],
        ]
        response = response_model.invoke(messages)
        return {"messages": [response], "plan": response.content}

    return understand_and_plan


def build_implement_solution(response_model):
    def implement_solution(state):
        task_prompt = make_problem_prompt(state["task"])
        revision_count = state.get("revision_count", 0)
        if state.get("review_decision") == "revise":
            revision_count += 1
        prompt = IMPLEMENT_PROMPT.format(
            task_prompt=task_prompt,
            plan=state.get("plan", ""),
            solution=state.get("solution") or "No current solution.",
            review_comments=state.get("review_comments") or "No review comments yet.",
        )
        response = response_model.invoke([{"role": "user", "content": prompt}])
        code = extract_code(response.content)
        solution = build_executable_solution(state["task"], code)
        return {
            "messages": [response],
            "solution": solution,
            "revision_count": revision_count,
        }

    return implement_solution


def build_generate_tests(response_model):
    def generate_tests(state):
        task_prompt = make_problem_prompt(state["task"])
        prompt = GENERATE_TESTS_PROMPT.format(
            task_prompt=task_prompt,
            plan=state.get("plan", ""),
            solution=state.get("solution", ""),
        )
        response = response_model.invoke([{"role": "user", "content": prompt}])
        generated_tests = extract_code(response.content)
        return {"messages": [response], "generated_tests": generated_tests}

    return generate_tests


def run_tests(state):
    test_result = run_generated_tests(
        state.get("solution", ""),
        state.get("generated_tests", ""),
    )
    message = AIMessage(content=json.dumps(test_result, default=str))
    return {"messages": [message], "generated_test_result": test_result}


def build_review_solution(response_model):
    def review_solution(state):
        task_prompt = make_problem_prompt(state["task"])
        prompt = REVIEW_PROMPT.format(
            task_prompt=task_prompt,
            plan=state.get("plan", ""),
            solution=state.get("solution", ""),
            generated_tests=state.get("generated_tests", ""),
            generated_test_result=json.dumps(
                state.get("generated_test_result", {}),
                indent=2,
                default=str,
            ),
        )
        response = response_model.invoke([{"role": "user", "content": prompt}])
        review = parse_review_response(response.content)
        message = AIMessage(
            content=json.dumps(
                {
                    "decision": review["decision"],
                    "comments": review["comments"],
                }
            )
        )
        return {
            "messages": [message],
            "review_decision": review["decision"],
            "review_comments": review["comments"],
        }

    return review_solution


def decide_after_review(state) -> Literal["implement_solution", "__end__"]:
    if state.get("review_decision") != "revise":
        return "__end__"
    if state.get("revision_count", 0) >= MAX_REVISIONS:
        return "__end__"
    return "implement_solution"
