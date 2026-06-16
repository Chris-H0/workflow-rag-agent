import json
from typing import Literal

from langchain_core.messages import AIMessage, SystemMessage

from agent.prompts import DEBUG_SOLUTION_PROMPT, GENERATE_SOLUTION_PROMPT
from benchmarks.mbpp_plus import make_problem_prompt
from execution.parsing import build_executable_solution, extract_code
from execution.sandbox import PASS, run_code_tests


MAX_DEBUG_ATTEMPTS = 2


def build_generate_solution(response_model):
    def generate_solution(state):
        task_prompt = make_problem_prompt(state["task"])
        messages = [
            SystemMessage(content=GENERATE_SOLUTION_PROMPT),
            *state["messages"],
        ]
        response = response_model.invoke(messages)
        code = extract_code(response.content)
        solution = build_executable_solution(state["task"], code)
        return {
            "messages": [response],
            "solution": solution,
            "attempts": 0,
        }

    return generate_solution


def run_visible_tests(state):
    test_result = run_code_tests(state["task"], state["solution"], "base")
    message = AIMessage(
        content=(
            "Visible tests passed."
            if test_result["status"] == PASS
            else f"Visible tests failed: {json.dumps(test_result, default=str)}"
        )
    )
    return {"messages": [message], "test_result": test_result}


def decide_after_tests(state) -> Literal["debug_solution", "__end__"]:
    if state["test_result"]["status"] == PASS:
        return "__end__"
    if state.get("attempts", 0) >= MAX_DEBUG_ATTEMPTS:
        return "__end__"
    return "debug_solution"


def build_debug_solution(response_model):
    def debug_solution(state):
        task_prompt = make_problem_prompt(state["task"])
        prompt = DEBUG_SOLUTION_PROMPT.format(
            task_prompt=task_prompt,
            solution=state["solution"],
            test_result=json.dumps(state["test_result"], indent=2, default=str),
        )
        response = response_model.invoke([{"role": "user", "content": prompt}])
        code = extract_code(response.content)
        solution = build_executable_solution(state["task"], code)
        return {
            "messages": [response],
            "solution": solution,
            "attempts": state.get("attempts", 0) + 1,
        }

    return debug_solution
