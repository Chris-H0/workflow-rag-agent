from datetime import UTC, datetime
import json
from pathlib import Path
from time import sleep

from langchain_core.messages import AIMessage, HumanMessage

from benchmarks.mbpp_plus import compact_example, make_problem_prompt
from evals.metrics import score_solution
from evals.traces import download_traces
from paths import analysis_config_dir


def make_run_id(config_id: str, example_id: str, repeat: int):
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    clean_id = str(example_id).replace("/", "-")
    return f"{config_id}-{timestamp}-{clean_id}-r{repeat}"


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def write_problem_log(config_id: str, examples, eval_examples):
    evaluated_ids = {example["task_id"] for example in eval_examples}
    problems = [
        {
            "task_id": example["task_id"],
            "entry_point": example["entry_point"],
            "evaluated": example["task_id"] in evaluated_ids,
        }
        for example in examples
    ]

    output_path = analysis_config_dir(config_id) / "config" / "problems.json"
    write_json(
        output_path,
        {
            "config_id": config_id,
            "benchmark": "MBPP+",
            "loaded_count": len(examples),
            "evaluated_count": len(eval_examples),
            "problems": problems,
        },
    )
    print(f"Saved problem log to {output_path}")


def run_eval_example(
    graph,
    example,
    config_id: str,
    repeat: int,
    print_updates: bool,
    model_config: dict,
):
    example_id = example["task_id"]
    run_id = make_run_id(config_id, example_id, repeat)
    task_prompt = make_problem_prompt(example)
    user_message = HumanMessage(content=task_prompt)
    nodes = []
    plan = ""
    solution = ""
    generated_tests = ""
    generated_test_result = {}
    review_comments = ""
    review_decision = ""
    revision_count = 0

    if print_updates:
        user_message.pretty_print()
        print("\n")

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
        config={
            "metadata": {
                "config_id": config_id,
                "run_id": run_id,
            }
        },
    ):
        for node, update in chunk.items():
            nodes.append(node)
            latest_message = update["messages"][-1]
            plan = update.get("plan", plan)
            solution = update.get("solution", solution)
            generated_tests = update.get("generated_tests", generated_tests)
            generated_test_result = update.get(
                "generated_test_result",
                generated_test_result,
            )
            review_comments = update.get("review_comments", review_comments)
            review_decision = update.get("review_decision", review_decision)
            revision_count = update.get("revision_count", revision_count)

            if print_updates:
                print("### Node: ", node)
                if isinstance(latest_message, AIMessage):
                    latest_message.pretty_print()
                print("\n")

    metrics = score_solution(example, solution)
    result = {
        "config_id": config_id,
        "run_id": run_id,
        "example_id": example_id,
        "repeat": repeat,
        "benchmark": "MBPP+",
        "task": compact_example(example),
        "plan": plan,
        "solution": solution,
        "generated_tests": generated_tests,
        "generated_test_result": generated_test_result,
        "review_comments": review_comments,
        "review_decision": review_decision,
        "revision_count": revision_count,
        "model_config": model_config,
        "metrics": metrics,
        "nodes": nodes,
    }

    output_path = analysis_config_dir(config_id) / "evals" / f"{run_id}.json"
    write_json(output_path, result)
    print(f"Saved eval result to {output_path}")

    return result


def run_eval_loop(
    graph,
    examples,
    all_examples,
    config_id: str,
    repeats: int,
    print_updates: bool,
    model_config: dict,
):
    write_problem_log(config_id, all_examples, examples)

    count = 0
    for repeat in range(repeats):
        for example in examples:
            run_eval_example(
                graph,
                example,
                config_id,
                repeat,
                print_updates,
                model_config,
            )
            count += 1
    print(f"Completed {count} evals for config ID: {config_id}.")


def run_download_traces(config_id: str):
    sleep(5)
    download_traces(config_id)
