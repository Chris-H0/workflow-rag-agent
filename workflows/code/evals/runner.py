from datetime import UTC, datetime
import json
from pathlib import Path

from workflows.code.benchmarks.mbpp_plus import compact_example
from workflows.code.evals.traces import download_trace
from workflows.code.paths import analysis_config_dir


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
    driver,
    model_resolver,
    example,
    config_id: str,
    repeat: int,
    print_updates: bool,
    model_config: dict,
):
    example_id = example["task_id"]
    run_id = make_run_id(config_id, example_id, repeat)
    run = driver.run_example(
        model_resolver,
        example,
        metadata={"config_id": config_id, "run_id": run_id},
        on_chunk=_print_chunk if print_updates else None,
    )
    output = run.output
    result = {
        "config_id": config_id,
        "run_id": run_id,
        "example_id": example_id,
        "repeat": repeat,
        "benchmark": "MBPP+",
        "task": compact_example(example),
        "plan": output.get("plan", ""),
        "solution": output.get("solution", ""),
        "generated_tests": output.get("generated_tests", ""),
        "generated_test_result": output.get("generated_test_result", {}),
        "review_comments": output.get("review_comments", ""),
        "review_decision": output.get("review_decision", ""),
        "revision_count": output.get("revision_count", 0),
        "model_config": model_config,
        "metrics": run.metrics,
        "nodes": output.get("nodes", []),
    }

    output_path = analysis_config_dir(config_id) / "evals" / f"{run_id}.json"
    write_json(output_path, result)
    print(f"Saved eval result to {output_path}")

    return result


def _print_chunk(chunk):
    for node, update in chunk.items():
        messages = update.get("messages", [])
        print("### Node: ", node)
        if messages:
            messages[-1].pretty_print()
        print("\n")


def run_eval_loop(
    driver,
    model_resolver,
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
            result = run_eval_example(
                driver,
                model_resolver,
                example,
                config_id,
                repeat,
                print_updates,
                model_config,
            )
            download_trace(config_id, result["run_id"])
            count += 1
    print(f"Completed {count} evals for config ID: {config_id}.")
