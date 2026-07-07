from datetime import UTC, datetime
import json
from pathlib import Path

import evaluation as workflow_evaluation
from evals.metrics import get_supporting_titles
from evals.traces import download_trace
from paths import analysis_config_dir


def make_run_id(config_id: str, example_id: str, repeat: int):
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{config_id}-{timestamp}-{example_id}-r{repeat}"


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def write_question_log(config_id: str, examples, eval_examples):
    evaluated_ids = {example["id"] for example in eval_examples}
    questions = [
        {
            "id": example["id"],
            "level": example["level"],
            "evaluated": example["id"] in evaluated_ids,
        }
        for example in examples
    ]

    output_path = analysis_config_dir(config_id) / "config" / "question.json"
    write_json(
        output_path,
        {
            "config_id": config_id,
            "selection": "first 500 validation examples in source order",
            "loaded_count": len(examples),
            "evaluated_count": len(eval_examples),
            "questions": questions,
        },
    )
    print(f"Saved question log to {output_path}")


def run_eval_example(
    graph,
    example,
    config_id: str,
    repeat: int,
    print_updates: bool,
    model_config: dict,
):
    example_id = example["id"]
    question = example["question"]
    gold_answer = example["answer"]
    supporting_titles = get_supporting_titles(example)
    run_id = make_run_id(config_id, example_id, repeat)
    graph_input = workflow_evaluation.make_input(example)
    chunks = []

    if print_updates:
        graph_input["messages"][0].pretty_print()
        print("\n")

    for chunk in graph.stream(
        graph_input,
        config={
            "metadata": {
                "config_id": config_id,
                "run_id": run_id,
            }
        },
    ):
        chunks.append(chunk)
        if print_updates:
            _print_chunk(chunk)

    output = workflow_evaluation.extract_output(chunks)
    retrieved_titles = output.get("retrieved_titles", [])
    supporting_titles = sorted(supporting_titles)

    result = {
        "config_id": config_id,
        "run_id": run_id,
        "example_id": example_id,
        "repeat": repeat,
        "question": question,
        "gold_answer": gold_answer,
        "agent_answer": output.get("agent_answer", ""),
        "supporting_titles": supporting_titles,
        "retrieved_titles": retrieved_titles,
        "retrieval_rounds": output.get("retrieval_rounds", 0),
        "model_config": model_config,
        "metrics": workflow_evaluation.score(example, output),
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
        if node == "retrieve":
            for message in messages:
                message.pretty_print()
        elif messages:
            messages[-1].pretty_print()
        print("\n")


def run_eval_loop(
    graph,
    examples,
    all_examples,
    config_id: str,
    repeats: int,
    print_updates: bool,
    model_config: dict,
):
    write_question_log(config_id, all_examples, examples)

    count = 0
    for repeat in range(repeats):
        for example in examples:
            result = run_eval_example(
                graph,
                example,
                config_id,
                repeat,
                print_updates,
                model_config,
            )
            download_trace(config_id, result["run_id"])
            count += 1
    print(f"Completed {count} evals for config ID: {config_id}.")
