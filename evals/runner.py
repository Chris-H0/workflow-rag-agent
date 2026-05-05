import json
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage

from evals.metrics import get_retrieved_titles, get_supporting_titles, score_result


def make_run_id(config_id: str, example_id: str, repeat: int):
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{config_id}-{timestamp}-{example_id}-r{repeat}"


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def model_kwargs_by_node(model_config):
    return {
        node_name: {
            key: value
            for key, value in node_config.items()
            if key not in {"provider", "model"}
        }
        for node_name, node_config in model_config.items()
        if isinstance(node_config, dict)
    }


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
    user_message = HumanMessage(content=question)
    nodes = []
    agent_answer = ""
    retrieved_titles = set()

    if print_updates:
        user_message.pretty_print()
        print("\n")

    for chunk in graph.stream(
        {"messages": [user_message]},
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

            if node == "retrieve":
                for message in update["messages"]:
                    retrieved_titles.update(get_retrieved_titles(message.content))

            if isinstance(latest_message, AIMessage) and latest_message.content:
                agent_answer = latest_message.content

            if print_updates:
                print("### Node: ", node)
                if node == "retrieve":
                    for message in update["messages"]:
                        message.pretty_print()
                else:
                    latest_message.pretty_print()
                print("\n")

    retrieved_titles = sorted(retrieved_titles)
    supporting_titles = sorted(supporting_titles)

    result = {
        "config_id": config_id,
        "run_id": run_id,
        "example_id": example_id,
        "repeat": repeat,
        "question": question,
        "gold_answer": gold_answer,
        "agent_answer": agent_answer,
        "supporting_titles": supporting_titles,
        "retrieved_titles": retrieved_titles,
        "retrieval_rounds": nodes.count("retrieve"),
        "model_config": model_config,
        "model_kwargs": model_kwargs_by_node(model_config),
        "metrics": score_result(
            agent_answer,
            gold_answer,
            retrieved_titles,
            supporting_titles,
        ),
        "nodes": nodes,
    }

    output_path = Path("analysis") / config_id / "evals" / f"{run_id}.json"
    write_json(output_path, result)
    print(f"Saved eval result to {output_path}")

    return result


def run_eval_loop(
    graph,
    examples,
    config_id: str,
    repeats: int,
    print_updates: bool,
    model_config: dict,
):
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
