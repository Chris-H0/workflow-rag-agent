try:
    from evalplus.data import get_mbpp_plus
except ImportError as error:
    get_mbpp_plus = None
    IMPORT_ERROR = error
else:
    IMPORT_ERROR = None


def load_mbpp_plus_examples(limit: int, task_ids=None):
    if get_mbpp_plus is None:
        raise RuntimeError(
            "EvalPlus is required for MBPP+. Install dependencies with `uv sync`."
        ) from IMPORT_ERROR

    problems = get_mbpp_plus()
    examples = sorted(problems.values(), key=lambda problem: task_sort_key(problem["task_id"]))

    if task_ids:
        task_ids = {normalize_task_id(task_id) for task_id in task_ids}
        examples = [
            example
            for example in examples
            if normalize_task_id(example["task_id"]) in task_ids
        ]

    return examples[:limit]


def task_sort_key(task_id):
    return int(str(task_id).split("/")[-1])


def normalize_task_id(task_id):
    return str(task_id).split("/")[-1]


def make_problem_prompt(example):
    tests = example.get("assertion") or ""
    return (
        "Solve this MBPP+ Python programming task.\n\n"
        f"Task ID: {example['task_id']}\n"
        f"Entry point: {example['entry_point']}\n\n"
        f"{example['prompt']}\n\n"
        "Visible tests:\n"
        f"{tests}\n\n"
        "Return only Python code."
    )


def compact_example(example):
    return {
        "task_id": example["task_id"],
        "entry_point": example["entry_point"],
        "prompt": example["prompt"],
        "assertion": example.get("assertion") or "",
    }
