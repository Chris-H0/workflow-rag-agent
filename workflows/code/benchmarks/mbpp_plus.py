import json

from workflows.code.paths import WORKFLOW_ROOT

MBPP_PLUS_DATA_PATH = WORKFLOW_ROOT / "data" / "mbpp_plus_valid.jsonl"
MBPP_PLUS_MANIFEST_PATH = MBPP_PLUS_DATA_PATH.with_suffix(".manifest.json")
MBPP_PLUS_DOWNLOAD_COMMAND = "python -m workflows.code.download_data"
MBPP_PLUS_JSON_TYPE_KEY = "__mbpp_plus_type__"
REQUIRED_FIELDS = (
    "task_id",
    "prompt",
    "entry_point",
    "canonical_solution",
    "base_input",
    "plus_input",
    "atol",
)


def load_mbpp_plus_examples():
    if not MBPP_PLUS_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing local MBPP+ data: {MBPP_PLUS_DATA_PATH}. "
            f"Run `{MBPP_PLUS_DOWNLOAD_COMMAND}` from the repository root."
        )

    with MBPP_PLUS_DATA_PATH.open(encoding="utf-8") as handle:
        examples = [
            decode_json_value(json.loads(line))
            for line in handle
            if line.strip()
        ]

    return examples


def task_sort_key(task_id):
    return int(str(task_id).split("/")[-1])


def encode_json_value(value):
    if isinstance(value, complex):
        return {
            MBPP_PLUS_JSON_TYPE_KEY: "complex",
            "real": value.real,
            "imag": value.imag,
        }
    if isinstance(value, dict):
        return {
            key: encode_json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list | tuple):
        return [encode_json_value(item) for item in value]
    return value


def decode_json_value(value):
    if isinstance(value, dict):
        if value.get(MBPP_PLUS_JSON_TYPE_KEY) == "complex":
            return complex(value["real"], value["imag"])
        return {
            key: decode_json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [decode_json_value(item) for item in value]
    return value


def make_problem_prompt(example):
    return (
        "Solve this MBPP+ Python programming task.\n\n"
        f"Task ID: {example['task_id']}\n"
        f"Entry point: {example['entry_point']}\n\n"
        f"{example['prompt']}"
    )
