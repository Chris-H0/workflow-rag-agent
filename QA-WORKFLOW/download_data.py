from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
import hashlib
import json

from datasets import load_dataset

from rag.sources import HOTPOTQA_DATA_PATH, HOTPOTQA_MANIFEST_PATH


DATASET_NAME = "hotpotqa/hotpot_qa"
CONFIG_NAME = "distractor"
SPLIT = "validation"
EXAMPLE_COUNT = 500


def main() -> None:
    dataset = load_dataset(DATASET_NAME, CONFIG_NAME, split=SPLIT)
    examples = []
    for example in dataset:
        examples.append(dict(example))
        if len(examples) == EXAMPLE_COUNT:
            break

    if len(examples) != EXAMPLE_COUNT:
        raise ValueError(
            f"Expected {EXAMPLE_COUNT} HotpotQA examples, got {len(examples)}."
        )

    write_jsonl(HOTPOTQA_DATA_PATH, examples)
    data_sha256 = sha256_file(HOTPOTQA_DATA_PATH)

    questions = [
        {
            "id": example["id"],
            "difficulty": example.get("level"),
        }
        for example in examples
    ]
    manifest = {
        "dataset": DATASET_NAME,
        "config": CONFIG_NAME,
        "split": SPLIT,
        "selection": f"first {EXAMPLE_COUNT} validation examples in source order",
        "count": len(examples),
        "generated_at": datetime.now(UTC).isoformat(),
        "source_fingerprint": getattr(dataset, "_fingerprint", None),
        "data_file": str(HOTPOTQA_DATA_PATH),
        "data_sha256": data_sha256,
        "difficulty_counts": dict(Counter(item["difficulty"] for item in questions)),
        "question_ids": [item["id"] for item in questions],
        "questions": questions,
    }
    write_json(HOTPOTQA_MANIFEST_PATH, manifest)

    print(f"Wrote {len(examples)} HotpotQA examples to {HOTPOTQA_DATA_PATH}")
    print(f"Wrote manifest to {HOTPOTQA_MANIFEST_PATH}")


def write_jsonl(path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
