from __future__ import annotations

from datetime import UTC, datetime
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json

from evalplus.data import get_mbpp_plus

from workflows.code.benchmarks.mbpp_plus import (
    MBPP_PLUS_DATA_PATH,
    MBPP_PLUS_MANIFEST_PATH,
    REQUIRED_FIELDS,
    encode_json_value,
    task_sort_key,
)


def main() -> None:
    problems = sorted(
        get_mbpp_plus().values(),
        key=lambda problem: task_sort_key(problem["task_id"]),
    )
    valid_problems = []
    skipped_problems = []

    for problem in problems:
        missing_fields = [
            field
            for field in REQUIRED_FIELDS
            if field not in problem or problem[field] is None
        ]
        if missing_fields:
            skipped_problems.append(
                {
                    "task_id": problem.get("task_id"),
                    "missing_fields": missing_fields,
                }
            )
            continue
        valid_problems.append(dict(problem))

    write_jsonl(MBPP_PLUS_DATA_PATH, valid_problems)
    data_sha256 = sha256_file(MBPP_PLUS_DATA_PATH)
    manifest = {
        "dataset": "MBPP+",
        "source": "evalplus.data.get_mbpp_plus",
        "evalplus_version": package_version("evalplus"),
        "selection": "all tasks with required fields sorted by numeric task id",
        "required_fields": list(REQUIRED_FIELDS),
        "count": len(valid_problems),
        "generated_at": datetime.now(UTC).isoformat(),
        "data_file": str(MBPP_PLUS_DATA_PATH),
        "data_sha256": data_sha256,
        "task_ids": [problem["task_id"] for problem in valid_problems],
        "skipped_count": len(skipped_problems),
        "skipped_tasks": skipped_problems,
    }
    write_json(MBPP_PLUS_MANIFEST_PATH, manifest)

    print(f"Wrote {len(valid_problems)} MBPP+ tasks to {MBPP_PLUS_DATA_PATH}")
    print(f"Wrote manifest to {MBPP_PLUS_MANIFEST_PATH}")


def write_jsonl(path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(encode_json_value(row), ensure_ascii=False, sort_keys=True)
                + "\n"
            )


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


def package_version(package_name: str) -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


if __name__ == "__main__":
    main()
