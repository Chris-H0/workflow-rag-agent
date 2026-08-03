import json

from langchain_core.documents import Document

from workflows.qa.paths import WORKFLOW_ROOT


HOTPOTQA_DATA_PATH = (
    WORKFLOW_ROOT / "data" / "hotpotqa_validation_distractor_first500.jsonl"
)
HOTPOTQA_MANIFEST_PATH = HOTPOTQA_DATA_PATH.with_suffix(".manifest.json")
HOTPOTQA_DOWNLOAD_COMMAND = "python -m workflows.qa.download_data"


def load_hotpotqa_examples():
    if not HOTPOTQA_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing local HotpotQA data: {HOTPOTQA_DATA_PATH}. "
            f"Run `{HOTPOTQA_DOWNLOAD_COMMAND}` from the repository root."
        )

    with HOTPOTQA_DATA_PATH.open(encoding="utf-8") as handle:
        return [
            json.loads(line)
            for line in handle
            if line.strip()
        ]


def hotpotqa_example_to_documents(example):
    documents = []

    for title, sentences in zip(
        example["context"]["title"],
        example["context"]["sentences"],
    ):
        documents.append(
            Document(
                page_content=f"{title}\n" + "\n".join(sentences),
                metadata={
                    "dataset": "hotpotqa",
                    "subset": "distractor",
                    "split": "validation",
                    "hotpot_id": example["id"],
                    "title": title,
                },
            )
        )

    return documents


def build_hotpotqa_documents(examples):
    documents = []
    seen = set()

    for example in examples:
        for document in hotpotqa_example_to_documents(example):
            key = (document.metadata["title"], document.page_content)
            if key in seen:
                continue

            seen.add(key)
            documents.append(document)

    return documents
