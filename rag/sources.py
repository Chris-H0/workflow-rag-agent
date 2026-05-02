from datasets import load_dataset
from langchain_core.documents import Document


HOTPOTQA_LEVEL = "hard"
HOTPOTQA_LIMIT = 50


def load_hotpotqa_examples():
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    examples = []

    for example in dataset:
        if example["level"] != HOTPOTQA_LEVEL:
            continue

        examples.append(example)

        if len(examples) == HOTPOTQA_LIMIT:
            break

    return examples


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
