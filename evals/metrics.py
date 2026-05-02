import re
import string


def normalize_answer(text: str):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def get_supporting_titles(example):
    return set(example["supporting_facts"]["title"])


def get_retrieved_titles(content: str):
    titles = []

    for line in content.splitlines():
        if line.startswith("Title: "):
            titles.append(line.removeprefix("Title: ").strip())

    return set(titles)


def score_result(agent_answer: str, gold_answer: str, retrieved_titles, supporting_titles):
    normalized_agent_answer = normalize_answer(agent_answer)
    normalized_gold_answer = normalize_answer(gold_answer)
    matching_titles = set(retrieved_titles).intersection(supporting_titles)

    return {
        "exact_match": normalized_agent_answer == normalized_gold_answer,
        "contains_gold_answer": normalized_gold_answer in normalized_agent_answer,
        "supporting_title_hit": bool(matching_titles),
        "supporting_title_recall": len(matching_titles) / len(supporting_titles)
        if supporting_titles
        else 0,
    }
