from collections import Counter
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


def score_token_overlap(normalized_agent_answer: str, normalized_gold_answer: str):
    agent_tokens = normalized_agent_answer.split()
    gold_tokens = normalized_gold_answer.split()

    if not agent_tokens or not gold_tokens:
        return {
            "answer_token_precision": 0.0,
            "answer_token_recall": 0.0,
            "answer_token_f1": 0.0,
        }

    shared_count = sum((Counter(agent_tokens) & Counter(gold_tokens)).values())
    if shared_count == 0:
        return {
            "answer_token_precision": 0.0,
            "answer_token_recall": 0.0,
            "answer_token_f1": 0.0,
        }

    precision = shared_count / len(agent_tokens)
    recall = shared_count / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)

    return {
        "answer_token_precision": precision,
        "answer_token_recall": recall,
        "answer_token_f1": f1,
    }


def score_result(agent_answer: str, gold_answer: str, retrieved_titles, supporting_titles):
    normalized_agent_answer = normalize_answer(agent_answer)
    normalized_gold_answer = normalize_answer(gold_answer)
    matching_titles = set(retrieved_titles).intersection(supporting_titles)
    contains_gold_answer = normalized_gold_answer in normalized_agent_answer
    token_scores = score_token_overlap(normalized_agent_answer, normalized_gold_answer)

    return {
        "exact_match": normalized_agent_answer == normalized_gold_answer,
        "contains_gold_answer": contains_gold_answer,
        "contains_partial_gold_answer": contains_gold_answer
        or normalized_agent_answer in normalized_gold_answer,
        **token_scores,
        "supporting_title_hit": bool(matching_titles),
        "supporting_title_recall": len(matching_titles) / len(supporting_titles)
        if supporting_titles
        else 0,
    }
