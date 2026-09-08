from workflows.runner import main

ASSIGNMENTS = {
    "generate_query_or_respond": "qwen-local",
    "decide_after_retrieval": "gpt-4.1-mini-cloud",
    "rewrite_question": "gpt-4.1-mini-cloud",
    "generate_answer": "gpt-4.1-mini-cloud",
}


if __name__ == "__main__":
    main("qa", ASSIGNMENTS)
