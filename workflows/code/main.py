from workflows.runner import main

ASSIGNMENTS = {
    "understand_and_plan": "gpt-5.5-cloud",
    "implement_solution": "qwen-local",
    "generate_tests": "qwen-local",
    "review_solution": "gpt-5.5-cloud",
}


if __name__ == "__main__":
    main("code", ASSIGNMENTS)
