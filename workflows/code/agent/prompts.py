PLAN_PROMPT = (
    "You are the planning step in a coding workflow. "
    "Understand the task and produce a concise implementation plan. "
    "Identify edge cases and the expected behavior from the task prompt. "
    "Do not write code."
)

IMPLEMENT_PROMPT = (
    "You are the implementation step in a coding workflow.\n\n"
    "Task:\n{task_prompt}\n\n"
    "Plan:\n{plan}\n\n"
    "Current solution:\n{solution}\n\n"
    "Review comments:\n{review_comments}\n\n"
    "Write the best corrected Python solution. "
    "Return only Python code. Do not include markdown, prose, or tests unless they are needed "
    "inside the implementation."
)

GENERATE_TESTS_PROMPT = (
    "You are the test-writing step in a coding workflow.\n\n"
    "Task:\n{task_prompt}\n\n"
    "Plan:\n{plan}\n\n"
    "Solution:\n{solution}\n\n"
    "Write a small set of Python assert statements that exercise the solution. "
    "Focus on representative behavior and edge cases from the task prompt. "
    "Use only the public function entry point. "
    "Return only Python code. Do not include markdown or prose."
)

REVIEW_PROMPT = (
    "You are reviewing a Python solution before final submission.\n\n"
    "Task:\n{task_prompt}\n\n"
    "Plan:\n{plan}\n\n"
    "Solution:\n{solution}\n\n"
    "Generated tests:\n{generated_tests}\n\n"
    "Generated test result:\n{generated_test_result}\n\n"
    "Decide whether the solution is ready or should be revised. "
    "Use the generated test result as feedback, but also review for correctness, edge cases, "
    "signature compatibility, and avoid unnecessary rewrites.\n\n"
    "Return only valid JSON. Do not include markdown, prose, explanations, or extra keys.\n"
    'The JSON must match exactly this shape: {{"decision": "<final or revise>", "comments": "<brief comments>"}}\n'
    'Allowed values for "decision": "final", "revise".'
)
