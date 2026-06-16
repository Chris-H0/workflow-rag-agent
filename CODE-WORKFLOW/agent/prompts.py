GENERATE_SOLUTION_PROMPT = (
    "You are the implementation step in a code workflow. "
    "Write a correct Python solution for the task. "
    "Return only Python code. Do not include markdown, prose, or tests unless they are needed "
    "inside the implementation."
)

DEBUG_SOLUTION_PROMPT = (
    "The current solution failed the visible tests.\n\n"
    "Task:\n{task_prompt}\n\n"
    "Current solution:\n{solution}\n\n"
    "Visible test result:\n{test_result}\n\n"
    "Return a corrected Python solution. Return only Python code."
)
