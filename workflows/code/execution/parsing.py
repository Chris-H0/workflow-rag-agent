import re


CODE_BLOCK_PATTERN = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(text: str):
    match = CODE_BLOCK_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def build_executable_solution(problem, code: str):
    entry_point = problem["entry_point"]
    if f"def {entry_point}" in code:
        return code
    return f"{problem['prompt']}\n{code}"
