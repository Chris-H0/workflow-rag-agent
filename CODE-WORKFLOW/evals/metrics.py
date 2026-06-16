from execution.sandbox import PASS, run_code_tests


def score_solution(problem, solution: str):
    base_result = run_code_tests(problem, solution, "base")
    plus_result = run_code_tests(problem, solution, "plus")
    base_pass = base_result["status"] == PASS
    plus_pass = plus_result["status"] == PASS

    return {
        "base_pass": base_pass,
        "plus_pass": plus_pass,
        "mbpp_plus_pass": base_pass and plus_pass,
        "accuracy": 1.0 if base_pass and plus_pass else 0.0,
        "base_status": base_result["status"],
        "plus_status": plus_result["status"],
        "base_failed_count": base_result.get("failed_count", 0),
        "plus_failed_count": plus_result.get("failed_count", 0),
        "base_total_count": base_result.get("total_count", 0),
        "plus_total_count": plus_result.get("total_count", 0),
        "base_error": base_result.get("error"),
        "plus_error": plus_result.get("error"),
    }
