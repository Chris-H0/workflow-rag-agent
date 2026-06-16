import os

from evalplus.eval import FAIL, PASS, TIMEOUT, untrusted_check
from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
from evalplus.gen.util import trusted_exec


ERROR = "error"


def run_reference(problem, test_set: str):
    output_not_none = problem["entry_point"] in MBPP_OUTPUT_NOT_NONE_TASKS
    return trusted_exec(
        problem["prompt"] + problem["canonical_solution"],
        problem[f"{test_set}_input"],
        problem["entry_point"],
        record_time=True,
        output_not_none=output_not_none,
    )


def run_code_tests(problem, code: str, test_set: str = "base", timeout: float = 5.0):
    os.environ.setdefault("EVALPLUS_MAX_MEMORY_BYTES", "-1")
    expected, ref_time = run_reference(problem, test_set)
    status, details = untrusted_check(
        "mbpp",
        code,
        problem[f"{test_set}_input"],
        problem["entry_point"],
        expected=expected,
        atol=problem["atol"],
        ref_time=ref_time,
        fast_check=False,
        min_time_limit=timeout,
    )
    details = [bool(value) for value in details]
    failed_indices = [index for index, passed in enumerate(details) if not passed]

    return {
        "status": status,
        "passed": status == PASS,
        "test_set": test_set,
        "failures": [{"index": index} for index in failed_indices[:5]],
        "failed_count": len(failed_indices) if status == FAIL else 0,
        "total_count": len(details),
        "error": None if status in {PASS, FAIL, TIMEOUT} else status,
    }
