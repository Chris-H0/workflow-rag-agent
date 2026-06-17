import ast
import base64
import json
import os
import pickle
import subprocess
import sys
import textwrap
import traceback

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


def count_asserts(code: str):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0
    return sum(isinstance(node, ast.Assert) for node in ast.walk(tree))


def run_generated_tests(solution: str, generated_tests: str, timeout: float = 5.0):
    if not generated_tests.strip():
        return {
            "status": ERROR,
            "passed": False,
            "test_set": "generated",
            "failures": [],
            "failed_count": 0,
            "total_count": 0,
            "error": "No generated tests were provided.",
        }

    total_count = count_asserts(generated_tests)
    payload = {
        "solution": solution,
        "generated_tests": generated_tests,
        "total_count": total_count,
    }
    encoded_payload = base64.b64encode(pickle.dumps(payload)).decode("ascii")
    script = build_generated_test_script(encoded_payload)

    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {
            "status": TIMEOUT,
            "passed": False,
            "test_set": "generated",
            "failures": [],
            "failed_count": 0,
            "total_count": total_count,
            "error": f"Timed out after {timeout} seconds.",
        }

    if completed.returncode != 0:
        return {
            "status": ERROR,
            "passed": False,
            "test_set": "generated",
            "failures": [],
            "failed_count": 0,
            "total_count": total_count,
            "error": completed.stderr.strip() or completed.stdout.strip(),
        }

    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {
            "status": ERROR,
            "passed": False,
            "test_set": "generated",
            "failures": [],
            "failed_count": 0,
            "total_count": total_count,
            "error": completed.stdout.strip(),
        }

    result["test_set"] = "generated"
    return result


def build_generated_test_script(encoded_payload: str):
    return textwrap.dedent(
        f"""
        import base64
        import json
        import pickle
        import traceback

        payload = pickle.loads(base64.b64decode({encoded_payload!r}))
        namespace = {{}}

        try:
            exec(payload["solution"], namespace)
        except Exception:
            print(json.dumps({{
                "status": "error",
                "passed": False,
                "failures": [{{"phase": "solution", "error": traceback.format_exc()}}],
                "failed_count": 0,
                "total_count": payload["total_count"],
                "error": traceback.format_exc(),
            }}))
            raise SystemExit(0)

        try:
            compiled_tests = compile(payload["generated_tests"], "<generated_tests>", "exec")
        except Exception:
            print(json.dumps({{
                "status": "error",
                "passed": False,
                "failures": [{{"phase": "compile_tests", "error": traceback.format_exc()}}],
                "failed_count": 0,
                "total_count": payload["total_count"],
                "error": traceback.format_exc(),
            }}))
            raise SystemExit(0)

        try:
            exec(compiled_tests, namespace)
        except AssertionError:
            print(json.dumps({{
                "status": "fail",
                "passed": False,
                "failures": [{{"phase": "assertion", "error": traceback.format_exc()}}],
                "failed_count": 1,
                "total_count": payload["total_count"],
                "error": None,
            }}))
            raise SystemExit(0)
        except Exception:
            print(json.dumps({{
                "status": "error",
                "passed": False,
                "failures": [{{"phase": "run_tests", "error": traceback.format_exc()}}],
                "failed_count": 0,
                "total_count": payload["total_count"],
                "error": traceback.format_exc(),
            }}))
            raise SystemExit(0)

        print(json.dumps({{
            "status": "pass",
            "passed": True,
            "failures": [],
            "failed_count": 0,
            "total_count": payload["total_count"],
            "error": None,
        }}))
        """
    )
