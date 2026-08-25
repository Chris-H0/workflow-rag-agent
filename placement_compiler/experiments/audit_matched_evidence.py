"""Audit the final matched QA and Code experiment artifacts and row counts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from placement_compiler.profiling.models import RunRecord, SearchArtifact


EXPECTED_FRESH_EXECUTIONS = 6_550
EXPECTED_LAYOUT = {
    "compiler-sequence-01": (13, 650),
    "compiler-sequence-02": (13, 650),
    "compiler-sequence-03": (13, 650),
    "exhaustive": (81, 4_050),
    "holdout-seed-43": (5, 250),
    "holdout-seed-56": (3, 150),
    "selected-repeats-01": (3, 150),
    "selected-repeats-additional-03": (3, 450),
    "runtime-routing": (1, 100),
    "runtime-routing-workflow-context": (1, 100),
}


def audit_matched_evidence(qa_root: Path, code_root: Path) -> dict[str, Any]:
    qa = _audit_workflow("qa", qa_root)
    code = _audit_workflow("code", code_root)
    if qa["method"] != code["method"]:
        raise ValueError("QA and Code evidence methods do not match")
    return {
        "status": "complete",
        "expected_fresh_executions_per_workflow": EXPECTED_FRESH_EXECUTIONS,
        "qa": qa,
        "code": code,
        "matched_method": qa["method"],
    }


def _audit_workflow(workflow: str, root: Path) -> dict[str, Any]:
    expected_primary_metric = (
        "answer_token_f1" if workflow == "qa" else "mbpp_plus_pass"
    )
    artifacts: dict[str, SearchArtifact] = {}
    runs_by_family: dict[str, list[RunRecord]] = {}
    for family, (expected_results, expected_runs) in EXPECTED_LAYOUT.items():
        results_path = root / family / "results.json"
        runs_path = root / family / "runs.jsonl"
        artifact = SearchArtifact.model_validate_json(results_path.read_text())
        runs = [
            RunRecord.model_validate_json(line)
            for line in runs_path.read_text().splitlines()
            if line.strip()
        ]
        if artifact.workflow != workflow:
            raise ValueError(f"{workflow}/{family} belongs to {artifact.workflow}")
        if artifact.primary_metric.name != expected_primary_metric:
            raise ValueError(
                f"{workflow}/{family} uses primary metric "
                f"{artifact.primary_metric.name!r}; expected {expected_primary_metric!r}"
            )
        if len(artifact.results) != expected_results or len(runs) != expected_runs:
            raise ValueError(
                f"{workflow}/{family} has {len(artifact.results)} results and "
                f"{len(runs)} runs; expected {expected_results} and {expected_runs}"
            )
        result_ids = {result.plan.id for result in artifact.results}
        if {run.plan_id for run in runs} != result_ids:
            raise ValueError(f"{workflow}/{family} result/run plan IDs differ")
        artifacts[family] = artifact
        runs_by_family[family] = runs

    development_ids = artifacts["compiler-sequence-01"].selected_example_ids
    if len(development_ids) != 50:
        raise ValueError(f"{workflow} development sample does not contain 50 examples")
    for family in (
        "compiler-sequence-02",
        "compiler-sequence-03",
        "exhaustive",
        "selected-repeats-01",
        "selected-repeats-additional-03",
    ):
        if artifacts[family].selected_example_ids != development_ids:
            raise ValueError(f"{workflow}/{family} development identifiers differ")

    exhaustive = artifacts["exhaustive"]
    signatures = {
        tuple(sorted(result.plan.assignments.items())) for result in exhaustive.results
    }
    if len(signatures) != 81:
        raise ValueError(f"{workflow} exhaustive assignments are not 81 unique plans")
    if int(exhaustive.search_config.get("valid_placement_count", 81)) != 81:
        raise ValueError(f"{workflow} exhaustive valid-placement count is not 81")

    holdout_43_ids = set(artifacts["holdout-seed-43"].selected_example_ids)
    holdout_56_ids = set(artifacts["holdout-seed-56"].selected_example_ids)
    if len(holdout_43_ids) != 50 or len(holdout_56_ids) != 50:
        raise ValueError(f"{workflow} holdouts must each contain 50 examples")
    if holdout_43_ids & set(development_ids) or holdout_56_ids & set(development_ids):
        raise ValueError(f"{workflow} holdout overlaps the development sample")
    if len(holdout_43_ids & holdout_56_ids) != 3:
        raise ValueError(f"{workflow} holdouts must overlap on exactly three examples")

    selected_signatures = {
        tuple(sorted(result.plan.assignments.items()))
        for result in artifacts["holdout-seed-56"].results
    }
    for family in ("selected-repeats-01", "selected-repeats-additional-03"):
        repeat_signatures = {
            tuple(sorted(result.plan.assignments.items()))
            for result in artifacts[family].results
        }
        if repeat_signatures != selected_signatures:
            raise ValueError(f"{workflow}/{family} assignments differ from seed 56")

    for family in ("runtime-routing", "runtime-routing-workflow-context"):
        if set(artifacts[family].selected_example_ids) != holdout_56_ids:
            raise ValueError(f"{workflow}/{family} does not use the seed-56 holdout")
        if {run.repeat for run in runs_by_family[family]} != {0, 1}:
            raise ValueError(f"{workflow}/{family} does not contain two generations")

    failed_runs = {
        family: sum(1 for run in runs if run.error)
        for family, runs in runs_by_family.items()
    }
    method = {
        "valid_placements": 81,
        "development_examples": 50,
        "compiler_sequences": 3,
        "compiler_placements_per_sequence": 13,
        "exhaustive_rows": 4_050,
        "first_holdout_plans": 5,
        "replication_holdout_plans": 3,
        "selected_assignments": 3,
        "selected_total_generations": 5,
        "runtime_policies": 2,
        "runtime_generations_per_policy": 2,
        "fresh_executions": EXPECTED_FRESH_EXECUTIONS,
    }
    return {
        "root": str(root.resolve()),
        "primary_metric": expected_primary_metric,
        "method": method,
        "artifact_rows": sum(len(runs) for runs in runs_by_family.values()),
        "fresh_executions": EXPECTED_FRESH_EXECUTIONS,
        "failed_runs": failed_runs,
        "holdout_unique_examples": len(holdout_43_ids | holdout_56_ids),
        "holdout_overlap": len(holdout_43_ids & holdout_56_ids),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-root", required=True, type=Path)
    parser.add_argument("--code-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    audit = audit_matched_evidence(args.qa_root.resolve(), args.code_root.resolve())
    rendered = json.dumps(audit, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
        print(f"Wrote matched evidence audit to {args.output.resolve()}")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
