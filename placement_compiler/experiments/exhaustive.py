"""Evaluate every valid placement on one fixed workflow sample."""

from __future__ import annotations

import argparse
import json
import os
from itertools import product
from pathlib import Path

from placement_compiler.adapters.workflows import load_workflow_driver
from placement_compiler.core.catalogue import load_model_catalogue
from placement_compiler.core.models import (
    ModelEndpoint,
    PlacementPlan,
    WorkflowMetadata,
)
from placement_compiler.core.validation import PlanValidationError, validate_plan
from placement_compiler.pipeline.config import PipelineConfig, load_pipeline_config
from placement_compiler.profiling.models import PlanResult, RunRecord, SearchArtifact
from placement_compiler.profiling.runner import PlanEvaluator, write_search_artifacts
from placement_compiler.runtime.endpoint_registry import EndpointRegistry


REPO_ROOT = Path(__file__).resolve().parents[2]


def enumerate_valid_plans(
    workflow: WorkflowMetadata,
    endpoints: list[ModelEndpoint],
    strongest_cloud_endpoint: str,
) -> list[PlacementPlan]:
    """Return the complete validated assignment space in stable order."""
    endpoint_ids = [strongest_cloud_endpoint]
    endpoint_ids.extend(
        endpoint.id
        for endpoint in sorted(
            (item for item in endpoints if item.id != strongest_cloud_endpoint),
            key=lambda item: (item.location == "local", item.id),
        )
    )
    unit_ids = [unit.id for unit in workflow.placement_units]
    assignments = []
    for endpoint_choices in product(endpoint_ids, repeat=len(unit_ids)):
        plan = PlacementPlan(
            id="pending",
            assignments=dict(zip(unit_ids, endpoint_choices, strict=True)),
        )
        try:
            validate_plan(plan, workflow, endpoints)
        except PlanValidationError:
            continue
        assignments.append(plan.assignments)

    plans = []
    baseline_assignments = {
        unit_id: strongest_cloud_endpoint for unit_id in unit_ids
    }
    for index, assignment in enumerate(assignments, start=1):
        plan_id = f"placement-{index:03d}"
        is_baseline = assignment == baseline_assignments
        plans.append(
            PlacementPlan(
                id=plan_id,
                source="baseline" if is_baseline else "candidate",
                description=f"Exhaustive valid placement {index:03d}",
                assignments=assignment,
            )
        )
    return plans


def run_exhaustive(
    *,
    config: PipelineConfig,
    source_results_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    _load_dotenvs(config.workflow)
    driver = load_workflow_driver(config.workflow)
    workflow = driver.metadata()
    endpoints = load_model_catalogue(config.models)
    evaluator = PlanEvaluator(
        workflow=workflow,
        endpoint_registry=EndpointRegistry(endpoints),
        driver=driver,
        settings=config,
    )
    selected_example_ids = evaluator.prepare()
    plans = enumerate_valid_plans(
        workflow,
        endpoints,
        config.strongest_cloud_endpoint,
    )

    results, runs = _load_or_seed_results(
        output_dir=output_dir,
        source_results_path=source_results_path,
        selected_example_ids=selected_example_ids,
        plans=plans,
        expected_runs_per_plan=config.sample_size * config.repeats,
    )
    paths = _write_checkpoint(
        config=config,
        driver=driver,
        plans=plans,
        selected_example_ids=selected_example_ids,
        results=results,
        runs=runs,
        output_dir=output_dir,
        source_results_path=source_results_path,
    )

    completed = {_signature(result.plan, workflow) for result in results}
    for plan in plans:
        signature = _signature(plan, workflow)
        if signature in completed:
            continue
        print(
            f"Evaluating {plan.id} ({len(completed) + 1}/{len(plans)}): "
            f"{plan.assignments}",
            flush=True,
        )
        result, plan_runs = evaluator.evaluate(plan)
        results.append(result)
        runs.extend(plan_runs)
        completed.add(signature)
        paths = _write_checkpoint(
            config=config,
            driver=driver,
            plans=plans,
            selected_example_ids=selected_example_ids,
            results=results,
            runs=runs,
            output_dir=output_dir,
            source_results_path=source_results_path,
        )
        metrics = result.metrics
        print(
            f"Completed {plan.id}: "
            f"{driver.primary_metric.name}="
            f"{metrics.get(driver.primary_metric.name)}, "
            f"failures={metrics.get('failed_run_count')}, "
            f"cost={metrics.get('cloud_api_cost')}, "
            f"mean_latency={metrics.get('mean_latency_seconds')}",
            flush=True,
        )

    expected_runs = len(plans) * config.sample_size * config.repeats
    if len(results) != len(plans) or len(runs) != expected_runs:
        raise RuntimeError(
            f"incomplete exhaustive result: {len(results)}/{len(plans)} placements, "
            f"{len(runs)}/{expected_runs} runs"
        )
    return paths


def _load_or_seed_results(
    *,
    output_dir: Path,
    source_results_path: Path,
    selected_example_ids: list[str],
    plans: list[PlacementPlan],
    expected_runs_per_plan: int,
) -> tuple[list[PlanResult], list[RunRecord]]:
    checkpoint_path = output_dir / "results.json"
    if checkpoint_path.exists():
        return _load_artifact(checkpoint_path, selected_example_ids, expected_runs_per_plan)

    source_artifact, source_runs = _load_artifact(
        source_results_path,
        selected_example_ids,
        expected_runs_per_plan,
    )
    plan_by_signature = {
        tuple(sorted(plan.assignments.items())): plan for plan in plans
    }
    source_id_by_signature = {
        tuple(sorted(result.plan.assignments.items())): result.plan.id
        for result in source_artifact
    }
    source_runs_by_id: dict[str, list[RunRecord]] = {}
    for run in source_runs:
        source_runs_by_id.setdefault(run.plan_id, []).append(run)

    results = []
    runs = []
    for source_result in source_artifact:
        signature = tuple(sorted(source_result.plan.assignments.items()))
        exhaustive_plan = plan_by_signature.get(signature)
        if exhaustive_plan is None:
            raise ValueError(
                f"source placement {source_result.plan.id!r} is not in the valid space"
            )
        source_id = source_id_by_signature[signature]
        reused_plan = exhaustive_plan.model_copy(
            update={
                "description": (
                    f"{exhaustive_plan.description}; reused from {source_id}"
                ),
                "rationale": source_result.plan.rationale,
            }
        )
        results.append(
            PlanResult(plan=reused_plan, metrics=source_result.metrics)
        )
        runs.extend(
            run.model_copy(
                update={
                    "plan_id": reused_plan.id,
                    "plan_source": reused_plan.source,
                }
            )
            for run in source_runs_by_id[source_id]
        )
    return results, runs


def _load_artifact(
    results_path: Path,
    selected_example_ids: list[str],
    expected_runs_per_plan: int,
) -> tuple[list[PlanResult], list[RunRecord]]:
    artifact = SearchArtifact.model_validate_json(results_path.read_text())
    if artifact.selected_example_ids != selected_example_ids:
        raise ValueError(f"sample mismatch in {results_path}")
    runs_path = results_path.with_name("runs.jsonl")
    runs = [
        RunRecord.model_validate_json(line)
        for line in runs_path.read_text().splitlines()
        if line.strip()
    ]
    counts: dict[str, int] = {}
    for run in runs:
        counts[run.plan_id] = counts.get(run.plan_id, 0) + 1
    result_ids = {result.plan.id for result in artifact.results}
    if set(counts) != result_ids:
        raise ValueError(f"result/run placement mismatch in {results_path}")
    bad_counts = {
        plan_id: count
        for plan_id, count in counts.items()
        if count != expected_runs_per_plan
    }
    if bad_counts:
        raise ValueError(f"incomplete source runs in {results_path}: {bad_counts}")
    return artifact.results, runs


def _write_checkpoint(
    *,
    config: PipelineConfig,
    driver,
    plans: list[PlacementPlan],
    selected_example_ids: list[str],
    results: list[PlanResult],
    runs: list[RunRecord],
    output_dir: Path,
    source_results_path: Path,
) -> dict[str, Path]:
    plan_order = {
        _signature(plan, driver.metadata()): index
        for index, plan in enumerate(plans)
    }
    ordered_results = sorted(
        results,
        key=lambda result: plan_order[_signature(result.plan, driver.metadata())],
    )
    run_order = {result.plan.id: index for index, result in enumerate(ordered_results)}
    ordered_runs = sorted(
        runs,
        key=lambda run: (run_order[run.plan_id], run.repeat, run.example_id),
    )
    artifact = SearchArtifact(
        workflow_id=driver.metadata().workflow_id,
        workflow=config.workflow,
        search_config={
            "kind": "exhaustive_valid_placement_evaluation",
            "models": str(config.models),
            "strongest_cloud_endpoint": config.strongest_cloud_endpoint,
            "sample_size": config.sample_size,
            "seed": config.seed,
            "repeats": config.repeats,
            "warmup_runs": config.warmup_runs,
            "valid_placement_count": len(plans),
            "source_results": str(source_results_path),
        },
        primary_metric=driver.primary_metric,
        selected_example_ids=selected_example_ids,
        results=ordered_results,
    )
    return write_search_artifacts(
        artifact=artifact,
        runs=ordered_runs,
        output_dir=output_dir,
    )


def _signature(plan: PlacementPlan, workflow: WorkflowMetadata) -> tuple[str, ...]:
    return tuple(plan.assignments[unit.id] for unit in workflow.placement_units)


def _load_dotenvs(workflow: str) -> None:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env", override=True)
    load_dotenv(load_workflow_driver(workflow).root / ".env", override=True)
    # Experiment artifacts already retain the traces needed for analysis.
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    paths = run_exhaustive(
        config=load_pipeline_config(args.config),
        source_results_path=args.source_results.resolve(),
        output_dir=args.output.resolve(),
    )
    print(f"Wrote exhaustive results to {paths['results']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
