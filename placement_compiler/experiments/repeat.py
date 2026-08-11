"""Repeat selected search placements on their original fixed sample."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from placement_compiler.adapters.workflows import load_workflow_driver
from placement_compiler.core.catalogue import load_model_catalogue
from placement_compiler.pipeline.config import PipelineConfig, load_pipeline_config
from placement_compiler.profiling.models import (
    EvaluationSettings,
    PlanResult,
    RunRecord,
    SearchArtifact,
)
from placement_compiler.profiling.runner import PlanEvaluator, write_search_artifacts
from placement_compiler.runtime.endpoint_registry import EndpointRegistry


REPO_ROOT = Path(__file__).resolve().parents[2]


def run_repeats(
    *,
    config: PipelineConfig,
    source_results_path: Path,
    output_dir: Path,
    plan_ids: list[str],
    additional_repeats: int,
) -> dict[str, Path]:
    if additional_repeats <= 0:
        raise ValueError("additional_repeats must be positive")

    _load_dotenvs(config.workflow)
    driver = load_workflow_driver(config.workflow)
    source = SearchArtifact.model_validate_json(source_results_path.read_text())
    if source.workflow != config.workflow:
        raise ValueError("source results belong to a different workflow")

    plans_by_id = {result.plan.id: result.plan for result in source.results}
    missing = [plan_id for plan_id in plan_ids if plan_id not in plans_by_id]
    if missing:
        raise ValueError(f"unknown source plan IDs: {missing}")
    if len(set(plan_ids)) != len(plan_ids):
        raise ValueError("repeat plan IDs must be unique")

    settings = EvaluationSettings(
        split=config.split,
        sample_size=config.sample_size,
        seed=config.seed,
        repeats=additional_repeats,
        warmup_runs=config.warmup_runs,
        examples=config.examples,
    )
    evaluator = PlanEvaluator(
        workflow=driver.metadata(),
        endpoint_registry=EndpointRegistry(load_model_catalogue(config.models)),
        driver=driver,
        settings=settings,
    )
    selected_example_ids = evaluator.prepare()
    if selected_example_ids != source.selected_example_ids:
        raise ValueError("source results do not match the configured sample")

    results: list[PlanResult] = []
    runs: list[RunRecord] = []
    paths: dict[str, Path] = {}
    for plan_id in plan_ids:
        result, plan_runs = evaluator.evaluate(plans_by_id[plan_id])
        results.append(result)
        runs.extend(plan_runs)
        artifact = SearchArtifact(
            workflow_id=driver.metadata().workflow_id,
            workflow=config.workflow,
            search_config={
                "kind": "post_exhaustive_selected_placement_repeats",
                "source_results": str(source_results_path),
                "seed": config.seed,
                "sample_size": config.sample_size,
                "additional_repeats": additional_repeats,
                "plan_ids": plan_ids,
            },
            primary_metric=driver.primary_metric,
            selected_example_ids=selected_example_ids,
            results=results,
        )
        paths = write_search_artifacts(
            artifact=artifact,
            runs=runs,
            output_dir=output_dir,
        )
        print(
            f"Completed {plan_id}: "
            f"{driver.primary_metric.name}="
            f"{result.metrics.get(driver.primary_metric.name)}, "
            f"failures={result.metrics.get('failed_run_count')}, "
            f"cost={result.metrics.get('cloud_api_cost')}, "
            f"mean_latency={result.metrics.get('mean_latency_seconds')}",
            flush=True,
        )
    return paths


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
    parser.add_argument("--plan-id", required=True, action="append", dest="plan_ids")
    parser.add_argument("--repeats", type=int, default=1, dest="additional_repeats")
    args = parser.parse_args(argv)
    paths = run_repeats(
        config=load_pipeline_config(args.config),
        source_results_path=args.source_results.resolve(),
        output_dir=args.output.resolve(),
        plan_ids=args.plan_ids,
        additional_repeats=args.additional_repeats,
    )
    print(f"Wrote repeat results to {paths['results']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
