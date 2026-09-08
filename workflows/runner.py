"""Run a individual workflow."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import os
from pathlib import Path

from dotenv import load_dotenv

from placement_compiler.adapters.workflows import load_workflow_driver
from placement_compiler.core.catalogue import load_model_catalogue
from placement_compiler.core.models import PlacementPlan
from placement_compiler.core.validation import validate_plan
from placement_compiler.profiling.models import EvaluationSettings, SearchArtifact
from placement_compiler.profiling.runner import PlanEvaluator, write_search_artifacts
from placement_compiler.runtime.endpoint_registry import EndpointRegistry


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = REPO_ROOT / "placement_compiler/examples/model_endpoints.yaml"


def run_workflow(
    workflow: str,
    assignments: dict[str, str],
    *,
    settings: EvaluationSettings,
    models: Path,
    output: Path,
) -> dict[str, Path]:
    output = output.expanduser().resolve()
    if any((output / name).exists() for name in ("results.json", "runs.jsonl")):
        raise FileExistsError(f"Results already exist in {output}; choose another output directory")

    driver = load_workflow_driver(workflow)
    load_dotenv(REPO_ROOT / ".env", override=True)
    load_dotenv(driver.root / ".env", override=True)
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

    models = models.expanduser().resolve()
    endpoints = load_model_catalogue(models)
    metadata = driver.metadata()
    plan = validate_plan(
        PlacementPlan(
            id="standalone",
            source="candidate",
            description="Manually configured assignment; no compiler proposals",
            assignments=assignments,
        ),
        metadata,
        endpoints,
    )
    evaluator = PlanEvaluator(
        workflow=metadata,
        endpoint_registry=EndpointRegistry(endpoints),
        driver=driver,
        settings=settings,
    )
    selected_ids = evaluator.prepare()
    result, runs = evaluator.evaluate(plan)
    artifact = SearchArtifact(
        workflow_id=metadata.workflow_id,
        workflow=workflow,
        search_config={
            "kind": "standalone_fixed_assignment",
            **settings.model_dump(mode="json", exclude_none=True),
            "models": str(models),
            "model_endpoints": [endpoint.model_dump(mode="json") for endpoint in endpoints],
        },
        primary_metric=driver.primary_metric,
        selected_example_ids=selected_ids,
        results=[result],
    )
    paths = write_search_artifacts(artifact=artifact, runs=runs, output_dir=output)
    print(
        f"Completed {workflow}: {driver.primary_metric.name}="
        f"{result.metrics[driver.primary_metric.name]}, "
        f"failures={result.metrics['failed_run_count']}, "
        f"cost={result.metrics['cloud_api_cost']}, "
        f"mean_latency={result.metrics['mean_latency_seconds']}",
        flush=True,
    )
    for name, path in paths.items():
        print(f"Wrote {name} to {path}")
    return paths


def main(workflow: str, assignments: dict[str, str]) -> None:
    parser = argparse.ArgumentParser(
        description=f"Evaluate one fixed {workflow} assignment."
    )
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or REPO_ROOT / "workflow_runs" / workflow / timestamp
    run_workflow(
        workflow,
        assignments,
        settings=EvaluationSettings(
            sample_size=args.sample_size,
            seed=args.seed,
            repeats=args.repeats,
            warmup_runs=0,
        ),
        models=args.models,
        output=output,
    )
