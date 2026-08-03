"""Evaluate complete model placements on one fixed workflow sample."""

from __future__ import annotations

import json
import math
import random
import statistics
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from placement_compiler.core.models import (
    ModelEndpoint,
    PlacementPlan,
    WorkflowMetadata,
)
from placement_compiler.core.validation import validate_plan
from placement_compiler.profiling.models import (
    PlanResult,
    EvaluationSettings,
    RunRecord,
    RunTrace,
    SearchArtifact,
)
from placement_compiler.runtime.endpoint_registry import (
    EndpointRegistry,
    ModelResolver,
    TraceCollector,
)

if TYPE_CHECKING:
    from placement_compiler.adapters.workflows import WorkflowDriver


class PlanEvaluator:
    def __init__(
        self,
        *,
        workflow: WorkflowMetadata,
        endpoint_registry: EndpointRegistry,
        driver: WorkflowDriver,
        settings: EvaluationSettings,
    ) -> None:
        self.workflow = workflow
        self.endpoint_registry = endpoint_registry
        self.driver = driver
        self.settings = settings
        self.examples: list[Any] = []

    def prepare(self) -> list[str]:
        loaded = self.driver.load_examples(self.settings)
        self.examples = deterministic_sample(
            loaded,
            sample_size=self.settings.sample_size,
            seed=self.settings.seed,
            example_id=self.driver.example_id,
        )
        self.driver.prepare(self.examples)
        return [self.driver.example_id(example) for example in self.examples]

    def evaluate(self, plan: PlacementPlan) -> tuple[PlanResult, list[RunRecord]]:
        if not self.examples:
            raise RuntimeError("prepare() must be called before evaluate()")

        for _ in range(self.settings.warmup_runs):
            self._run(plan, self.examples[0], repeat=-1)

        runs = [
            self._run(plan, example, repeat=repeat)
            for repeat in range(self.settings.repeats)
            for example in self.examples
        ]
        return aggregate_plan_result(plan, runs), runs

    def _run(
        self,
        plan: PlacementPlan,
        example: Any,
        *,
        repeat: int,
    ) -> RunRecord:
        started = time.perf_counter()
        trace_collector = TraceCollector()
        resolver = ModelResolver(
            plan=plan,
            endpoint_registry=self.endpoint_registry,
            workflow=self.workflow,
            trace_collector=trace_collector,
        )
        identity = {
            "plan_id": plan.id,
            "plan_source": plan.source,
            "example_id": self.driver.example_id(example),
            "repeat": repeat,
        }
        try:
            result = self.driver.run_example(
                resolver,
                example,
                metadata={"placement_id": plan.id},
            )
            return RunRecord(**identity, **result.model_dump())
        except Exception as exc:
            return RunRecord(
                **identity,
                latency_seconds=time.perf_counter() - started,
                error=repr(exc),
                trace=RunTrace(invocations=trace_collector.invocations),
            )


def build_cloud_baseline(
    workflow: WorkflowMetadata,
    endpoints: list[ModelEndpoint],
    endpoint_id: str,
) -> PlacementPlan:
    endpoint = next((item for item in endpoints if item.id == endpoint_id), None)
    if endpoint is None:
        raise ValueError(f"strongest cloud endpoint {endpoint_id!r} was not found")
    if endpoint.location != "cloud":
        raise ValueError(f"strongest cloud endpoint {endpoint_id!r} is not cloud-hosted")

    plan = PlacementPlan(
        id="baseline",
        source="baseline",
        description=f"Strongest cloud model on every placement unit: {endpoint_id}",
        assignments={unit.id: endpoint_id for unit in workflow.placement_units},
    )
    return validate_plan(plan, workflow, endpoints)


def deterministic_sample(
    examples: list[Any],
    *,
    sample_size: int,
    seed: int,
    example_id,
) -> list[Any]:
    if sample_size > len(examples):
        raise ValueError(
            f"sample_size={sample_size} exceeds loaded example count {len(examples)}"
        )
    ordered = sorted(examples, key=example_id)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    return sorted(ordered[:sample_size], key=example_id)


def aggregate_plan_result(
    plan: PlacementPlan,
    runs: list[RunRecord],
) -> PlanResult:
    invocations = [
        invocation
        for run in runs
        for invocation in run.trace.invocations
    ]
    metrics: dict[str, Any] = {
        "run_count": len(runs),
        "failed_run_count": sum(1 for run in runs if run.error),
        "local_call_count": sum(
            1 for invocation in invocations if invocation.location == "local"
        ),
        "cloud_call_count": sum(
            1 for invocation in invocations if invocation.location == "cloud"
        ),
        "invocation_count": len(invocations),
    }
    metrics.update(_latency_summary([run.latency_seconds for run in runs]))
    metrics["cloud_api_cost"] = _cloud_cost(invocations)
    metrics["cloud_api_cost_unknown"] = any(
        invocation.location == "cloud" and invocation.cloud_api_cost is None
        for invocation in invocations
    )
    metrics.update(_token_totals(invocations))
    metrics["per_placement_unit"] = _per_unit_stats(invocations)
    metrics.update(_quality_metrics(runs))
    return PlanResult(plan=plan, metrics=metrics)


def write_search_artifacts(
    *,
    artifact: SearchArtifact,
    runs: list[RunRecord],
    output_dir: str | Path,
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results_path = output / "results.json"
    runs_path = output / "runs.jsonl"

    results_path.write_text(
        json.dumps(artifact.model_dump(mode="json", exclude_none=True), indent=2) + "\n"
    )
    with runs_path.open("w", encoding="utf-8") as handle:
        for run in runs:
            handle.write(json.dumps(run.model_dump(mode="json", exclude_none=True)) + "\n")
    return {"results": results_path, "runs": runs_path}


def _quality_metrics(runs: list[RunRecord]) -> dict[str, Any]:
    values_by_metric: dict[str, list[float]] = {}
    for run in runs:
        if run.error:
            continue
        for key, value in run.metrics.items():
            if isinstance(value, bool):
                values_by_metric.setdefault(key, []).append(float(value))
            elif isinstance(value, int | float):
                values_by_metric.setdefault(key, []).append(float(value))
    return {
        key: statistics.fmean(values)
        for key, values in values_by_metric.items()
        if values
    }


def _latency_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "mean_latency_seconds": None,
            "median_latency_seconds": None,
            "p95_latency_seconds": None,
        }
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)
    return {
        "mean_latency_seconds": statistics.fmean(values),
        "median_latency_seconds": statistics.median(values),
        "p95_latency_seconds": ordered[p95_index],
    }


def _sum_nullable(values) -> int | float | None:
    values = list(values)
    if any(value is None for value in values):
        return None
    return sum(values)


def _cloud_cost(invocations: list[Any]) -> float | None:
    return _sum_nullable(
        invocation.cloud_api_cost
        for invocation in invocations
        if invocation.location == "cloud"
    )


def _token_totals(invocations: list[Any]) -> dict[str, int | None]:
    return {
        "input_tokens": _sum_nullable(item.input_tokens for item in invocations),
        "output_tokens": _sum_nullable(item.output_tokens for item in invocations),
        "total_tokens": _sum_nullable(item.total_tokens for item in invocations),
    }


def _per_unit_stats(invocations: list[Any]) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for invocation in invocations:
        unit = stats.setdefault(
            invocation.placement_unit_id,
            {
                "invocation_count": 0,
                "latency_seconds": 0.0,
                "cloud_api_cost": 0.0,
                "cloud_api_cost_unknown": False,
            },
        )
        unit["invocation_count"] += 1
        unit["latency_seconds"] += invocation.latency_seconds
        if invocation.location == "cloud" and invocation.cloud_api_cost is None:
            unit["cloud_api_cost_unknown"] = True
        else:
            unit["cloud_api_cost"] += invocation.cloud_api_cost or 0.0
    return stats
