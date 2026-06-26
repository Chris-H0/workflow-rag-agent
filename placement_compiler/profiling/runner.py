from __future__ import annotations

import csv
import json
import math
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path
from typing import Any, Protocol

from placement_compiler.core.models import PlacementArtifact
from placement_compiler.profiling.models import (
    CandidateProfile,
    LoadedCandidateArtifact,
    MetricDefinition,
    PlacementPlan,
    ProfileArtifact,
    ProfileSettings,
    QualityConstraint,
    RankingConfig,
    RankingObjective,
    RunRecord,
    RunTrace,
    WorkflowExecution,
)


class EvaluationAdapter(Protocol):
    @property
    def primary_metric(self) -> MetricDefinition:
        ...

    def load_examples(self, profile: ProfileSettings) -> list[Any]:
        ...

    def example_id(self, example: Any) -> str:
        ...

    def score(self, example: Any, output: dict[str, Any]) -> dict[str, Any]:
        ...


class RuntimeAdapter(Protocol):
    def prepare(self, examples: list[Any]) -> None:
        ...

    def run_example(
        self,
        plan: PlacementPlan,
        example: Any,
        repeat: int,
    ) -> WorkflowExecution:
        ...


def load_candidate_artifact(path: str | Path) -> LoadedCandidateArtifact:
    artifact_path = Path(path).resolve()
    data = json.loads(artifact_path.read_text())
    return LoadedCandidateArtifact(
        path=artifact_path,
        artifact=PlacementArtifact.model_validate(data),
    )


class CandidateProfiler:
    def __init__(
        self,
        *,
        candidate_artifact: LoadedCandidateArtifact,
        evaluation_adapter: EvaluationAdapter,
        runtime_adapter: RuntimeAdapter,
        profile: ProfileSettings,
        baseline_plan: PlacementPlan,
        quality_constraint: QualityConstraint,
        ranking: RankingConfig,
        workflow_name: str,
    ) -> None:
        self.candidate_artifact = candidate_artifact
        self.evaluation_adapter = evaluation_adapter
        self.runtime_adapter = runtime_adapter
        self.profile = profile
        self.baseline_plan = baseline_plan
        self.quality_constraint = quality_constraint
        self.ranking = ranking
        self.workflow_name = workflow_name
        self.warnings: list[str] = []
        self.failures: list[str] = []

    def run(self) -> tuple[ProfileArtifact, list[RunRecord]]:
        all_examples = self.evaluation_adapter.load_examples(self.profile)
        selected_examples = deterministic_sample(
            all_examples,
            sample_size=self.profile.sample_size,
            seed=self.profile.seed,
            example_id=self.evaluation_adapter.example_id,
        )
        selected_example_ids = [
            self.evaluation_adapter.example_id(example) for example in selected_examples
        ]
        self.runtime_adapter.prepare(selected_examples)

        plans = [self.baseline_plan, *self._candidate_plans()]
        run_records: list[RunRecord] = []
        for plan in plans:
            self._run_warmups(plan, selected_examples)
            for repeat in range(self.profile.repeats):
                for example in selected_examples:
                    run_records.append(self._run_measured(plan, example, repeat))

        baseline_profile = aggregate_candidate_profile(self.baseline_plan, run_records)
        candidate_profiles = [
            aggregate_candidate_profile(plan, run_records)
            for plan in plans
            if plan.source == "candidate"
        ]
        apply_quality_constraints(
            baseline=baseline_profile,
            candidates=candidate_profiles,
            primary_metric=self.evaluation_adapter.primary_metric,
            quality_constraint=self.quality_constraint,
        )
        selected_candidate_id = rank_candidates(
            candidate_profiles,
            ranking=self.ranking,
            primary_metric=self.evaluation_adapter.primary_metric,
        )

        return (
            ProfileArtifact(
                source_candidate_artifact=str(self.candidate_artifact.path),
                workflow_id=self.candidate_artifact.artifact.workflow_id,
                workflow=self.workflow_name,
                profile_config=self.profile.model_dump(mode="json", exclude_none=True),
                primary_metric=self.evaluation_adapter.primary_metric,
                selected_example_ids=selected_example_ids,
                baseline=baseline_profile,
                quality_constraint=self.quality_constraint,
                ranking=self.ranking,
                candidates=candidate_profiles,
                selected_candidate_id=selected_candidate_id,
                failures=self.failures,
                warnings=self.warnings,
            ),
            run_records,
        )

    def _candidate_plans(self) -> list[PlacementPlan]:
        return [
            PlacementPlan(
                id=candidate.id,
                assignments=candidate.assignments,
                source="candidate",
                description=candidate.description,
            )
            for candidate in self.candidate_artifact.artifact.candidates
        ]

    def _run_warmups(self, plan: PlacementPlan, examples: list[Any]) -> None:
        if not examples:
            return
        for repeat in range(self.profile.warmup_runs):
            self._run_with_timeout(plan, examples[0], repeat=-repeat - 1)

    def _run_measured(self, plan: PlacementPlan, example: Any, repeat: int) -> RunRecord:
        example_id = self.evaluation_adapter.example_id(example)
        execution = self._run_with_timeout(plan, example, repeat)
        metrics: dict[str, Any] = {}
        if not execution.error and not execution.timed_out:
            try:
                metrics = self.evaluation_adapter.score(example, execution.output)
            except Exception as exc:
                execution.error = f"scoring failed: {exc!r}"
        return run_record_from_execution(
            plan=plan,
            example_id=example_id,
            repeat=repeat,
            execution=execution,
            metrics=metrics,
        )

    def _run_with_timeout(
        self,
        plan: PlacementPlan,
        example: Any,
        repeat: int,
    ) -> WorkflowExecution:
        started = time.perf_counter()
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(self.runtime_adapter.run_example, plan, example, repeat)
        try:
            return future.result(timeout=self.profile.timeout_seconds)
        except TimeoutError:
            future.cancel()
            return WorkflowExecution(
                latency_seconds=time.perf_counter() - started,
                timed_out=True,
                error=f"timed out after {self.profile.timeout_seconds} seconds",
            )
        except Exception as exc:
            return WorkflowExecution(
                latency_seconds=time.perf_counter() - started,
                error=repr(exc),
            )
        finally:
            executor.shutdown(wait=False, cancel_futures=True)


def build_baseline_plan(
    *,
    artifact: PlacementArtifact,
    baseline_type: str,
    endpoint_id: str | None = None,
    candidate_id: str | None = None,
) -> PlacementPlan:
    if baseline_type == "all_endpoint":
        if not endpoint_id:
            raise ValueError("endpoint_id is required for all_endpoint baseline")
        return PlacementPlan(
            id=f"baseline-all-{endpoint_id}",
            source="baseline",
            assignments={
                node.id: endpoint_id for node in artifact.workflow.placement_units()
            },
            description=f"All placement units assigned to {endpoint_id}",
        )
    if baseline_type == "candidate":
        if not candidate_id:
            raise ValueError("candidate_id is required for candidate baseline")
        for candidate in artifact.candidates:
            if candidate.id == candidate_id:
                return PlacementPlan(
                    id=f"baseline-candidate-{candidate.id}",
                    source="baseline",
                    assignments=candidate.assignments,
                    description=f"Candidate baseline {candidate.id}",
                )
        raise ValueError(f"candidate baseline {candidate_id!r} not found")
    raise ValueError(f"unknown baseline type {baseline_type!r}")


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


def run_record_from_execution(
    *,
    plan: PlacementPlan,
    example_id: str,
    repeat: int,
    execution: WorkflowExecution,
    metrics: dict[str, Any],
) -> RunRecord:
    trace = execution.trace or RunTrace()
    token_totals = _token_totals(trace)
    return RunRecord(
        candidate_id=plan.id,
        candidate_source=plan.source,
        example_id=example_id,
        repeat=repeat,
        metrics=metrics,
        latency_seconds=execution.latency_seconds,
        cloud_api_cost=_trace_cloud_cost(trace),
        input_tokens=token_totals["input_tokens"],
        output_tokens=token_totals["output_tokens"],
        total_tokens=token_totals["total_tokens"],
        local_call_count=sum(
            1 for invocation in trace.invocations if invocation.location == "local"
        ),
        cloud_call_count=sum(
            1 for invocation in trace.invocations if invocation.location == "cloud"
        ),
        invocation_count=len(trace.invocations),
        per_placement_unit=_per_unit_stats(trace),
        error=execution.error,
        timed_out=execution.timed_out,
        output=execution.output,
        trace=trace,
    )


def aggregate_candidate_profile(
    plan: PlacementPlan,
    run_records: list[RunRecord],
) -> CandidateProfile:
    records = [record for record in run_records if record.candidate_id == plan.id]
    metrics: dict[str, Any] = {
        "run_count": len(records),
        "failed_run_count": sum(1 for record in records if record.error),
        "timeout_run_count": sum(1 for record in records if record.timed_out),
        "local_call_count": sum(record.local_call_count for record in records),
        "cloud_call_count": sum(record.cloud_call_count for record in records),
        "invocation_count": sum(record.invocation_count for record in records),
    }

    latencies = [
        record.latency_seconds for record in records if record.latency_seconds is not None
    ]
    metrics.update(_latency_summary(latencies))
    metrics["cloud_api_cost"] = _sum_nullable(record.cloud_api_cost for record in records)
    metrics["cloud_api_cost_unknown"] = any(
        record.cloud_api_cost is None
        and any(invocation.location == "cloud" for invocation in record.trace.invocations)
        for record in records
    )
    metrics["input_tokens"] = _sum_nullable(record.input_tokens for record in records)
    metrics["output_tokens"] = _sum_nullable(record.output_tokens for record in records)
    metrics["total_tokens"] = _sum_nullable(record.total_tokens for record in records)
    metrics["per_placement_unit"] = _aggregate_per_unit(records)
    metrics.update(_aggregate_quality_metrics(records))

    return CandidateProfile(
        candidate_id=plan.id,
        candidate_source=plan.source,
        assignments=plan.assignments,
        metrics=metrics,
    )


def apply_quality_constraints(
    *,
    baseline: CandidateProfile,
    candidates: list[CandidateProfile],
    primary_metric: MetricDefinition,
    quality_constraint: QualityConstraint,
) -> None:
    metric_name = quality_constraint.metric
    baseline_value = _metric_value(baseline.metrics, metric_name)
    if baseline_value is None:
        for candidate in candidates:
            candidate.feasible = False
            candidate.infeasible_reason = f"baseline missing metric {metric_name!r}"
        return

    for candidate in candidates:
        value = _metric_value(candidate.metrics, metric_name)
        if value is None:
            candidate.feasible = False
            candidate.infeasible_reason = f"candidate missing metric {metric_name!r}"
            continue
        feasible = _quality_passes(
            value=value,
            baseline_value=baseline_value,
            direction=primary_metric.direction,
            constraint=quality_constraint,
        )
        candidate.feasible = feasible
        if not feasible:
            candidate.infeasible_reason = (
                f"{metric_name}={value} violates quality constraint "
                f"against baseline {baseline_value}"
            )


def rank_candidates(
    candidates: list[CandidateProfile],
    *,
    ranking: RankingConfig,
    primary_metric: MetricDefinition,
) -> str | None:
    feasible = [candidate for candidate in candidates if candidate.feasible]
    if feasible:
        objectives = ranking.objectives or [
            RankingObjective(metric="cloud_api_cost", direction="minimise"),
            RankingObjective(metric="mean_latency_seconds", direction="minimise"),
        ]
        feasible.sort(
            key=lambda candidate: _ranking_key(candidate, objectives, primary_metric)
        )
        for index, candidate in enumerate(feasible, start=1):
            candidate.rank = index
        return feasible[0].candidate_id

    diagnostic = sorted(
        candidates,
        key=lambda candidate: _quality_diagnostic_key(candidate, primary_metric),
    )
    for index, candidate in enumerate(diagnostic, start=1):
        candidate.diagnostic_rank = index
    return None


def write_profile_artifacts(
    *,
    profile: ProfileArtifact,
    runs: list[RunRecord],
    output_dir: str | Path,
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    profile_path = output / "profile.json"
    runs_path = output / "runs.jsonl"
    candidates_path = output / "candidates.csv"
    selected_plan_path = output / "selected_plan.json"

    profile_path.write_text(
        json.dumps(profile.model_dump(mode="json", exclude_none=True), indent=2) + "\n"
    )
    with runs_path.open("w", encoding="utf-8") as handle:
        for run in runs:
            handle.write(json.dumps(run.model_dump(mode="json", exclude_none=True)) + "\n")
    _write_candidates_csv(candidates_path, profile)
    _write_selected_plan(selected_plan_path, profile)
    return {
        "profile": profile_path,
        "runs": runs_path,
        "candidates": candidates_path,
        "selected_plan": selected_plan_path,
    }


def _write_candidates_csv(path: Path, profile: ProfileArtifact) -> None:
    rows = [profile.baseline, *profile.candidates]
    fieldnames = [
        "candidate_id",
        "candidate_source",
        "feasible",
        "rank",
        "diagnostic_rank",
        "infeasible_reason",
        "quality_metric",
        "quality_value",
        "cloud_api_cost",
        "mean_latency_seconds",
        "failed_run_count",
        "timeout_run_count",
    ]
    quality_metric = profile.quality_constraint.metric
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "candidate_id": row.candidate_id,
                    "candidate_source": row.candidate_source,
                    "feasible": row.feasible,
                    "rank": row.rank,
                    "diagnostic_rank": row.diagnostic_rank,
                    "infeasible_reason": row.infeasible_reason,
                    "quality_metric": quality_metric,
                    "quality_value": row.metrics.get(quality_metric),
                    "cloud_api_cost": row.metrics.get("cloud_api_cost"),
                    "mean_latency_seconds": row.metrics.get("mean_latency_seconds"),
                    "failed_run_count": row.metrics.get("failed_run_count"),
                    "timeout_run_count": row.metrics.get("timeout_run_count"),
                }
            )


def _write_selected_plan(path: Path, profile: ProfileArtifact) -> None:
    selected = next(
        (
            candidate
            for candidate in profile.candidates
            if candidate.candidate_id == profile.selected_candidate_id
        ),
        None,
    )
    payload: dict[str, Any] = {
        "schema_version": "2.0",
        "generated_at": profile.generated_at,
        "source_candidate_artifact": profile.source_candidate_artifact,
        "workflow_id": profile.workflow_id,
        "selected_candidate_id": profile.selected_candidate_id,
        "selected_plan": selected.assignments if selected else None,
        "quality_constraint": profile.quality_constraint.model_dump(mode="json"),
        "ranking": profile.ranking.model_dump(mode="json"),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _quality_passes(
    *,
    value: float,
    baseline_value: float,
    direction: str,
    constraint: QualityConstraint,
) -> bool:
    allowed = constraint.max_drop_from_baseline
    if direction == "maximise":
        if allowed is not None and value < baseline_value - allowed:
            return False
        if constraint.absolute_minimum is not None and value < constraint.absolute_minimum:
            return False
        return True

    if allowed is not None and value > baseline_value + allowed:
        return False
    if constraint.absolute_maximum is not None and value > constraint.absolute_maximum:
        return False
    return True


def _ranking_key(
    candidate: CandidateProfile,
    objectives: list[RankingObjective],
    primary_metric: MetricDefinition,
) -> tuple[Any, ...]:
    parts: list[Any] = []
    for objective in objectives:
        value = _metric_value(candidate.metrics, objective.metric)
        if value is None:
            value = math.inf if objective.direction == "minimise" else -math.inf
        parts.append(value if objective.direction == "minimise" else -value)

    quality = _metric_value(candidate.metrics, primary_metric.name)
    if quality is None:
        quality = -math.inf if primary_metric.direction == "maximise" else math.inf
    parts.append(-quality if primary_metric.direction == "maximise" else quality)
    parts.append(candidate.candidate_id)
    return tuple(parts)


def _quality_diagnostic_key(
    candidate: CandidateProfile,
    primary_metric: MetricDefinition,
) -> tuple[Any, str]:
    quality = _metric_value(candidate.metrics, primary_metric.name)
    if quality is None:
        quality = -math.inf if primary_metric.direction == "maximise" else math.inf
    return (
        -quality if primary_metric.direction == "maximise" else quality,
        candidate.candidate_id,
    )


def _metric_value(metrics: dict[str, Any], name: str) -> float | None:
    value = metrics.get(name)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, int | float):
        return float(value)
    return None


def _latency_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "mean_latency_seconds": None,
            "median_latency_seconds": None,
            "p95_latency_seconds": None,
        }
    sorted_values = sorted(values)
    p95_index = min(len(sorted_values) - 1, math.ceil(len(sorted_values) * 0.95) - 1)
    return {
        "mean_latency_seconds": statistics.fmean(values),
        "median_latency_seconds": statistics.median(values),
        "p95_latency_seconds": sorted_values[p95_index],
    }


def _aggregate_quality_metrics(records: list[RunRecord]) -> dict[str, Any]:
    values_by_metric: dict[str, list[Any]] = {}
    for record in records:
        if record.error or record.timed_out:
            continue
        for key, value in record.metrics.items():
            if isinstance(value, bool):
                values_by_metric.setdefault(key, []).append(1.0 if value else 0.0)
            elif isinstance(value, int | float):
                values_by_metric.setdefault(key, []).append(float(value))

    return {
        key: statistics.fmean(values)
        for key, values in values_by_metric.items()
        if values
    }


def _sum_nullable(values) -> int | float | None:
    values = list(values)
    if any(value is None for value in values):
        return None
    return sum(values)


def _trace_cloud_cost(trace: RunTrace) -> float | None:
    costs = [
        invocation.cloud_api_cost
        for invocation in trace.invocations
        if invocation.location == "cloud"
    ]
    if any(cost is None for cost in costs):
        return None
    return sum(costs)


def _token_totals(trace: RunTrace) -> dict[str, int | None]:
    return {
        "input_tokens": _sum_nullable(invocation.input_tokens for invocation in trace.invocations),
        "output_tokens": _sum_nullable(invocation.output_tokens for invocation in trace.invocations),
        "total_tokens": _sum_nullable(invocation.total_tokens for invocation in trace.invocations),
    }


def _per_unit_stats(trace: RunTrace) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for invocation in trace.invocations:
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
        if invocation.cloud_api_cost is None and invocation.location == "cloud":
            unit["cloud_api_cost_unknown"] = True
        else:
            unit["cloud_api_cost"] += invocation.cloud_api_cost or 0.0
    return stats


def _aggregate_per_unit(records: list[RunRecord]) -> dict[str, dict[str, Any]]:
    aggregate: dict[str, dict[str, Any]] = {}
    for record in records:
        for unit_id, stats in record.per_placement_unit.items():
            unit = aggregate.setdefault(
                unit_id,
                {
                    "invocation_count": 0,
                    "latency_seconds": 0.0,
                    "cloud_api_cost": 0.0,
                    "cloud_api_cost_unknown": False,
                },
            )
            unit["invocation_count"] += stats.get("invocation_count", 0)
            unit["latency_seconds"] += stats.get("latency_seconds", 0.0)
            unit["cloud_api_cost"] += stats.get("cloud_api_cost", 0.0)
            unit["cloud_api_cost_unknown"] = (
                unit["cloud_api_cost_unknown"]
                or stats.get("cloud_api_cost_unknown", False)
            )
    return aggregate
