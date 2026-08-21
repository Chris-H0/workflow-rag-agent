"""Evaluate an LLM-guided per-invocation runtime router on a fixed holdout."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pydantic import field_validator, model_validator

from placement_compiler.adapters.workflows import load_workflow_driver
from placement_compiler.core.catalogue import load_json_or_yaml, load_model_catalogue
from placement_compiler.core.models import PlacementPlan, StrictModel
from placement_compiler.experiments.holdout import select_disjoint_examples
from placement_compiler.profiling.models import (
    EvaluationSettings,
    RunRecord,
    RunTrace,
    SearchArtifact,
)
from placement_compiler.profiling.runner import (
    aggregate_plan_result,
    deterministic_sample,
    write_search_artifacts,
)
from placement_compiler.runtime.dynamic_router import (
    ROUTER_TRACE_PREFIX,
    RuntimeRouterResolver,
)
from placement_compiler.runtime.endpoint_registry import EndpointRegistry, TraceCollector


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HARDWARE_CONTEXT = (
    "Apple M4 Pro with a 16-core integrated GPU and 24 GB unified memory. "
    "The local qwen3.5:2b endpoint runs through Ollama 0.23.0 with reasoning "
    "disabled and a maximum output of 768 tokens. Local calls have no recorded "
    "cloud API cost, but may be slower and less capable than cloud models."
)


class RuntimeRoutingConfig(StrictModel):
    workflow: str
    models: Path
    source_results: Path
    router_endpoint: str
    router_temperature: float = 0.0
    fallback_endpoint: str
    sample_size: int = 50
    development_seed: int = 42
    holdout_seed: int
    repeats: int = 2
    warmup_runs: int = 0
    hardware_context: str = DEFAULT_HARDWARE_CONTEXT
    include_full_workflow_metadata: bool = False
    output: Path

    @field_validator(
        "workflow",
        "router_endpoint",
        "fallback_endpoint",
        "hardware_context",
    )
    @classmethod
    def non_empty_string(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    @model_validator(mode="after")
    def validate_counts(self) -> "RuntimeRoutingConfig":
        if self.sample_size <= 0:
            raise ValueError("sample_size must be positive")
        if self.repeats <= 0:
            raise ValueError("repeats must be positive")
        if self.warmup_runs < 0:
            raise ValueError("warmup_runs must not be negative")
        return self


def load_runtime_routing_config(path: str | Path) -> RuntimeRoutingConfig:
    config_path = Path(path).resolve()
    config = RuntimeRoutingConfig.model_validate(load_json_or_yaml(config_path))
    return config.model_copy(
        update={
            "models": _resolve_path(config_path, config.models),
            "source_results": _resolve_path(config_path, config.source_results),
            "output": _resolve_path(config_path, config.output),
        }
    )


def run_runtime_routing(config: RuntimeRoutingConfig) -> dict[str, Path]:
    _load_dotenvs(config.workflow)
    driver = load_workflow_driver(config.workflow)
    workflow = driver.metadata()
    source = SearchArtifact.model_validate_json(config.source_results.read_text())
    if source.workflow != config.workflow:
        raise ValueError("source results belong to a different workflow")

    loaded = driver.load_examples(
        EvaluationSettings(sample_size=config.sample_size, seed=config.development_seed)
    )
    development_examples = deterministic_sample(
        loaded,
        sample_size=config.sample_size,
        seed=config.development_seed,
        example_id=driver.example_id,
    )
    development_ids = {driver.example_id(example) for example in development_examples}
    if development_ids != set(source.selected_example_ids):
        raise ValueError("source results do not match the configured development sample")

    holdout_examples = select_disjoint_examples(
        loaded,
        excluded_ids=development_ids,
        sample_size=config.sample_size,
        seed=config.holdout_seed,
        example_id=driver.example_id,
    )
    selected_example_ids = [driver.example_id(example) for example in holdout_examples]
    driver.prepare(holdout_examples)

    endpoints = load_model_catalogue(config.models)
    registry = EndpointRegistry(endpoints)
    # Resolve these once so configuration errors fail before any paid run.
    registry.endpoint(config.router_endpoint)
    registry.endpoint(config.fallback_endpoint)

    plan_id = (
        "runtime-router-workflow-context"
        if config.include_full_workflow_metadata
        else "runtime-router"
    )
    plan = PlacementPlan(
        id=plan_id,
        source="candidate",
        description=(
            "GPT-4.1 mini selects a compatible endpoint per invocation using "
            "the complete workflow metadata."
            if config.include_full_workflow_metadata
            else "GPT-4.1 mini selects a compatible endpoint per invocation."
        ),
        assignments={unit.id: "runtime-selected" for unit in workflow.placement_units},
    )
    runs = _load_checkpoint(config.output, config, selected_example_ids)
    completed = {(run.example_id, run.repeat) for run in runs}

    for repeat in range(config.repeats):
        for index, example in enumerate(holdout_examples, start=1):
            example_id = driver.example_id(example)
            if (example_id, repeat) in completed:
                continue
            run = _run_example(
                driver=driver,
                workflow=workflow,
                endpoint_registry=registry,
                config=config,
                plan=plan,
                example=example,
                example_id=example_id,
                repeat=repeat,
            )
            runs.append(run)
            completed.add((example_id, repeat))
            paths = _write_checkpoint(
                config=config,
                workflow_id=workflow.workflow_id,
                primary_metric=driver.primary_metric,
                selected_example_ids=selected_example_ids,
                plan=plan,
                runs=runs,
            )
            print(
                f"Completed {config.workflow} repeat {repeat + 1}/{config.repeats}, "
                f"example {index}/{config.sample_size} ({example_id}); "
                f"quality={run.metrics.get(driver.primary_metric.name)}, "
                f"error={run.error is not None}, "
                f"router_calls={len(_routing_decisions(run))}",
                flush=True,
            )

    return _write_checkpoint(
        config=config,
        workflow_id=workflow.workflow_id,
        primary_metric=driver.primary_metric,
        selected_example_ids=selected_example_ids,
        plan=plan,
        runs=runs,
    )


def _run_example(
    *,
    driver,
    workflow,
    endpoint_registry: EndpointRegistry,
    config: RuntimeRoutingConfig,
    plan: PlacementPlan,
    example: Any,
    example_id: str,
    repeat: int,
) -> RunRecord:
    started = time.perf_counter()
    trace_collector = TraceCollector()
    resolver = RuntimeRouterResolver(
        workflow=workflow,
        endpoint_registry=endpoint_registry,
        router_endpoint_id=config.router_endpoint,
        fallback_endpoint_id=config.fallback_endpoint,
        router_temperature=config.router_temperature,
        hardware_context=config.hardware_context,
        trace_collector=trace_collector,
        include_full_workflow_metadata=config.include_full_workflow_metadata,
    )
    identity = {
        "plan_id": plan.id,
        "plan_source": plan.source,
        "example_id": example_id,
        "repeat": repeat,
    }
    for _ in range(config.warmup_runs):
        driver.run_example(resolver, example, metadata={"placement_id": plan.id})
    try:
        result = driver.run_example(
            resolver,
            example,
            metadata={"placement_id": plan.id},
        )
        result.trace.system_metrics["runtime_routing"] = list(resolver.decisions)
        return RunRecord(**identity, **result.model_dump())
    except Exception as exc:
        return RunRecord(
            **identity,
            metrics={driver.primary_metric.name: 0.0},
            latency_seconds=time.perf_counter() - started,
            error=repr(exc),
            trace=RunTrace(
                invocations=trace_collector.invocations,
                system_metrics={"runtime_routing": list(resolver.decisions)},
            ),
        )


def _write_checkpoint(
    *,
    config: RuntimeRoutingConfig,
    workflow_id: str,
    primary_metric,
    selected_example_ids: list[str],
    plan: PlacementPlan,
    runs: list[RunRecord],
) -> dict[str, Path]:
    result = aggregate_plan_result(
        plan,
        runs,
        primary_metric_name=primary_metric.name,
    )
    result.metrics.update(_runtime_metrics(runs))
    artifact = SearchArtifact(
        workflow_id=workflow_id,
        workflow=config.workflow,
        search_config=_search_config(config),
        primary_metric=primary_metric,
        selected_example_ids=selected_example_ids,
        results=[result],
    )
    return write_search_artifacts(
        artifact=artifact,
        runs=runs,
        output_dir=config.output,
    )


def _runtime_metrics(runs: list[RunRecord]) -> dict[str, Any]:
    router_invocations = [
        invocation
        for run in runs
        for invocation in run.trace.invocations
        if invocation.placement_unit_id.startswith(ROUTER_TRACE_PREFIX)
    ]
    decisions = [decision for run in runs for decision in _routing_decisions(run)]
    endpoint_counts = Counter(
        str(decision["selected_endpoint_id"]) for decision in decisions
    )
    by_unit: dict[str, Counter[str]] = defaultdict(Counter)
    for decision in decisions:
        by_unit[str(decision["placement_unit_id"])][
            str(decision["selected_endpoint_id"])
        ] += 1

    router_cloud_cost_unknown = any(
        invocation.location == "cloud" and invocation.cloud_api_cost is None
        for invocation in router_invocations
    )
    router_cloud_cost = (
        None
        if router_cloud_cost_unknown
        else sum(invocation.cloud_api_cost or 0.0 for invocation in router_invocations)
    )
    return {
        "router_call_count": len(router_invocations),
        "router_fallback_count": sum(
            bool(decision.get("fallback_used")) for decision in decisions
        ),
        "router_latency_seconds": sum(
            invocation.latency_seconds for invocation in router_invocations
        ),
        "router_cloud_api_cost": router_cloud_cost,
        "router_cloud_api_cost_unknown": router_cloud_cost_unknown,
        "router_input_tokens": _sum_nullable(
            invocation.input_tokens for invocation in router_invocations
        ),
        "router_output_tokens": _sum_nullable(
            invocation.output_tokens for invocation in router_invocations
        ),
        "router_total_tokens": _sum_nullable(
            invocation.total_tokens for invocation in router_invocations
        ),
        "selected_endpoint_counts": dict(sorted(endpoint_counts.items())),
        "selected_endpoints_by_unit": {
            unit: dict(sorted(counts.items()))
            for unit, counts in sorted(by_unit.items())
        },
    }


def _routing_decisions(run: RunRecord) -> list[dict[str, Any]]:
    value = run.trace.system_metrics.get("runtime_routing", [])
    return list(value) if isinstance(value, list) else []


def _sum_nullable(values) -> int | None:
    values = list(values)
    if any(value is None for value in values):
        return None
    return sum(values)


def _load_checkpoint(
    output_dir: Path,
    config: RuntimeRoutingConfig,
    selected_example_ids: list[str],
) -> list[RunRecord]:
    results_path = output_dir / "results.json"
    runs_path = output_dir / "runs.jsonl"
    if not results_path.exists() and not runs_path.exists():
        return []
    if not results_path.exists() or not runs_path.exists():
        raise ValueError("runtime-router checkpoint is incomplete")

    artifact = SearchArtifact.model_validate_json(results_path.read_text())
    if artifact.workflow != config.workflow:
        raise ValueError("runtime-router checkpoint belongs to another workflow")
    if artifact.selected_example_ids != selected_example_ids:
        raise ValueError("runtime-router checkpoint uses different examples")
    if artifact.search_config != _search_config(config):
        raise ValueError("runtime-router checkpoint uses a different configuration")

    runs = [
        RunRecord.model_validate_json(line)
        for line in runs_path.read_text().splitlines()
        if line.strip()
    ]
    keys = [(run.example_id, run.repeat) for run in runs]
    if len(keys) != len(set(keys)):
        raise ValueError("runtime-router checkpoint contains duplicate runs")
    return runs


def _search_config(config: RuntimeRoutingConfig) -> dict[str, Any]:
    data = config.model_dump(mode="json")
    # Preserve resume compatibility with the completed original-policy artifacts.
    if not config.include_full_workflow_metadata:
        data.pop("include_full_workflow_metadata")
    data["kind"] = "llm_guided_runtime_routing_holdout"
    return data


def _resolve_path(config_path: Path, value: Path) -> Path:
    path = value.expanduser()
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def _load_dotenvs(workflow: str) -> None:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env", override=True)
    load_dotenv(load_workflow_driver(workflow).root / ".env", override=True)
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    paths = run_runtime_routing(load_runtime_routing_config(args.config))
    print(f"Wrote runtime-routing results to {paths['results']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
