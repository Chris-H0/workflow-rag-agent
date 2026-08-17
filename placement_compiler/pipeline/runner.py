"""Run the single sequential model-routing search."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from placement_compiler.adapters.workflows import load_workflow_driver
from placement_compiler.core.catalogue import load_model_catalogue
from placement_compiler.core.models import CompilerAttempt, ModelEndpoint
from placement_compiler.generation.candidates import (
    CandidateGenerationError,
    CandidateGenerator,
    CompilerLLM,
)
from placement_compiler.generation.llm import LangChainCompilerLLM
from placement_compiler.pipeline.config import PipelineConfig
from placement_compiler.profiling.models import (
    CompilationMetrics,
    EvaluationSettings,
    MetricDefinition,
    PlanResult,
    RunRecord,
    SearchArtifact,
)
from placement_compiler.profiling.runner import (
    PlanEvaluator,
    build_cloud_baseline,
    write_search_artifacts,
)
from placement_compiler.runtime.endpoint_registry import EndpointRegistry


def run_pipeline(
    config: PipelineConfig,
    *,
    compiler_llm: CompilerLLM | None = None,
    model_factory: Callable[[ModelEndpoint], Any] | None = None,
) -> dict[str, Path]:
    driver = load_workflow_driver(config.workflow)
    workflow = driver.metadata()
    endpoints = load_model_catalogue(config.models)
    endpoint_registry = EndpointRegistry(endpoints, model_factory=model_factory)
    evaluator = PlanEvaluator(
        workflow=workflow,
        endpoint_registry=endpoint_registry,
        driver=driver,
        settings=_evaluation_settings(config),
    )
    selected_example_ids = evaluator.prepare()

    baseline = build_cloud_baseline(
        workflow,
        endpoints,
        config.strongest_cloud_endpoint,
    )
    profiling_started = time.perf_counter()
    baseline_result, baseline_runs = evaluator.evaluate(baseline)
    profiling_elapsed_seconds = time.perf_counter() - profiling_started
    results = [baseline_result]
    runs = list(baseline_runs)
    compiler_attempts: list[CompilerAttempt] = []
    compiler_history = [_compiler_history_entry(baseline_result, baseline_runs)]
    paths = _write_checkpoint(
        config=config,
        workflow_id=workflow.workflow_id,
        primary_metric=driver.primary_metric,
        selected_example_ids=selected_example_ids,
        results=results,
        runs=runs,
        compiler_attempts=compiler_attempts,
        profiling_elapsed_seconds=profiling_elapsed_seconds,
    )
    _print_progress(baseline_result, driver.primary_metric.name)

    generator = CandidateGenerator(
        compiler_llm=compiler_llm or build_compiler_llm(config, endpoints),
        max_attempts=config.max_attempts,
    )
    for iteration in range(1, config.iterations + 1):
        try:
            generated = generator.propose_with_metrics(
                workflow=workflow,
                model_endpoints=endpoints,
                iteration=iteration,
                evaluated_results=compiler_history,
            )
        except CandidateGenerationError as exc:
            compiler_attempts.extend(exc.attempts)
            _write_checkpoint(
                config=config,
                workflow_id=workflow.workflow_id,
                primary_metric=driver.primary_metric,
                selected_example_ids=selected_example_ids,
                results=results,
                runs=runs,
                compiler_attempts=compiler_attempts,
                profiling_elapsed_seconds=profiling_elapsed_seconds,
            )
            raise

        compiler_attempts.extend(generated.attempts)
        candidate = generated.plan
        profiling_started = time.perf_counter()
        result, candidate_runs = evaluator.evaluate(candidate)
        profiling_elapsed_seconds += time.perf_counter() - profiling_started
        results.append(result)
        runs.extend(candidate_runs)
        compiler_history.append(_compiler_history_entry(result, candidate_runs))
        paths = _write_checkpoint(
            config=config,
            workflow_id=workflow.workflow_id,
            primary_metric=driver.primary_metric,
            selected_example_ids=selected_example_ids,
            results=results,
            runs=runs,
            compiler_attempts=compiler_attempts,
            profiling_elapsed_seconds=profiling_elapsed_seconds,
        )
        _print_progress(result, driver.primary_metric.name)

    return paths


def _write_checkpoint(
    *,
    config: PipelineConfig,
    workflow_id: str,
    primary_metric: MetricDefinition,
    selected_example_ids: list[str],
    results: list[PlanResult],
    runs: list[RunRecord],
    compiler_attempts: list[CompilerAttempt],
    profiling_elapsed_seconds: float,
) -> dict[str, Path]:
    artifact = SearchArtifact(
        workflow_id=workflow_id,
        workflow=config.workflow,
        search_config=config.model_dump(mode="json", exclude_none=True),
        primary_metric=primary_metric,
        selected_example_ids=selected_example_ids,
        results=results,
        compiler_attempts=compiler_attempts,
        compilation_metrics=_compilation_metrics(
            results,
            compiler_attempts,
            profiling_elapsed_seconds=profiling_elapsed_seconds,
        ),
    )
    return write_search_artifacts(
        artifact=artifact,
        runs=runs,
        output_dir=config.output,
    )


def _print_progress(result: PlanResult, primary_metric_name: str) -> None:
    metrics = result.metrics
    print(
        f"Completed {result.plan.id}: "
        f"{primary_metric_name}={metrics.get(primary_metric_name)}, "
        f"failures={metrics.get('failed_run_count')}, "
        f"cost={metrics.get('cloud_api_cost')}, "
        f"mean_latency={metrics.get('mean_latency_seconds')}",
        flush=True,
    )


def build_compiler_llm(
    config: PipelineConfig,
    endpoints: list[ModelEndpoint] | None = None,
) -> LangChainCompilerLLM:
    model_kwargs = dict(config.compiler.model_kwargs)
    if config.compiler.temperature is not None:
        model_kwargs["temperature"] = config.compiler.temperature
    input_price, output_price = _compiler_prices(config, endpoints or [])
    return LangChainCompilerLLM(
        provider=config.compiler.provider,
        model=config.compiler.model,
        structured_output_method=config.compiler.structured_output_method,
        input_cost_per_million_tokens=input_price,
        output_cost_per_million_tokens=output_price,
        **model_kwargs,
    )


def _compiler_prices(
    config: PipelineConfig,
    endpoints: list[ModelEndpoint],
) -> tuple[float | None, float | None]:
    rates = {
        (
            endpoint.input_cost_per_million_tokens,
            endpoint.output_cost_per_million_tokens,
        )
        for endpoint in endpoints
        if endpoint.location == "cloud"
        and endpoint.provider == config.compiler.provider
        and endpoint.model == config.compiler.model
    }
    if len(rates) != 1:
        return None, None
    return next(iter(rates))


def _compilation_metrics(
    results: list[PlanResult],
    attempts: list[CompilerAttempt],
    *,
    profiling_elapsed_seconds: float,
) -> CompilationMetrics:
    token_usage_unknown = any(
        attempt.prompt_tokens is None
        or attempt.completion_tokens is None
        or attempt.total_tokens is None
        for attempt in attempts
    )
    direct_cost_unknown = any(not attempt.cost_known for attempt in attempts)
    profiling_cost_unknown = any(
        bool(result.metrics.get("cloud_api_cost_unknown", False))
        or result.metrics.get("cloud_api_cost") is None
        for result in results
    )

    prompt_tokens = (
        None
        if token_usage_unknown
        else sum(attempt.prompt_tokens or 0 for attempt in attempts)
    )
    completion_tokens = (
        None
        if token_usage_unknown
        else sum(attempt.completion_tokens or 0 for attempt in attempts)
    )
    total_tokens = (
        None
        if token_usage_unknown
        else sum(attempt.total_tokens or 0 for attempt in attempts)
    )
    direct_cost = (
        None
        if direct_cost_unknown
        else sum(attempt.api_cost or 0.0 for attempt in attempts)
    )
    profiling_cost = (
        None
        if profiling_cost_unknown
        else sum(float(result.metrics["cloud_api_cost"]) for result in results)
    )
    total_cost_unknown = direct_cost_unknown or profiling_cost_unknown
    total_cost = (
        None
        if total_cost_unknown
        else (direct_cost or 0.0) + (profiling_cost or 0.0)
    )
    proposal_latency = sum(
        attempt.proposal_latency_seconds for attempt in attempts
    )

    return CompilationMetrics(
        attempt_count=len(attempts),
        validation_error_count=sum(
            len(attempt.validation_errors) for attempt in attempts
        ),
        duplicate_proposal_count=sum(
            1 for attempt in attempts if attempt.duplicate_proposal
        ),
        compiler_prompt_tokens=prompt_tokens,
        compiler_completion_tokens=completion_tokens,
        compiler_total_tokens=total_tokens,
        compiler_token_usage_unknown=token_usage_unknown,
        direct_compiler_api_cost=direct_cost,
        direct_compiler_api_cost_unknown=direct_cost_unknown,
        compiler_proposal_latency_seconds=proposal_latency,
        workflow_profiling_api_cost=profiling_cost,
        workflow_profiling_api_cost_unknown=profiling_cost_unknown,
        workflow_profiling_elapsed_seconds=profiling_elapsed_seconds,
        total_compilation_api_cost=total_cost,
        total_compilation_api_cost_unknown=total_cost_unknown,
        total_compilation_elapsed_seconds=(
            proposal_latency + profiling_elapsed_seconds
        ),
    )


def _evaluation_settings(config: PipelineConfig) -> EvaluationSettings:
    return EvaluationSettings(
        split=config.split,
        sample_size=config.sample_size,
        seed=config.seed,
        repeats=config.repeats,
        warmup_runs=config.warmup_runs,
        examples=config.examples,
    )


def _compiler_history_entry(
    result: PlanResult,
    runs: list[RunRecord],
) -> dict[str, Any]:
    return {
        **result.model_dump(mode="json", exclude_none=True),
        "evaluation_runs": [
            run.model_dump(
                mode="json",
                include={"example_id", "repeat", "metrics", "latency_seconds", "error"},
                exclude_none=True,
            )
            for run in runs
        ],
    }
