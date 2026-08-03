"""Run the single sequential model-routing search."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from placement_compiler.adapters.workflows import load_workflow_driver
from placement_compiler.core.catalogue import load_model_catalogue
from placement_compiler.core.models import ModelEndpoint
from placement_compiler.generation.candidates import CandidateGenerator, CompilerLLM
from placement_compiler.generation.llm import LangChainCompilerLLM
from placement_compiler.pipeline.config import PipelineConfig
from placement_compiler.profiling.models import (
    PlanResult,
    EvaluationSettings,
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
    baseline_result, baseline_runs = evaluator.evaluate(baseline)
    results = [baseline_result]
    runs = list(baseline_runs)
    compiler_history = [_compiler_history_entry(baseline_result, baseline_runs)]

    generator = CandidateGenerator(
        compiler_llm=compiler_llm or build_compiler_llm(config),
        max_attempts=config.max_attempts,
    )
    for iteration in range(1, config.iterations + 1):
        candidate = generator.propose(
            workflow=workflow,
            model_endpoints=endpoints,
            iteration=iteration,
            evaluated_results=compiler_history,
        )
        result, candidate_runs = evaluator.evaluate(candidate)
        results.append(result)
        runs.extend(candidate_runs)
        compiler_history.append(_compiler_history_entry(result, candidate_runs))

    artifact = SearchArtifact(
        workflow_id=workflow.workflow_id,
        workflow=config.workflow,
        search_config=config.model_dump(mode="json", exclude_none=True),
        primary_metric=driver.primary_metric,
        selected_example_ids=selected_example_ids,
        results=results,
    )
    return write_search_artifacts(
        artifact=artifact,
        runs=runs,
        output_dir=config.output,
    )


def build_compiler_llm(config: PipelineConfig) -> LangChainCompilerLLM:
    model_kwargs = dict(config.compiler.model_kwargs)
    if config.compiler.temperature is not None:
        model_kwargs["temperature"] = config.compiler.temperature
    return LangChainCompilerLLM(
        provider=config.compiler.provider,
        model=config.compiler.model,
        structured_output_method=config.compiler.structured_output_method,
        **model_kwargs,
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
