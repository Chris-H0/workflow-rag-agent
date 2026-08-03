"""Orchestrate compile-only and compile-plus-profile pipeline phases."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from placement_compiler.adapters.workflows import load_workflow_driver
from placement_compiler.core.artifacts import build_artifact, write_artifact
from placement_compiler.core.catalogue import load_model_catalogue
from placement_compiler.core.models import ModelEndpoint
from placement_compiler.generation.candidates import CandidateGenerator, CompilerLLM
from placement_compiler.generation.llm import LangChainCompilerLLM
from placement_compiler.pipeline.config import PipelineConfig, PipelinePhase
from placement_compiler.profiling.runner import (
    CandidateProfiler,
    build_baseline_plan,
    load_candidate_artifact,
    write_profile_artifacts,
)
from placement_compiler.runtime.endpoint_registry import EndpointRegistry


def run_pipeline(
    config: PipelineConfig,
    *,
    phase: PipelinePhase | None = None,
    compiler_llm: CompilerLLM | None = None,
    model_factory: Callable[[ModelEndpoint], Any] | None = None,
) -> dict[str, Path | dict[str, Path]]:
    selected_phase = phase or config.default_phase
    if selected_phase == "compile":
        return {
            "candidate_artifact": compile_candidates(
                config,
                compiler_llm=compiler_llm,
            )
        }
    if selected_phase == "compile_and_profile":
        candidate_path = compile_candidates(config, compiler_llm=compiler_llm)
        return {
            "candidate_artifact": candidate_path,
            "profile_artifacts": profile_candidates(
                config, model_factory=model_factory
            ),
        }
    raise ValueError(f"unknown pipeline phase {selected_phase!r}")


def compile_candidates(
    config: PipelineConfig,
    *,
    compiler_llm: CompilerLLM | None = None,
) -> Path:
    workflow = load_workflow_driver(config.workflow).metadata()
    endpoints = load_model_catalogue(config.models)
    if compiler_llm is None:
        compiler_llm = build_compiler_llm(config)

    generator = CandidateGenerator(
        compiler_llm=compiler_llm,
        max_attempts=config.compile.max_attempts,
    )
    candidates = generator.generate(
        workflow=workflow,
        model_endpoints=endpoints,
        candidate_count=config.compile.candidates,
        priorities=config.compile.priorities,
    )
    artifact = build_artifact(
        workflow=workflow,
        model_endpoints=endpoints,
        candidates=candidates,
    )
    return write_artifact(artifact, config.candidate_artifact)


def profile_candidates(
    config: PipelineConfig,
    *,
    model_factory: Callable[[ModelEndpoint], Any] | None = None,
) -> dict[str, Path]:
    if config.profile is None:
        raise ValueError("pipeline profile section is required for compile_and_profile")

    loaded_artifact = load_candidate_artifact(config.candidate_artifact)
    endpoint_registry = EndpointRegistry(
        loaded_artifact.model_endpoints,
        model_factory=model_factory,
    )
    baseline_plan = build_baseline_plan(
        artifact=loaded_artifact,
        baseline_type=config.profile.baseline.type,
        endpoint_id=config.profile.baseline.endpoint_id,
        candidate_id=config.profile.baseline.candidate_id,
    )
    driver = load_workflow_driver(config.workflow)
    profiler = CandidateProfiler(
        candidate_artifact=loaded_artifact,
        source_candidate_artifact=config.candidate_artifact,
        endpoint_registry=endpoint_registry,
        driver=driver,
        profile=config.profile,
        baseline_plan=baseline_plan,
        quality_constraint=config.profile.quality_constraint,
        ranking=config.profile.ranking,
        workflow_name=config.workflow,
    )
    profile, runs = profiler.run()
    return write_profile_artifacts(
        profile=profile,
        runs=runs,
        output_dir=config.profile.output,
    )


def build_compiler_llm(config: PipelineConfig) -> LangChainCompilerLLM:
    model_kwargs = dict(config.compile.compiler.model_kwargs)
    if config.compile.compiler.temperature is not None:
        model_kwargs["temperature"] = config.compile.compiler.temperature
    return LangChainCompilerLLM(
        provider=config.compile.compiler.provider,
        model=config.compile.compiler.model,
        structured_output_method=config.compile.compiler.structured_output_method,
        **model_kwargs,
    )
