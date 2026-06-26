"""Orchestrate compile-only and compile-plus-profile pipeline phases."""

from __future__ import annotations

from pathlib import Path

from placement_compiler.adapters.profile_workflows import build_repository_profile_adapters
from placement_compiler.adapters.repository_workflows import load_existing_workflow_metadata
from placement_compiler.core.artifacts import build_artifact, write_artifact
from placement_compiler.core.catalogue import load_model_catalogue, load_node_registry
from placement_compiler.generation.candidates import CandidateGenerator, CompilerLLM
from placement_compiler.generation.llm import LangChainCompilerLLM
from placement_compiler.pipeline.config import PipelinePhase, ResolvedPipelineConfig
from placement_compiler.profiling.runner import (
    CandidateProfiler,
    build_baseline_plan,
    load_candidate_artifact,
    write_profile_artifacts,
)
from placement_compiler.runtime.endpoint_registry import EndpointRegistry


def run_pipeline(
    config: ResolvedPipelineConfig,
    *,
    phase: PipelinePhase | None = None,
    compiler_llm: CompilerLLM | None = None,
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
            "profile_artifacts": profile_candidates(config),
        }
    raise ValueError(f"unknown pipeline phase {selected_phase!r}")


def compile_candidates(
    config: ResolvedPipelineConfig,
    *,
    compiler_llm: CompilerLLM | None = None,
) -> Path:
    registry = load_node_registry(config.metadata) if config.metadata else None
    workflow = load_existing_workflow_metadata(
        config.workflow,
        registry_overrides=registry,
    )
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


def profile_candidates(config: ResolvedPipelineConfig) -> dict[str, Path]:
    if config.profile is None:
        raise ValueError("pipeline profile section is required for compile_and_profile")

    loaded_artifact = load_candidate_artifact(config.candidate_artifact)
    endpoint_registry = EndpointRegistry(loaded_artifact.artifact.model_endpoints)
    baseline_plan = build_baseline_plan(
        artifact=loaded_artifact.artifact,
        baseline_type=config.profile.baseline.type,
        endpoint_id=config.profile.baseline.endpoint_id,
        candidate_id=config.profile.baseline.candidate_id,
    )
    evaluation_adapter, runtime_adapter = build_repository_profile_adapters(
        workflow=config.workflow,
        candidate_artifact=loaded_artifact.artifact,
        endpoint_registry=endpoint_registry,
    )
    profiler = CandidateProfiler(
        candidate_artifact=loaded_artifact,
        evaluation_adapter=evaluation_adapter,
        runtime_adapter=runtime_adapter,
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


def build_compiler_llm(config: ResolvedPipelineConfig) -> LangChainCompilerLLM:
    model_kwargs = dict(config.compile.compiler.model_kwargs)
    if config.compile.compiler.temperature is not None:
        model_kwargs["temperature"] = config.compile.compiler.temperature
    return LangChainCompilerLLM(
        provider=config.compile.compiler.provider,
        model=config.compile.compiler.model,
        structured_output_method=config.compile.compiler.structured_output_method,
        **model_kwargs,
    )
