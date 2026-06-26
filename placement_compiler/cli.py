"""Command-line interface for compile, profile, and unified pipeline runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from placement_compiler.core.artifacts import build_artifact, write_artifact
from placement_compiler.core.catalogue import load_model_catalogue, load_node_registry
from placement_compiler.generation.candidates import CandidateGenerator, CompilerLLM
from placement_compiler.generation.config import ResolvedPlacementRunConfig, load_run_config
from placement_compiler.generation.llm import LangChainCompilerLLM
from placement_compiler.runtime.endpoint_registry import EndpointRegistry
from placement_compiler.profiling.config import load_profile_run_config
from placement_compiler.adapters.profile_workflows import build_repository_profile_adapters
from placement_compiler.profiling.runner import (
    CandidateProfiler,
    build_baseline_plan,
    load_candidate_artifact,
    write_profile_artifacts,
)
from placement_compiler.pipeline.config import PipelinePhase, load_pipeline_config
from placement_compiler.pipeline.runner import run_pipeline
from placement_compiler.adapters.repository_workflows import (
    REPO_ROOT,
    WORKFLOW_SPECS,
    load_existing_workflow_metadata,
)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "generate":
        output_path = generate_command(args)
        print(f"Wrote placement candidates to {output_path}")
        return 0
    if args.command == "profile":
        output_paths = profile_command(args)
        print(f"Wrote placement profile to {output_paths['profile']}")
        return 0
    if args.command == "run":
        output_paths = pipeline_command(args)
        print_pipeline_outputs(output_paths)
        return 0
    parser.error("unknown command")
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m placement_compiler",
        description="Generate compile-time model placement candidates.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument(
        "--config",
        required=True,
        help="Path to the placement run YAML/JSON config.",
    )
    profile = subparsers.add_parser("profile")
    profile.add_argument(
        "--config",
        required=True,
        help="Path to the profiling run YAML/JSON config.",
    )
    run = subparsers.add_parser("run")
    run.add_argument(
        "--config",
        required=True,
        help="Path to the unified compile/profile pipeline YAML/JSON config.",
    )
    run.add_argument(
        "--phase",
        choices=["compile", "compile_and_profile"],
        help="Override the config default_phase.",
    )
    return parser


def generate_command(
    args: argparse.Namespace,
    *,
    compiler_llm: CompilerLLM | None = None,
) -> Path:
    return generate_from_config(args.config, compiler_llm=compiler_llm)


def generate_from_config(
    config_path: str | Path,
    *,
    compiler_llm: CompilerLLM | None = None,
) -> Path:
    config = load_run_config(config_path)
    _load_dotenvs(config.workflow)
    registry = load_node_registry(config.metadata) if config.metadata else None
    workflow = load_existing_workflow_metadata(
        config.workflow,
        registry_overrides=registry,
    )
    endpoints = load_model_catalogue(config.models)
    if compiler_llm is None:
        compiler_llm = _build_compiler_llm(config)

    generator = CandidateGenerator(
        compiler_llm=compiler_llm,
        max_attempts=config.max_attempts,
    )
    candidates = generator.generate(
        workflow=workflow,
        model_endpoints=endpoints,
        candidate_count=config.candidates,
        priorities=config.priorities,
    )
    artifact = build_artifact(
        workflow=workflow,
        model_endpoints=endpoints,
        candidates=candidates,
    )
    return write_artifact(artifact, config.output)


def profile_command(args: argparse.Namespace) -> dict[str, Path]:
    return profile_from_config(args.config)


def profile_from_config(config_path: str | Path) -> dict[str, Path]:
    config = load_profile_run_config(config_path)
    _load_dotenvs(config.workflow)
    loaded_artifact = load_candidate_artifact(config.candidate_artifact)
    endpoint_registry = EndpointRegistry(loaded_artifact.artifact.model_endpoints)
    baseline_plan = build_baseline_plan(
        artifact=loaded_artifact.artifact,
        baseline_type=config.baseline.type,
        endpoint_id=config.baseline.endpoint_id,
        candidate_id=config.baseline.candidate_id,
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
        quality_constraint=config.quality_constraint,
        ranking=config.ranking,
        workflow_name=config.workflow,
    )
    profile, runs = profiler.run()
    return write_profile_artifacts(profile=profile, runs=runs, output_dir=config.output)


def pipeline_command(args: argparse.Namespace) -> dict[str, Path | dict[str, Path]]:
    return pipeline_from_config(args.config, phase=args.phase)


def pipeline_from_config(
    config_path: str | Path,
    *,
    phase: PipelinePhase | None = None,
    compiler_llm: CompilerLLM | None = None,
) -> dict[str, Path | dict[str, Path]]:
    config = load_pipeline_config(config_path)
    _load_dotenvs(config.workflow)
    return run_pipeline(config, phase=phase, compiler_llm=compiler_llm)


def print_pipeline_outputs(output_paths: dict[str, Path | dict[str, Path]]) -> None:
    candidate_path = output_paths.get("candidate_artifact")
    if candidate_path:
        print(f"Wrote placement candidates to {candidate_path}")
    profile_artifacts = output_paths.get("profile_artifacts")
    if isinstance(profile_artifacts, dict):
        print(f"Wrote placement profile to {profile_artifacts['profile']}")


def _build_compiler_llm(config: ResolvedPlacementRunConfig) -> LangChainCompilerLLM:
    model_kwargs = dict(config.compiler.model_kwargs)
    if config.compiler.temperature is not None:
        model_kwargs["temperature"] = config.compiler.temperature
    return LangChainCompilerLLM(
        provider=config.compiler.provider,
        model=config.compiler.model,
        structured_output_method=config.compiler.structured_output_method,
        **model_kwargs,
    )


def _load_dotenvs(workflow: str) -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    load_dotenv(REPO_ROOT / ".env", override=True)
    spec = WORKFLOW_SPECS.get(workflow)
    if spec is not None:
        load_dotenv(spec.source_dir / ".env", override=True)
