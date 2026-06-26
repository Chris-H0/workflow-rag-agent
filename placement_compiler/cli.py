"""Command-line interface for unified placement compiler pipeline runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from placement_compiler.generation.candidates import CompilerLLM
from placement_compiler.pipeline.config import PipelinePhase, load_pipeline_config
from placement_compiler.pipeline.runner import run_pipeline
from placement_compiler.adapters.repository_workflows import (
    REPO_ROOT,
    WORKFLOW_SPECS,
)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        output_paths = pipeline_command(args)
        print_pipeline_outputs(output_paths)
        return 0
    parser.error("unknown command")
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m placement_compiler",
        description="Run model placement compilation and profiling.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
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


def _load_dotenvs(workflow: str) -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    load_dotenv(REPO_ROOT / ".env", override=True)
    spec = WORKFLOW_SPECS.get(workflow)
    if spec is not None:
        load_dotenv(spec.source_dir / ".env", override=True)
