from __future__ import annotations

import argparse
from pathlib import Path

from placement_compiler.artifacts import build_artifact, write_artifact
from placement_compiler.candidates import CandidateGenerator, CompilerLLM
from placement_compiler.catalogue import load_model_catalogue, load_node_registry
from placement_compiler.config import ResolvedPlacementRunConfig, load_run_config
from placement_compiler.llm import LangChainCompilerLLM
from placement_compiler.workflows import (
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


def _build_compiler_llm(config: ResolvedPlacementRunConfig) -> LangChainCompilerLLM:
    model_kwargs = dict(config.compiler.model_kwargs)
    if config.compiler.temperature is not None:
        model_kwargs["temperature"] = config.compiler.temperature
    return LangChainCompilerLLM(
        provider=config.compiler.provider,
        model=config.compiler.model,
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
