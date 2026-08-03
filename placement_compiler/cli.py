"""Command-line interface for unified placement compiler pipeline runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from placement_compiler.generation.candidates import CompilerLLM
from placement_compiler.pipeline.config import load_pipeline_config
from placement_compiler.pipeline.runner import run_pipeline
from placement_compiler.adapters.workflows import load_workflow_driver


REPO_ROOT = Path(__file__).resolve().parents[1]


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
        description="Run sequential model-routing search.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument(
        "--config",
        required=True,
        help="Path to the model-routing search YAML/JSON config.",
    )
    return parser


def pipeline_command(args: argparse.Namespace) -> dict[str, Path]:
    return pipeline_from_config(args.config)


def pipeline_from_config(
    config_path: str | Path,
    *,
    compiler_llm: CompilerLLM | None = None,
) -> dict[str, Path]:
    config = load_pipeline_config(config_path)
    _load_dotenvs(config.workflow)
    return run_pipeline(config, compiler_llm=compiler_llm)


def print_pipeline_outputs(output_paths: dict[str, Path]) -> None:
    print(f"Wrote routing results to {output_paths['results']}")


def _load_dotenvs(workflow: str) -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    load_dotenv(REPO_ROOT / ".env", override=True)
    load_dotenv(load_workflow_driver(workflow).root / ".env", override=True)
