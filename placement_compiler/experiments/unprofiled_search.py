"""Generate compiler proposal sequences without profiling workflow executions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Literal

from placement_compiler.adapters.workflows import load_workflow_driver
from placement_compiler.core.catalogue import load_model_catalogue
from placement_compiler.core.models import (
    CompilerAttempt,
    ModelEndpoint,
    PlacementPlan,
    PlacementUnit,
    WorkflowMetadata,
)
from placement_compiler.core.validation import validate_plan
from placement_compiler.generation.candidates import (
    CandidateGenerationError,
    CandidateGenerator,
    CompilerLLM,
)
from placement_compiler.pipeline.config import PipelineConfig, load_pipeline_config
from placement_compiler.pipeline.runner import build_compiler_llm


REPO_ROOT = Path(__file__).resolve().parents[2]
MetadataMode = Literal["full", "none"]


def run_unprofiled_search(
    *,
    config: PipelineConfig,
    metadata_mode: MetadataMode,
    output_dir: Path,
    sequence_count: int = 3,
    proposal_count: int = 12,
    compiler_llm: CompilerLLM | None = None,
) -> Path:
    """Generate and checkpoint independent proposal-only compiler sequences."""
    if sequence_count <= 0 or proposal_count <= 0:
        raise ValueError("sequence_count and proposal_count must be positive")

    _load_dotenvs(config.workflow)
    driver = load_workflow_driver(config.workflow)
    workflow = driver.metadata()
    endpoints = load_model_catalogue(config.models)
    context = build_prompt_context(workflow, endpoints, metadata_mode)
    generator = CandidateGenerator(
        compiler_llm=compiler_llm or build_compiler_llm(config, endpoints),
        max_attempts=config.max_attempts,
    )

    output_path = output_dir / "proposals.json"
    artifact = _load_or_create_artifact(
        output_path=output_path,
        config=config,
        metadata_mode=metadata_mode,
        sequence_count=sequence_count,
        proposal_count=proposal_count,
        context=context,
    )

    for sequence_number in range(1, sequence_count + 1):
        sequence = artifact["sequences"][sequence_number - 1]
        proposals = [
            PlacementPlan.model_validate(item) for item in sequence["proposals"]
        ]
        prompt_assignments = [
            context.to_prompt_assignments(plan.assignments) for plan in proposals
        ]
        for iteration in range(len(proposals) + 1, proposal_count + 1):
            try:
                generated = generator.propose_with_metrics(
                    workflow=context.workflow,
                    model_endpoints=context.endpoints,
                    iteration=iteration,
                    evaluated_results=[],
                    excluded_assignments=prompt_assignments,
                )
            except CandidateGenerationError as exc:
                sequence["compiler_attempts"].extend(
                    attempt.model_dump(mode="json", exclude_none=True)
                    for attempt in exc.attempts
                )
                _update_summary(artifact)
                _write_artifact(output_path, artifact)
                raise

            sequence["compiler_attempts"].extend(
                attempt.model_dump(mode="json", exclude_none=True)
                for attempt in generated.attempts
            )
            plan = context.to_actual_plan(
                generated.plan,
                plan_id=(
                    f"sequence-{sequence_number:02d}-proposal-{iteration:02d}"
                ),
            )
            validate_plan(plan, workflow, endpoints)
            proposals.append(plan)
            prompt_assignments.append(generated.plan.assignments)
            sequence["proposals"].append(
                plan.model_dump(mode="json", exclude_none=True)
            )
            _update_summary(artifact)
            _write_artifact(output_path, artifact)
            print(
                f"Completed {config.workflow} {metadata_mode} metadata "
                f"sequence {sequence_number} proposal {iteration}: "
                f"{plan.assignments}",
                flush=True,
            )

    return output_path


class PromptContext:
    """Prompt-visible identifiers and their mapping to the actual experiment."""

    def __init__(
        self,
        *,
        workflow: WorkflowMetadata,
        endpoints: list[ModelEndpoint],
        unit_to_actual: dict[str, str],
        endpoint_to_actual: dict[str, str],
    ) -> None:
        self.workflow = workflow
        self.endpoints = endpoints
        self.unit_to_actual = unit_to_actual
        self.endpoint_to_actual = endpoint_to_actual
        self._unit_to_prompt = {
            actual: prompt for prompt, actual in unit_to_actual.items()
        }
        self._endpoint_to_prompt = {
            actual: prompt for prompt, actual in endpoint_to_actual.items()
        }

    def to_prompt_assignments(self, assignments: dict[str, str]) -> dict[str, str]:
        return {
            self._unit_to_prompt[unit]: self._endpoint_to_prompt[endpoint]
            for unit, endpoint in assignments.items()
        }

    def to_actual_plan(self, plan: PlacementPlan, *, plan_id: str) -> PlacementPlan:
        rationale = {
            self.unit_to_actual.get(unit, unit): text
            for unit, text in plan.rationale.items()
        }
        return PlacementPlan(
            id=plan_id,
            description=plan.description,
            assignments={
                self.unit_to_actual[unit]: self.endpoint_to_actual[endpoint]
                for unit, endpoint in plan.assignments.items()
            },
            rationale=rationale,
        )

    def artifact_payload(self) -> dict[str, Any]:
        return {
            "workflow": self.workflow.model_dump(mode="json", exclude_none=True),
            "model_endpoints": [
                endpoint.model_dump(mode="json", exclude_none=True)
                for endpoint in self.endpoints
            ],
            "unit_identifier_map": self.unit_to_actual,
            "endpoint_identifier_map": self.endpoint_to_actual,
        }


def build_prompt_context(
    workflow: WorkflowMetadata,
    endpoints: list[ModelEndpoint],
    metadata_mode: MetadataMode,
) -> PromptContext:
    if metadata_mode == "full":
        return PromptContext(
            workflow=workflow,
            endpoints=endpoints,
            unit_to_actual={unit.id: unit.id for unit in workflow.placement_units},
            endpoint_to_actual={endpoint.id: endpoint.id for endpoint in endpoints},
        )
    if metadata_mode != "none":
        raise ValueError(f"unknown metadata mode {metadata_mode!r}")

    unit_to_actual = {unit.id: unit.id for unit in workflow.placement_units}
    endpoint_to_actual = {endpoint.id: endpoint.id for endpoint in endpoints}
    metadata_free_workflow = WorkflowMetadata(
        workflow_id=workflow.workflow_id,
        placement_units=[PlacementUnit(id=unit.id) for unit in workflow.placement_units],
    )
    return PromptContext(
        workflow=metadata_free_workflow,
        endpoints=endpoints,
        unit_to_actual=unit_to_actual,
        endpoint_to_actual=endpoint_to_actual,
    )


def _load_or_create_artifact(
    *,
    output_path: Path,
    config: PipelineConfig,
    metadata_mode: MetadataMode,
    sequence_count: int,
    proposal_count: int,
    context: PromptContext,
) -> dict[str, Any]:
    expected = {
        "workflow": config.workflow,
        "metadata_mode": metadata_mode,
        "sequence_count": sequence_count,
        "proposal_count": proposal_count,
        "compiler": config.compiler.model_dump(mode="json", exclude_none=True),
        "prompt_context": context.artifact_payload(),
    }
    if output_path.exists():
        artifact = json.loads(output_path.read_text())
        actual = {key: artifact.get(key) for key in expected}
        if actual != expected:
            raise ValueError(f"existing proposal checkpoint does not match: {output_path}")
        return artifact

    return {
        "schema_version": "1.0",
        "kind": "unprofiled_compiler_proposals",
        **expected,
        "baseline_supplied": False,
        "evaluated_results_supplied": False,
        "prompt_builder": (
            "placement_compiler.generation.candidates."
            "CandidateGenerator._build_messages"
        ),
        "sequences": [
            {
                "sequence": sequence,
                "proposals": [],
                "compiler_attempts": [],
            }
            for sequence in range(1, sequence_count + 1)
        ],
        "compiler_metrics": {},
    }


def _update_summary(artifact: dict[str, Any]) -> None:
    attempts = [
        CompilerAttempt.model_validate(attempt)
        for sequence in artifact["sequences"]
        for attempt in sequence["compiler_attempts"]
    ]
    cost_known = all(attempt.cost_known for attempt in attempts)
    artifact["compiler_metrics"] = {
        "proposal_count": sum(
            len(sequence["proposals"]) for sequence in artifact["sequences"]
        ),
        "attempt_count": len(attempts),
        "duplicate_proposal_count": sum(
            int(attempt.duplicate_proposal) for attempt in attempts
        ),
        "validation_error_count": sum(
            len(attempt.validation_errors) for attempt in attempts
        ),
        "prompt_tokens": _sum_known(attempt.prompt_tokens for attempt in attempts),
        "completion_tokens": _sum_known(
            attempt.completion_tokens for attempt in attempts
        ),
        "total_tokens": _sum_known(attempt.total_tokens for attempt in attempts),
        "api_cost": (
            sum(attempt.api_cost or 0.0 for attempt in attempts)
            if cost_known
            else None
        ),
        "api_cost_unknown": not cost_known,
        "proposal_latency_seconds": sum(
            attempt.proposal_latency_seconds for attempt in attempts
        ),
        "workflow_execution_count": 0,
    }


def _sum_known(values) -> int | None:
    values = list(values)
    if any(value is None for value in values):
        return None
    return sum(value or 0 for value in values)


def _write_artifact(path: Path, artifact: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2) + "\n")


def _load_dotenvs(workflow: str) -> None:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env", override=True)
    load_dotenv(load_workflow_driver(workflow).root / ".env", override=True)
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--metadata-mode", choices=["full", "none"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequences", type=int, default=3)
    parser.add_argument("--proposals", type=int, default=12)
    args = parser.parse_args(argv)
    output = run_unprofiled_search(
        config=load_pipeline_config(args.config),
        metadata_mode=args.metadata_mode,
        output_dir=args.output.resolve(),
        sequence_count=args.sequences,
        proposal_count=args.proposals,
    )
    print(f"Wrote unprofiled proposals to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
