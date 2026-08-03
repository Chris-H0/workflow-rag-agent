"""Propose one measured-informed model placement at a time."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError

from placement_compiler.core.models import (
    ModelEndpoint,
    PlacementPlan,
    PlacementProposal,
    WorkflowMetadata,
)
from placement_compiler.core.validation import PlanValidationError, validate_plan


class CandidateGenerationError(RuntimeError):
    pass


class CompilerLLM(Protocol):
    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        output_schema: type[PlacementProposal],
    ) -> PlacementProposal | Mapping[str, Any] | str:
        """Return one complete placement proposal."""


@dataclass
class CandidateGenerator:
    compiler_llm: CompilerLLM
    max_attempts: int = 2

    def propose(
        self,
        *,
        workflow: WorkflowMetadata,
        model_endpoints: list[ModelEndpoint],
        iteration: int,
        evaluated_results: Sequence[Mapping[str, Any]],
    ) -> PlacementPlan:
        if not workflow.placement_units:
            raise ValueError("workflow has no declared LLM placement units")
        if not model_endpoints:
            raise ValueError("model endpoint catalogue must not be empty")

        previous_errors: list[str] = []
        for _ in range(self.max_attempts):
            raw = self.compiler_llm.generate(
                self._build_messages(
                    workflow=workflow,
                    model_endpoints=model_endpoints,
                    iteration=iteration,
                    evaluated_results=evaluated_results,
                    previous_errors=previous_errors,
                ),
                PlacementProposal,
            )
            try:
                proposal = _coerce_proposal(raw)
            except CandidateGenerationError as exc:
                previous_errors = [str(exc)]
                continue

            candidate = PlacementPlan(
                id=f"candidate-{iteration}",
                description=proposal.description,
                assignments=proposal.assignments,
                rationale=proposal.rationale,
            )
            errors = _candidate_errors(
                candidate,
                workflow=workflow,
                model_endpoints=model_endpoints,
                evaluated_results=evaluated_results,
            )
            if not errors:
                return candidate
            previous_errors = errors

        raise CandidateGenerationError(
            "compiler LLM did not produce a valid new placement: "
            + "; ".join(previous_errors)
        )

    def _build_messages(
        self,
        *,
        workflow: WorkflowMetadata,
        model_endpoints: list[ModelEndpoint],
        iteration: int,
        evaluated_results: Sequence[Mapping[str, Any]],
        previous_errors: Sequence[str],
    ) -> list[dict[str, str]]:
        payload = {
            "iteration": iteration,
            "workflow": workflow.model_dump(mode="json", exclude_none=True),
            "model_endpoints": [
                endpoint.model_dump(mode="json", exclude_none=True)
                for endpoint in model_endpoints
            ],
            "evaluated_results": list(evaluated_results),
            "previous_validation_errors": list(previous_errors),
        }
        return [
            {
                "role": "system",
                "content": (
                    "You optimise model routing for a fixed workflow. Propose one new, "
                    "complete placement using the measured evaluation results from every "
                    "previous run. You may change any number of placement units. Explore "
                    "quality, cloud cost, and latency trade-offs, but do not repeat an "
                    "evaluated assignment map. Respect endpoint tool-calling, structured-"
                    "output, and context-window capabilities. Do not select a final winner."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Return one complete, previously unevaluated placement using the "
                    "provided structured schema. The evaluated_results field contains the "
                    "baseline and all earlier aggregate and per-example evaluation metrics.\n\n"
                    + json.dumps(payload, indent=2)
                ),
            },
        ]


def _candidate_errors(
    candidate: PlacementPlan,
    *,
    workflow: WorkflowMetadata,
    model_endpoints: list[ModelEndpoint],
    evaluated_results: Sequence[Mapping[str, Any]],
) -> list[str]:
    errors: list[str] = []
    try:
        validate_plan(candidate, workflow, model_endpoints)
    except PlanValidationError as exc:
        errors.append(str(exc))

    signature = tuple(sorted(candidate.assignments.items()))
    evaluated_signatures = {
        tuple(sorted(result["plan"]["assignments"].items()))
        for result in evaluated_results
    }
    if signature in evaluated_signatures:
        errors.append("duplicates an evaluated placement")
    return errors


def _coerce_proposal(
    raw: PlacementProposal | Mapping[str, Any] | str,
) -> PlacementProposal:
    if isinstance(raw, PlacementProposal):
        return raw
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CandidateGenerationError(
                f"invalid structured LLM output: {exc}"
            ) from exc
    if isinstance(raw, BaseModel):
        raw = raw.model_dump()
    try:
        return PlacementProposal.model_validate(raw)
    except (ValidationError, TypeError) as exc:
        raise CandidateGenerationError(f"invalid structured LLM output: {exc}") from exc
