"""Propose one measured-informed model placement at a time."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError

from placement_compiler.core.models import (
    CompilerAttempt,
    ModelEndpoint,
    PlacementPlan,
    PlacementProposal,
    WorkflowMetadata,
)
from placement_compiler.core.validation import PlanValidationError, validate_plan


class CandidateGenerationError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        attempts: Sequence[CompilerAttempt] = (),
    ) -> None:
        super().__init__(message)
        self.attempts = list(attempts)


CompilerOutput = PlacementProposal | Mapping[str, Any] | str


@dataclass(frozen=True)
class CompilerGeneration:
    """A compiler response with provider usage metadata when available."""

    output: CompilerOutput | None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    api_cost: float | None = None
    cost_known: bool = False
    error: str | None = None


@dataclass(frozen=True)
class CandidateGenerationResult:
    plan: PlacementPlan
    attempts: list[CompilerAttempt]


class CompilerLLM(Protocol):
    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        output_schema: type[PlacementProposal],
    ) -> CompilerOutput | CompilerGeneration:
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
        return self.propose_with_metrics(
            workflow=workflow,
            model_endpoints=model_endpoints,
            iteration=iteration,
            evaluated_results=evaluated_results,
        ).plan

    def propose_with_metrics(
        self,
        *,
        workflow: WorkflowMetadata,
        model_endpoints: list[ModelEndpoint],
        iteration: int,
        evaluated_results: Sequence[Mapping[str, Any]],
    ) -> CandidateGenerationResult:
        if not workflow.placement_units:
            raise ValueError("workflow has no declared LLM placement units")
        if not model_endpoints:
            raise ValueError("model endpoint catalogue must not be empty")

        previous_errors: list[str] = []
        attempts: list[CompilerAttempt] = []
        for attempt_number in range(1, self.max_attempts + 1):
            started = time.perf_counter()
            try:
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
                generation = _normalise_generation(raw)
            except Exception as exc:
                attempts.append(
                    CompilerAttempt(
                        iteration=iteration,
                        attempt=attempt_number,
                        accepted=False,
                        proposal_latency_seconds=time.perf_counter() - started,
                        generation_error=repr(exc),
                    )
                )
                raise CandidateGenerationError(
                    f"compiler LLM call failed: {exc}", attempts=attempts
                ) from exc

            latency = time.perf_counter() - started
            if generation.error is not None:
                previous_errors = [generation.error]
                attempts.append(
                    _attempt_record(
                        generation,
                        iteration=iteration,
                        attempt=attempt_number,
                        latency=latency,
                        generation_error=generation.error,
                    )
                )
                continue

            try:
                proposal = _coerce_proposal(generation.output)
            except CandidateGenerationError as exc:
                previous_errors = [str(exc)]
                attempts.append(
                    _attempt_record(
                        generation,
                        iteration=iteration,
                        attempt=attempt_number,
                        latency=latency,
                        generation_error=str(exc),
                    )
                )
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
            duplicate = "duplicates an evaluated placement" in errors
            attempts.append(
                _attempt_record(
                    generation,
                    iteration=iteration,
                    attempt=attempt_number,
                    latency=latency,
                    accepted=not errors,
                    validation_errors=errors,
                    duplicate_proposal=duplicate,
                )
            )
            if not errors:
                return CandidateGenerationResult(plan=candidate, attempts=attempts)
            previous_errors = errors

        raise CandidateGenerationError(
            "compiler LLM did not produce a valid new placement: "
            + "; ".join(previous_errors),
            attempts=attempts,
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
    raw: CompilerOutput | None,
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


def _normalise_generation(
    raw: CompilerOutput | CompilerGeneration,
) -> CompilerGeneration:
    if isinstance(raw, CompilerGeneration):
        return raw
    return CompilerGeneration(output=raw)


def _attempt_record(
    generation: CompilerGeneration,
    *,
    iteration: int,
    attempt: int,
    latency: float,
    accepted: bool = False,
    validation_errors: Sequence[str] = (),
    duplicate_proposal: bool = False,
    generation_error: str | None = None,
) -> CompilerAttempt:
    return CompilerAttempt(
        iteration=iteration,
        attempt=attempt,
        accepted=accepted,
        proposal_latency_seconds=latency,
        prompt_tokens=generation.prompt_tokens,
        completion_tokens=generation.completion_tokens,
        total_tokens=generation.total_tokens,
        api_cost=generation.api_cost,
        cost_known=generation.cost_known,
        validation_errors=list(validation_errors),
        duplicate_proposal=duplicate_proposal,
        generation_error=generation_error,
    )
