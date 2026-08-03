"""Prompt and validate compiler LLM placement candidate proposals."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError

from placement_compiler.core.models import (
    CandidateSetDraft,
    ModelEndpoint,
    PlacementPlan,
    WorkflowMetadata,
)
from placement_compiler.core.validation import PlanValidationError, validate_plan


class CandidateGenerationError(RuntimeError):
    pass


class CompilerLLM(Protocol):
    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        output_schema: type[CandidateSetDraft],
    ) -> CandidateSetDraft | Mapping[str, Any] | str:
        """Return structured candidate proposals from an LLM."""


@dataclass
class CandidateGenerator:
    compiler_llm: CompilerLLM
    max_attempts: int = 2

    def generate(
        self,
        *,
        workflow: WorkflowMetadata,
        model_endpoints: list[ModelEndpoint],
        candidate_count: int,
        priorities: Sequence[str] | None = None,
    ) -> list[PlacementPlan]:
        if candidate_count <= 0:
            raise ValueError("candidate_count must be positive")
        if not workflow.placement_units():
            raise ValueError("workflow has no declared LLM placement units")
        if not model_endpoints:
            raise ValueError("model endpoint catalogue must not be empty")

        previous_errors: list[str] = []
        for _ in range(self.max_attempts):
            raw = self.compiler_llm.generate(
                self._build_messages(
                    workflow=workflow,
                    model_endpoints=model_endpoints,
                    candidate_count=candidate_count,
                    priorities=priorities or [],
                    previous_errors=previous_errors,
                ),
                CandidateSetDraft,
            )
            draft = _coerce_candidate_set(raw)
            candidates, errors = self._validate_candidates(
                draft.candidates,
                workflow=workflow,
                model_endpoints=model_endpoints,
                candidate_count=candidate_count,
            )
            if len(candidates) == candidate_count and not errors:
                return candidates
            previous_errors = errors

        raise CandidateGenerationError(
            "compiler LLM did not produce a valid candidate set: "
            + "; ".join(previous_errors)
        )

    def _build_messages(
        self,
        *,
        workflow: WorkflowMetadata,
        model_endpoints: list[ModelEndpoint],
        candidate_count: int,
        priorities: Sequence[str],
        previous_errors: Sequence[str],
    ) -> list[dict[str, str]]:
        placement_units = [
            node.model_dump(mode="json", exclude_none=True)
            for node in workflow.placement_units()
        ]
        endpoint_data = [
            endpoint.model_dump(mode="json", exclude_none=True)
            for endpoint in model_endpoints
        ]
        workflow_summary = {
            "workflow_id": workflow.workflow_id,
            "entry_nodes": workflow.entry_nodes,
            "terminal_nodes": workflow.terminal_nodes,
            "placement_units": placement_units,
            "edges": [
                edge.model_dump(mode="json", exclude_none=True)
                for edge in workflow.edges
            ],
        }
        user_payload = {
            "candidate_count": candidate_count,
            "priorities": list(priorities),
            "workflow": workflow_summary,
            "model_endpoints": endpoint_data,
            "previous_validation_errors": list(previous_errors),
        }
        return [
            {
                "role": "system",
                "content": (
                    "You are a compile-time workflow placement planner. "
                    "Propose diverse complete model-placement configurations. "
                    "Do not modify workflow code and do not choose a final winner. "
                    "Every LLM placement unit must be assigned exactly one valid endpoint. "
                    "Respect endpoint capabilities for tool calling and structured output."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Return exactly the requested number of candidates using the "
                    "provided structured schema. Prefer a mix of quality-oriented, "
                    "balanced local/cloud, and local-first or low-cloud-cost trade-offs "
                    "where the endpoint catalogue supports them.\n\n"
                    + json.dumps(user_payload, indent=2)
                ),
            },
        ]

    def _validate_candidates(
        self,
        candidates: Sequence[PlacementPlan],
        *,
        workflow: WorkflowMetadata,
        model_endpoints: list[ModelEndpoint],
        candidate_count: int,
    ) -> tuple[list[PlacementPlan], list[str]]:
        accepted: list[PlacementPlan] = []
        errors: list[str] = []
        seen_signatures: set[tuple[tuple[str, str], ...]] = set()
        seen_candidate_ids: set[str] = set()

        if len(candidates) != candidate_count:
            errors.append(
                f"LLM returned {len(candidates)} candidates, expected {candidate_count}"
            )

        for candidate in candidates:
            try:
                validate_plan(candidate, workflow, model_endpoints)
                candidate_errors: list[str] = []
            except PlanValidationError as exc:
                candidate_errors = [str(exc)]
            if candidate.source != "candidate":
                candidate_errors.append("source must be 'candidate'")
            signature = tuple(sorted(candidate.assignments.items()))
            if signature in seen_signatures:
                candidate_errors.append("duplicates another candidate configuration")
            if candidate.id in seen_candidate_ids:
                candidate_errors.append("duplicates another candidate ID")

            if candidate_errors:
                errors.append(f"{candidate.id}: {', '.join(candidate_errors)}")
                continue

            seen_signatures.add(signature)
            seen_candidate_ids.add(candidate.id)
            accepted.append(candidate)

        if len(accepted) != candidate_count:
            errors.append(
                f"expected {candidate_count} valid candidates, got {len(accepted)}"
            )
        return accepted, errors


def _coerce_candidate_set(raw: CandidateSetDraft | Mapping[str, Any] | str) -> CandidateSetDraft:
    if isinstance(raw, CandidateSetDraft):
        return raw
    if isinstance(raw, str):
        raw = json.loads(raw)
    if isinstance(raw, BaseModel):
        raw = raw.model_dump()
    try:
        return CandidateSetDraft.model_validate(raw)
    except ValidationError as exc:
        raise CandidateGenerationError(f"invalid structured LLM output: {exc}") from exc
