"""Prompt and validate compiler LLM placement candidate proposals."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError

from placement_compiler.core.models import (
    Candidate,
    CandidateSetDraft,
    ModelEndpoint,
    WorkflowMetadata,
    WorkflowNode,
)


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
    ) -> list[Candidate]:
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
        candidates: Sequence[Candidate],
        *,
        workflow: WorkflowMetadata,
        model_endpoints: list[ModelEndpoint],
        candidate_count: int,
    ) -> tuple[list[Candidate], list[str]]:
        placement_units = {node.id: node for node in workflow.placement_units()}
        endpoints_by_id = {endpoint.id: endpoint for endpoint in model_endpoints}
        accepted: list[Candidate] = []
        errors: list[str] = []
        seen_signatures: set[tuple[tuple[str, str], ...]] = set()
        seen_candidate_ids: set[str] = set()

        if len(candidates) != candidate_count:
            errors.append(
                f"LLM returned {len(candidates)} candidates, expected {candidate_count}"
            )

        for candidate in candidates:
            candidate_errors = _candidate_errors(
                candidate,
                placement_units=placement_units,
                endpoints_by_id=endpoints_by_id,
            )
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


def _candidate_errors(
    candidate: Candidate,
    *,
    placement_units: Mapping[str, WorkflowNode],
    endpoints_by_id: Mapping[str, ModelEndpoint],
) -> list[str]:
    assignment_nodes = set(candidate.assignments)
    expected_nodes = set(placement_units)
    errors: list[str] = []

    unknown_nodes = sorted(assignment_nodes - expected_nodes)
    if unknown_nodes:
        errors.append(f"unknown or non-placement node IDs {unknown_nodes}")

    missing_nodes = sorted(expected_nodes - assignment_nodes)
    if missing_nodes:
        errors.append(f"missing assignments for {missing_nodes}")

    for node_id, endpoint_id in sorted(candidate.assignments.items()):
        endpoint = endpoints_by_id.get(endpoint_id)
        if endpoint is None:
            errors.append(f"{node_id} references unknown endpoint {endpoint_id!r}")
            continue
        node = placement_units.get(node_id)
        if node is None:
            continue
        if node.tool_use and not endpoint.tool_calling:
            errors.append(
                f"{node_id} requires tool calling but {endpoint_id!r} does not support it"
            )
        if node.structured_output_required and not endpoint.structured_output:
            errors.append(
                f"{node_id} requires structured output but {endpoint_id!r} does not support it"
            )
        if (
            node.required_context_window is not None
            and endpoint.context_window is not None
            and endpoint.context_window < node.required_context_window
        ):
            errors.append(
                f"{node_id} requires context window {node.required_context_window} "
                f"but {endpoint_id!r} only declares {endpoint.context_window}"
            )
    return errors
