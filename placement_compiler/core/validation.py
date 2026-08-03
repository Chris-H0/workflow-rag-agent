"""Validation shared by candidate generation, artifacts, and runtime resolution."""

from __future__ import annotations

from collections.abc import Sequence

from placement_compiler.core.models import ModelEndpoint, PlacementPlan, WorkflowMetadata


class PlanValidationError(ValueError):
    pass


def validate_plan(
    plan: PlacementPlan,
    workflow: WorkflowMetadata,
    endpoints: Sequence[ModelEndpoint],
) -> PlacementPlan:
    units = {node.id: node for node in workflow.placement_units()}
    endpoints_by_id = {endpoint.id: endpoint for endpoint in endpoints}
    assigned = set(plan.assignments)
    expected = set(units)
    errors: list[str] = []

    unknown = sorted(assigned - expected)
    if unknown:
        errors.append(f"unknown or non-placement unit IDs {unknown}")

    missing = sorted(expected - assigned)
    if missing:
        errors.append(f"missing assignments for {missing}")

    for unit_id, endpoint_id in sorted(plan.assignments.items()):
        unit = units.get(unit_id)
        endpoint = endpoints_by_id.get(endpoint_id)
        if endpoint is None:
            errors.append(f"{unit_id} references unknown endpoint {endpoint_id!r}")
            continue
        if unit is None:
            continue
        if unit.tool_use and not endpoint.tool_calling:
            errors.append(
                f"{unit_id} requires tool calling but {endpoint_id!r} does not support it"
            )
        if unit.structured_output_required and not endpoint.structured_output:
            errors.append(
                f"{unit_id} requires structured output but {endpoint_id!r} does not support it"
            )
        if (
            unit.required_context_window is not None
            and endpoint.context_window is not None
            and endpoint.context_window < unit.required_context_window
        ):
            errors.append(
                f"{unit_id} requires context window {unit.required_context_window} "
                f"but {endpoint_id!r} only declares {endpoint.context_window}"
            )

    if errors:
        raise PlanValidationError("; ".join(errors))
    return plan
