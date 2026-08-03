"""Build manifest-first workflow metadata with optional graph structure."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from placement_compiler.core.models import (
    PlacementUnit,
    PlacementUnitSpec,
    WorkflowEdge,
    WorkflowMetadata,
)


START_NODE_IDS = {"__start__", "START", "start"}
END_NODE_IDS = {"__end__", "END", "end"}


def build_workflow_metadata(
    *,
    workflow_id: str,
    placement_units: Mapping[str, PlacementUnitSpec | Mapping[str, object]],
    node_ids: Sequence[str] = (),
    edges: Sequence[WorkflowEdge] = (),
) -> WorkflowMetadata:
    """Use the manifest as truth and enrich it with available graph structure."""

    if not placement_units:
        raise ValueError("placement manifest must declare at least one placement unit")

    graph_nodes = {node_id for node_id in node_ids if not _is_sentinel(node_id)}
    normal_edges = [
        edge
        for edge in edges
        if edge.source in graph_nodes and edge.target in graph_nodes
    ]
    entry_nodes = {
        edge.target
        for edge in edges
        if edge.source in START_NODE_IDS and edge.target in graph_nodes
    }
    entry_nodes.update(graph_nodes - {edge.target for edge in normal_edges})
    terminal_nodes = {
        edge.source
        for edge in edges
        if edge.target in END_NODE_IDS and edge.source in graph_nodes
    }
    terminal_nodes.update(graph_nodes - {edge.source for edge in normal_edges})

    units = []
    for unit_id, raw_spec in sorted(placement_units.items()):
        spec = (
            raw_spec
            if isinstance(raw_spec, PlacementUnitSpec)
            else PlacementUnitSpec.model_validate(raw_spec)
        )
        units.append(
            PlacementUnit(
                id=unit_id,
                structural=unit_id in graph_nodes,
                **spec.model_dump(),
            )
        )

    return WorkflowMetadata(
        workflow_id=workflow_id,
        placement_units=units,
        edges=normal_edges,
        entry_nodes=sorted(entry_nodes),
        terminal_nodes=sorted(terminal_nodes),
        conditional_edges=[edge for edge in normal_edges if edge.conditional],
    )


def _is_sentinel(node_id: str) -> bool:
    return node_id in START_NODE_IDS or node_id in END_NODE_IDS
