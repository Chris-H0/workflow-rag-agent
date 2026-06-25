from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence

from placement_compiler.models import (
    NodeRegistryMetadata,
    Stage,
    WorkflowEdge,
    WorkflowMetadata,
    WorkflowNode,
)

START_NODE_IDS = {"__start__", "START", "start"}
END_NODE_IDS = {"__end__", "END", "end"}


def normalise_registry(
    registry: Mapping[str, NodeRegistryMetadata | Mapping[str, object]] | None,
) -> dict[str, NodeRegistryMetadata]:
    if not registry:
        return {}
    return {
        node_id: (
            metadata
            if isinstance(metadata, NodeRegistryMetadata)
            else NodeRegistryMetadata.model_validate(metadata)
        )
        for node_id, metadata in registry.items()
    }


def build_workflow_metadata(
    *,
    workflow_id: str,
    node_ids: Sequence[str],
    edges: Sequence[WorkflowEdge],
    registry: Mapping[str, NodeRegistryMetadata | Mapping[str, object]] | None = None,
) -> WorkflowMetadata:
    """Build serialisable workflow metadata from normalised graph primitives."""

    registry_by_node = normalise_registry(registry)
    normal_nodes = sorted(
        node_id for node_id in set(node_ids) if not _is_sentinel(node_id)
    )
    normal_node_set = set(normal_nodes)

    predecessors: dict[str, set[str]] = {node_id: set() for node_id in normal_nodes}
    successors: dict[str, set[str]] = {node_id: set() for node_id in normal_nodes}
    entry_nodes: set[str] = set()
    terminal_nodes: set[str] = set()
    normal_edges: list[WorkflowEdge] = []

    for edge in edges:
        source_is_start = _is_start(edge.source)
        target_is_end = _is_end(edge.target)

        if source_is_start and edge.target in normal_node_set:
            entry_nodes.add(edge.target)
            continue
        if target_is_end and edge.source in normal_node_set:
            terminal_nodes.add(edge.source)
            continue
        if edge.source not in normal_node_set or edge.target not in normal_node_set:
            continue

        normal_edges.append(edge)
        successors[edge.source].add(edge.target)
        predecessors[edge.target].add(edge.source)

    for node_id in normal_nodes:
        if not predecessors[node_id]:
            entry_nodes.add(node_id)
        if not successors[node_id]:
            terminal_nodes.add(node_id)

    cycle_nodes = _find_cycle_nodes(normal_nodes, successors)
    distances_from_entry = _shortest_distances(sorted(entry_nodes), successors)
    max_distance = max(distances_from_entry.values(), default=0)

    nodes: list[WorkflowNode] = []
    for node_id in normal_nodes:
        manual = registry_by_node.get(node_id, NodeRegistryMetadata())
        node = WorkflowNode(
            id=node_id,
            name=node_id,
            **manual.model_dump(),
            predecessors=sorted(predecessors[node_id]),
            successors=sorted(successors[node_id]),
            in_degree=len(predecessors[node_id]),
            out_degree=len(successors[node_id]),
            reachable_downstream=_reachable_downstream(node_id, successors),
            in_cycle=node_id in cycle_nodes,
            stage=_approximate_stage(
                node_id=node_id,
                entry_nodes=entry_nodes,
                terminal_nodes=terminal_nodes,
                distances=distances_from_entry,
                max_distance=max_distance,
            ),
        )
        nodes.append(node)

    for node_id in sorted(set(registry_by_node) - set(normal_nodes)):
        if _is_sentinel(node_id):
            continue
        manual = registry_by_node[node_id]
        nodes.append(
            WorkflowNode(
                id=node_id,
                name=node_id,
                **manual.model_dump(),
                structural=False,
                stage="middle",
            )
        )

    return WorkflowMetadata(
        workflow_id=workflow_id,
        nodes=nodes,
        edges=normal_edges,
        entry_nodes=sorted(entry_nodes),
        terminal_nodes=sorted(terminal_nodes),
        conditional_edges=[edge for edge in normal_edges if edge.conditional],
    )


def _is_start(node_id: str) -> bool:
    return node_id in START_NODE_IDS


def _is_end(node_id: str) -> bool:
    return node_id in END_NODE_IDS


def _is_sentinel(node_id: str) -> bool:
    return _is_start(node_id) or _is_end(node_id)


def _reachable_downstream(
    node_id: str,
    successors: Mapping[str, set[str]],
) -> list[str]:
    seen: set[str] = set()
    stack = list(successors[node_id])
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(successors[current] - seen)
    seen.discard(node_id)
    return sorted(seen)


def _shortest_distances(
    start_nodes: Sequence[str],
    successors: Mapping[str, set[str]],
) -> dict[str, int]:
    distances: dict[str, int] = {}
    queue: deque[tuple[str, int]] = deque((node_id, 0) for node_id in start_nodes)
    while queue:
        node_id, distance = queue.popleft()
        if node_id in distances:
            continue
        distances[node_id] = distance
        for successor in sorted(successors[node_id]):
            queue.append((successor, distance + 1))
    return distances


def _approximate_stage(
    *,
    node_id: str,
    entry_nodes: set[str],
    terminal_nodes: set[str],
    distances: Mapping[str, int],
    max_distance: int,
) -> Stage:
    if node_id in entry_nodes:
        return "entry"
    if node_id in terminal_nodes:
        return "terminal"
    if max_distance <= 1:
        return "middle"

    ratio = distances.get(node_id, max_distance) / max_distance
    if ratio <= 0.34:
        return "early"
    if ratio <= 0.67:
        return "middle"
    return "late"


def _find_cycle_nodes(
    node_ids: Sequence[str],
    successors: Mapping[str, set[str]],
) -> set[str]:
    index = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    cycle_nodes: set[str] = set()

    def strongconnect(node_id: str) -> None:
        nonlocal index
        indices[node_id] = index
        lowlinks[node_id] = index
        index += 1
        stack.append(node_id)
        on_stack.add(node_id)

        for successor in successors[node_id]:
            if successor not in indices:
                strongconnect(successor)
                lowlinks[node_id] = min(lowlinks[node_id], lowlinks[successor])
            elif successor in on_stack:
                lowlinks[node_id] = min(lowlinks[node_id], indices[successor])

        if lowlinks[node_id] != indices[node_id]:
            return

        component: list[str] = []
        while True:
            current = stack.pop()
            on_stack.remove(current)
            component.append(current)
            if current == node_id:
                break

        if len(component) > 1:
            cycle_nodes.update(component)
        elif component and component[0] in successors[component[0]]:
            cycle_nodes.add(component[0])

    for node_id in node_ids:
        if node_id not in indices:
            strongconnect(node_id)

    return cycle_nodes
