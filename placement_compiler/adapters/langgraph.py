"""Extract framework-neutral workflow metadata from compiled LangGraph graphs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from placement_compiler.core.metadata import build_workflow_metadata
from placement_compiler.core.models import NodeRegistryMetadata, WorkflowEdge, WorkflowMetadata


def extract_langgraph_metadata(
    workflow: Any | Callable[[], Any],
    *,
    workflow_id: str,
    registry: Mapping[str, NodeRegistryMetadata | Mapping[str, object]] | None = None,
) -> WorkflowMetadata:
    """Extract metadata from a compiled or buildable LangGraph workflow.

    The adapter only relies on the public compiled-graph ``get_graph()`` method
    and the public drawable graph's ``nodes`` and ``edges`` attributes.
    """

    compiled = workflow() if callable(workflow) else workflow
    if not hasattr(compiled, "get_graph"):
        raise TypeError("LangGraph workflow must be compiled or buildable to a compiled graph")

    drawable_graph = compiled.get_graph()
    nodes = list(getattr(drawable_graph, "nodes", {}).keys())
    raw_edges = getattr(drawable_graph, "edges", None)
    if raw_edges is None:
        raise TypeError("LangGraph drawable graph does not expose edges")

    edges = [
        WorkflowEdge(
            source=str(edge.source),
            target=str(edge.target),
            conditional=bool(getattr(edge, "conditional", False)),
            label=(
                str(edge.data)
                if getattr(edge, "data", None) is not None
                else None
            ),
        )
        for edge in raw_edges
    ]
    return build_workflow_metadata(
        workflow_id=workflow_id,
        node_ids=nodes,
        edges=edges,
        registry=registry,
    )
