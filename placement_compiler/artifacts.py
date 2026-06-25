from __future__ import annotations

import json
from pathlib import Path

from placement_compiler.models import Candidate, ModelEndpoint, PlacementArtifact, WorkflowMetadata


def build_artifact(
    *,
    workflow: WorkflowMetadata,
    model_endpoints: list[ModelEndpoint],
    candidates: list[Candidate],
) -> PlacementArtifact:
    return PlacementArtifact(
        workflow_id=workflow.workflow_id,
        candidate_count=len(candidates),
        workflow=workflow,
        model_endpoints=model_endpoints,
        candidates=candidates,
    )


def write_artifact(artifact: PlacementArtifact, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(artifact.model_dump(mode="json", exclude_none=True), indent=2)
        + "\n"
    )
    return path
