from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


Stage = Literal["entry", "early", "middle", "late", "terminal"]
EndpointLocation = Literal["local", "cloud"]
StructuredOutputMethod = Literal["function_calling", "json_mode", "json_schema"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NodeRegistryMetadata(StrictModel):
    """Optional human-authored metadata that graph structure cannot infer."""

    is_llm_placement_unit: bool = False
    semantic_role: str | None = None
    description: str | None = None
    user_facing_output: bool = False
    tool_use: bool = False
    structured_output_required: bool = False
    branch_control: bool = False
    required_context_window: int | None = None


class WorkflowEdge(StrictModel):
    source: str
    target: str
    conditional: bool = False
    label: str | None = None


class WorkflowNode(NodeRegistryMetadata):
    id: str
    name: str
    structural: bool = True
    predecessors: list[str] = Field(default_factory=list)
    successors: list[str] = Field(default_factory=list)
    in_degree: int = 0
    out_degree: int = 0
    reachable_downstream: list[str] = Field(default_factory=list)
    in_cycle: bool = False
    stage: Stage = "middle"


class WorkflowMetadata(StrictModel):
    workflow_id: str
    nodes: list[WorkflowNode]
    edges: list[WorkflowEdge]
    entry_nodes: list[str] = Field(default_factory=list)
    terminal_nodes: list[str] = Field(default_factory=list)
    conditional_edges: list[WorkflowEdge] = Field(default_factory=list)

    def placement_units(self) -> list[WorkflowNode]:
        return [node for node in self.nodes if node.is_llm_placement_unit]


class ModelEndpoint(StrictModel):
    id: str
    model: str
    location: EndpointLocation
    provider: str
    context_window: int | None = None
    tool_calling: bool = False
    structured_output: bool = False
    input_cost_per_million_tokens: float | None = None
    output_cost_per_million_tokens: float | None = None
    model_kwargs: dict[str, object] = Field(default_factory=dict)

    @field_validator("id", "model", "provider")
    @classmethod
    def non_empty_string(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value


class Candidate(StrictModel):
    id: str
    description: str
    assignments: dict[str, str]
    rationale: dict[str, str] = Field(default_factory=dict)

    @field_validator("id", "description")
    @classmethod
    def non_empty_string(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value


class CandidateSetDraft(StrictModel):
    """Structured output expected from the compiler LLM."""

    candidates: list[Candidate]


class PlacementArtifact(StrictModel):
    schema_version: str = "1.0"
    workflow_id: str
    generated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    candidate_count: int
    workflow: WorkflowMetadata
    model_endpoints: list[ModelEndpoint]
    candidates: list[Candidate]

    @model_validator(mode="after")
    def candidate_count_matches(self) -> "PlacementArtifact":
        if self.candidate_count != len(self.candidates):
            raise ValueError("candidate_count must match the number of candidates")
        return self
