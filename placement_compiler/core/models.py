"""Shared Pydantic models for workflows, endpoints, candidates, and artifacts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


EndpointLocation = Literal["local", "cloud"]
StructuredOutputMethod = Literal["function_calling", "json_mode", "json_schema"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PlacementUnitSpec(StrictModel):
    """Authoritative workflow-owned placement metadata."""

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


class PlacementUnit(PlacementUnitSpec):
    id: str
    structural: bool = False


class WorkflowMetadata(StrictModel):
    workflow_id: str
    placement_units: list[PlacementUnit]
    edges: list[WorkflowEdge] = Field(default_factory=list)
    entry_nodes: list[str] = Field(default_factory=list)
    terminal_nodes: list[str] = Field(default_factory=list)
    conditional_edges: list[WorkflowEdge] = Field(default_factory=list)


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


class PlacementPlan(StrictModel):
    id: str
    description: str | None = None
    assignments: dict[str, str]
    rationale: dict[str, str] = Field(default_factory=dict)
    source: Literal["candidate", "baseline"] = "candidate"

    @field_validator("id")
    @classmethod
    def non_empty_string(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value


class PlacementProposal(StrictModel):
    """One complete placement proposed by the compiler."""

    description: str | None = None
    assignments: dict[str, str]
    rationale: dict[str, str] = Field(default_factory=dict)


class CompilerAttempt(StrictModel):
    """Observed cost and outcome of one compiler-LLM proposal attempt."""

    iteration: int
    attempt: int
    accepted: bool
    proposal_latency_seconds: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    api_cost: float | None = None
    cost_known: bool = False
    validation_errors: list[str] = Field(default_factory=list)
    duplicate_proposal: bool = False
    generation_error: str | None = None
