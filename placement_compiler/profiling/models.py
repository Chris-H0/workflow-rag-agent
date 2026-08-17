"""Pydantic models for placement evaluation, traces, and search results."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import Field, model_validator

from placement_compiler.core.models import CompilerAttempt, PlacementPlan, StrictModel


MetricDirection = Literal["maximise", "minimise"]


class MetricDefinition(StrictModel):
    name: str
    direction: MetricDirection


class EvaluationSettings(StrictModel):
    split: str = "validation"
    sample_size: int = 10
    seed: int = 42
    repeats: int = 1
    warmup_runs: int = 0
    examples: list[dict[str, Any]] | None = None

    @model_validator(mode="after")
    def validate_positive_counts(self) -> "EvaluationSettings":
        if self.sample_size <= 0:
            raise ValueError("sample_size must be positive")
        if self.repeats <= 0:
            raise ValueError("repeats must be positive")
        if self.warmup_runs < 0:
            raise ValueError("warmup_runs must not be negative")
        return self


class ModelInvocation(StrictModel):
    placement_unit_id: str
    endpoint_id: str
    provider: str
    location: str
    model: str
    latency_seconds: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cloud_api_cost: float | None = None
    cost_known: bool = True
    error: str | None = None


class RunTrace(StrictModel):
    invocations: list[ModelInvocation] = Field(default_factory=list)
    system_metrics: dict[str, Any] = Field(default_factory=dict)


class WorkflowResult(StrictModel):
    output: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    trace: RunTrace = Field(default_factory=RunTrace)
    latency_seconds: float


class RunRecord(StrictModel):
    plan_id: str
    plan_source: Literal["candidate", "baseline"]
    example_id: str
    repeat: int
    metrics: dict[str, Any] = Field(default_factory=dict)
    latency_seconds: float
    error: str | None = None
    output: dict[str, Any] = Field(default_factory=dict)
    trace: RunTrace = Field(default_factory=RunTrace)


class PlanResult(StrictModel):
    plan: PlacementPlan
    metrics: dict[str, Any] = Field(default_factory=dict)


class CompilationMetrics(StrictModel):
    """Aggregate overhead of proposal generation and workflow profiling."""

    attempt_count: int = 0
    validation_error_count: int = 0
    duplicate_proposal_count: int = 0
    compiler_prompt_tokens: int | None = 0
    compiler_completion_tokens: int | None = 0
    compiler_total_tokens: int | None = 0
    compiler_token_usage_unknown: bool = False
    direct_compiler_api_cost: float | None = 0.0
    direct_compiler_api_cost_unknown: bool = False
    compiler_proposal_latency_seconds: float = 0.0
    workflow_profiling_api_cost: float | None = 0.0
    workflow_profiling_api_cost_unknown: bool = False
    workflow_profiling_elapsed_seconds: float = 0.0
    total_compilation_api_cost: float | None = 0.0
    total_compilation_api_cost_unknown: bool = False
    total_compilation_elapsed_seconds: float = 0.0


class SearchArtifact(StrictModel):
    schema_version: str = "5.0"
    generated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    workflow_id: str
    workflow: str
    search_config: dict[str, Any]
    primary_metric: MetricDefinition
    selected_example_ids: list[str]
    results: list[PlanResult]
    compiler_attempts: list[CompilerAttempt] = Field(default_factory=list)
    compilation_metrics: CompilationMetrics | None = None
