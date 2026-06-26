from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from placement_compiler.models import Candidate, ModelEndpoint, PlacementArtifact, StrictModel


MetricDirection = Literal["maximise", "minimise"]
BaselineType = Literal["all_endpoint", "candidate"]


class MetricDefinition(StrictModel):
    name: str
    direction: MetricDirection


class ProfileSettings(StrictModel):
    split: str = "validation"
    sample_size: int = 10
    seed: int = 42
    repeats: int = 1
    timeout_seconds: float = 120
    warmup_runs: int = 0
    dataset_options: dict[str, Any] = Field(default_factory=dict)
    examples: list[dict[str, Any]] | None = None

    @model_validator(mode="after")
    def validate_positive_counts(self) -> "ProfileSettings":
        if self.sample_size <= 0:
            raise ValueError("profile.sample_size must be positive")
        if self.repeats <= 0:
            raise ValueError("profile.repeats must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("profile.timeout_seconds must be positive")
        if self.warmup_runs < 0:
            raise ValueError("profile.warmup_runs must not be negative")
        return self


class BaselineConfig(StrictModel):
    type: BaselineType
    endpoint_id: str | None = None
    candidate_id: str | None = None

    @model_validator(mode="after")
    def validate_type_fields(self) -> "BaselineConfig":
        if self.type == "all_endpoint" and not self.endpoint_id:
            raise ValueError("baseline.endpoint_id is required for all_endpoint")
        if self.type == "candidate" and not self.candidate_id:
            raise ValueError("baseline.candidate_id is required for candidate")
        return self


class QualityConstraint(StrictModel):
    metric: str
    max_drop_from_baseline: float | None = None
    absolute_minimum: float | None = None
    absolute_maximum: float | None = None


class RankingObjective(StrictModel):
    metric: str
    direction: MetricDirection


class RankingConfig(StrictModel):
    objectives: list[RankingObjective] = Field(default_factory=list)


class ProfileRunConfig(StrictModel):
    workflow: str
    candidate_artifact: str
    profile: ProfileSettings = Field(default_factory=ProfileSettings)
    baseline: BaselineConfig
    quality_constraint: QualityConstraint
    ranking: RankingConfig = Field(default_factory=RankingConfig)
    output: str

    @field_validator("workflow", "candidate_artifact", "output")
    @classmethod
    def non_empty_string(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value


class ResolvedProfileRunConfig(StrictModel):
    workflow: str
    candidate_artifact: Path
    profile: ProfileSettings
    baseline: BaselineConfig
    quality_constraint: QualityConstraint
    ranking: RankingConfig
    output: Path


class PlacementPlan(StrictModel):
    id: str
    assignments: dict[str, str]
    source: Literal["candidate", "baseline"] = "candidate"
    description: str | None = None


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


class WorkflowExecution(StrictModel):
    output: dict[str, Any] = Field(default_factory=dict)
    trace: RunTrace = Field(default_factory=RunTrace)
    latency_seconds: float
    error: str | None = None
    timed_out: bool = False


class RunRecord(StrictModel):
    candidate_id: str
    candidate_source: Literal["candidate", "baseline"]
    example_id: str
    repeat: int
    metrics: dict[str, Any] = Field(default_factory=dict)
    latency_seconds: float | None = None
    cloud_api_cost: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    local_call_count: int = 0
    cloud_call_count: int = 0
    invocation_count: int = 0
    per_placement_unit: dict[str, dict[str, Any]] = Field(default_factory=dict)
    error: str | None = None
    timed_out: bool = False
    output: dict[str, Any] = Field(default_factory=dict)
    trace: RunTrace = Field(default_factory=RunTrace)


class CandidateProfile(StrictModel):
    candidate_id: str
    candidate_source: Literal["candidate", "baseline"]
    assignments: dict[str, str]
    metrics: dict[str, Any] = Field(default_factory=dict)
    feasible: bool = False
    infeasible_reason: str | None = None
    rank: int | None = None
    diagnostic_rank: int | None = None


class ProfileArtifact(StrictModel):
    schema_version: str = "2.0"
    generated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    source_candidate_artifact: str
    workflow_id: str
    workflow: str
    profile_config: dict[str, Any]
    primary_metric: MetricDefinition
    selected_example_ids: list[str]
    baseline: CandidateProfile
    quality_constraint: QualityConstraint
    ranking: RankingConfig
    candidates: list[CandidateProfile]
    selected_candidate_id: str | None = None
    failures: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class LoadedCandidateArtifact(StrictModel):
    path: Path
    artifact: PlacementArtifact

    def endpoint_by_id(self) -> dict[str, ModelEndpoint]:
        return {endpoint.id: endpoint for endpoint in self.artifact.model_endpoints}

    def candidate_by_id(self) -> dict[str, Candidate]:
        return {candidate.id: candidate for candidate in self.artifact.candidates}
