"""Typed config loading for unified compile and profile pipeline runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from placement_compiler.core.catalogue import load_json_or_yaml
from placement_compiler.core.models import StrictModel, StructuredOutputMethod
from placement_compiler.profiling.models import (
    BaselineConfig,
    ProfileSettings,
    QualityConstraint,
    RankingConfig,
)


PipelinePhase = Literal["compile", "compile_and_profile"]


class CompilerConfig(StrictModel):
    provider: str
    model: str
    temperature: float | None = None
    structured_output_method: StructuredOutputMethod = "function_calling"
    model_kwargs: dict[str, Any] = Field(default_factory=dict)

    @field_validator("provider", "model")
    @classmethod
    def non_empty_string(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value


class CompileConfig(StrictModel):
    candidates: int
    compiler: CompilerConfig
    priorities: list[str] = Field(default_factory=list)
    max_attempts: int = 2

    @model_validator(mode="after")
    def validate_positive_counts(self) -> "CompileConfig":
        if self.candidates <= 0:
            raise ValueError("compile.candidates must be positive")
        if self.max_attempts <= 0:
            raise ValueError("compile.max_attempts must be positive")
        return self


class ProfileConfig(ProfileSettings):
    baseline: BaselineConfig
    quality_constraint: QualityConstraint
    ranking: RankingConfig = Field(default_factory=RankingConfig)
    output: str

    @field_validator("output")
    @classmethod
    def non_empty_string(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value


class PipelineConfig(StrictModel):
    workflow: str
    models: str
    candidate_artifact: str
    compile: CompileConfig
    profile: ProfileConfig | None = None
    default_phase: PipelinePhase = "compile_and_profile"

    @field_validator("workflow", "models", "candidate_artifact")
    @classmethod
    def non_empty_string(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value


class ResolvedProfileConfig(ProfileSettings):
    baseline: BaselineConfig
    quality_constraint: QualityConstraint
    ranking: RankingConfig
    output: Path


class ResolvedPipelineConfig(StrictModel):
    workflow: str
    models: Path
    candidate_artifact: Path
    compile: CompileConfig
    profile: ResolvedProfileConfig | None = None
    default_phase: PipelinePhase = "compile_and_profile"


def load_pipeline_config(path: str | Path) -> ResolvedPipelineConfig:
    config_path = Path(path).resolve()
    config = PipelineConfig.model_validate(load_json_or_yaml(config_path))
    return ResolvedPipelineConfig(
        workflow=config.workflow,
        models=_resolve_config_path(config_path, config.models),
        candidate_artifact=_resolve_config_path(config_path, config.candidate_artifact),
        compile=config.compile,
        profile=(
            ResolvedProfileConfig(
                split=config.profile.split,
                sample_size=config.profile.sample_size,
                seed=config.profile.seed,
                repeats=config.profile.repeats,
                timeout_seconds=config.profile.timeout_seconds,
                warmup_runs=config.profile.warmup_runs,
                examples=config.profile.examples,
                baseline=config.profile.baseline,
                quality_constraint=config.profile.quality_constraint,
                ranking=config.profile.ranking,
                output=_resolve_config_path(config_path, config.profile.output),
            )
            if config.profile
            else None
        ),
        default_phase=config.default_phase,
    )


def _resolve_config_path(config_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()
