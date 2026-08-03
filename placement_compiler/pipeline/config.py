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
    output: Path


class PipelineConfig(StrictModel):
    workflow: str
    models: Path
    candidate_artifact: Path
    compile: CompileConfig
    profile: ProfileConfig | None = None
    default_phase: PipelinePhase = "compile_and_profile"

    @field_validator("workflow")
    @classmethod
    def non_empty_string(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path).resolve()
    config = PipelineConfig.model_validate(load_json_or_yaml(config_path))
    profile = (
        config.profile.model_copy(
            update={"output": _resolve_config_path(config_path, config.profile.output)}
        )
        if config.profile
        else None
    )
    return config.model_copy(
        update={
            "models": _resolve_config_path(config_path, config.models),
            "candidate_artifact": _resolve_config_path(
                config_path, config.candidate_artifact
            ),
            "profile": profile,
        }
    )


def _resolve_config_path(config_path: Path, value: Path) -> Path:
    path = value.expanduser()
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()
