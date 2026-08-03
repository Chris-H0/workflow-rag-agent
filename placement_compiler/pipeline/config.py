"""Typed config for the sequential model-routing search."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field, field_validator, model_validator

from placement_compiler.core.catalogue import load_json_or_yaml
from placement_compiler.core.models import StrictModel, StructuredOutputMethod
from placement_compiler.profiling.models import EvaluationSettings


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


class PipelineConfig(EvaluationSettings):
    workflow: str
    models: Path
    strongest_cloud_endpoint: str
    iterations: int = 3
    compiler: CompilerConfig
    max_attempts: int = 2
    output: Path

    @field_validator("workflow", "strongest_cloud_endpoint")
    @classmethod
    def non_empty_string(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    @model_validator(mode="after")
    def validate_counts(self) -> "PipelineConfig":
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        return self


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path).resolve()
    config = PipelineConfig.model_validate(load_json_or_yaml(config_path))
    return config.model_copy(
        update={
            "models": _resolve_config_path(config_path, config.models),
            "output": _resolve_config_path(config_path, config.output),
        }
    )


def _resolve_config_path(config_path: Path, value: Path) -> Path:
    path = value.expanduser()
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()
