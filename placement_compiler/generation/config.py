from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field, field_validator, model_validator

from placement_compiler.core.catalogue import load_json_or_yaml
from placement_compiler.core.models import StrictModel, StructuredOutputMethod


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


class PlacementRunConfig(StrictModel):
    workflow: str
    models: str
    candidates: int
    output: str
    compiler: CompilerConfig
    metadata: str | None = None
    priorities: list[str] = Field(default_factory=list)
    max_attempts: int = 2

    @field_validator("workflow", "models", "output")
    @classmethod
    def non_empty_string(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    @model_validator(mode="after")
    def positive_counts(self) -> "PlacementRunConfig":
        if self.candidates <= 0:
            raise ValueError("candidates must be positive")
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        return self


class ResolvedPlacementRunConfig(StrictModel):
    workflow: str
    models: Path
    candidates: int
    output: Path
    compiler: CompilerConfig
    metadata: Path | None = None
    priorities: list[str] = Field(default_factory=list)
    max_attempts: int = 2


def load_run_config(path: str | Path) -> ResolvedPlacementRunConfig:
    config_path = Path(path).resolve()
    data = load_json_or_yaml(config_path)
    config = PlacementRunConfig.model_validate(data)
    return ResolvedPlacementRunConfig(
        workflow=config.workflow,
        models=_resolve_config_path(config_path, config.models),
        candidates=config.candidates,
        output=_resolve_config_path(config_path, config.output),
        compiler=config.compiler,
        metadata=(
            _resolve_config_path(config_path, config.metadata)
            if config.metadata
            else None
        ),
        priorities=config.priorities,
        max_attempts=config.max_attempts,
    )


def _resolve_config_path(config_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()
