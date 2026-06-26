from __future__ import annotations

from pathlib import Path

from placement_compiler.catalogue import load_json_or_yaml
from placement_compiler.profile_models import ProfileRunConfig, ResolvedProfileRunConfig


def load_profile_run_config(path: str | Path) -> ResolvedProfileRunConfig:
    config_path = Path(path).resolve()
    config = ProfileRunConfig.model_validate(load_json_or_yaml(config_path))
    return ResolvedProfileRunConfig(
        workflow=config.workflow,
        candidate_artifact=_resolve_config_path(config_path, config.candidate_artifact),
        profile=config.profile,
        baseline=config.baseline,
        quality_constraint=config.quality_constraint,
        ranking=config.ranking,
        output=_resolve_config_path(config_path, config.output),
    )


def _resolve_config_path(config_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()
