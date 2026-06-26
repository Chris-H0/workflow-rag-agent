from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from placement_compiler.core.models import ModelEndpoint, NodeRegistryMetadata


def load_model_catalogue(path: str | Path) -> list[ModelEndpoint]:
    data = load_json_or_yaml(Path(path))
    if isinstance(data, Mapping):
        raw_endpoints = data.get("model_endpoints") or data.get("endpoints")
    else:
        raw_endpoints = data

    if not isinstance(raw_endpoints, list):
        raise ValueError("model catalogue must be a list or contain model_endpoints")

    endpoints = [ModelEndpoint.model_validate(item) for item in raw_endpoints]
    _ensure_unique_endpoint_ids(endpoints)
    return endpoints


def load_node_registry(
    path: str | Path,
) -> dict[str, NodeRegistryMetadata]:
    data = load_json_or_yaml(Path(path))
    if isinstance(data, Mapping) and "nodes" in data:
        data = data["nodes"]
    if not isinstance(data, Mapping):
        raise ValueError("node registry must be a mapping of node ID to metadata")
    return {
        str(node_id): NodeRegistryMetadata.model_validate(metadata)
        for node_id, metadata in data.items()
    }


def load_json_or_yaml(path: str | Path) -> Any:
    path = Path(path)
    text = path.read_text()
    if path.suffix.lower() == ".json":
        return json.loads(text)

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - PyYAML is installed via LangChain.
        raise RuntimeError("YAML support requires PyYAML to be installed") from exc
    return yaml.safe_load(text)


def _ensure_unique_endpoint_ids(endpoints: list[ModelEndpoint]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for endpoint in endpoints:
        if endpoint.id in seen:
            duplicates.add(endpoint.id)
        seen.add(endpoint.id)
    if duplicates:
        raise ValueError(f"duplicate model endpoint IDs: {sorted(duplicates)}")
