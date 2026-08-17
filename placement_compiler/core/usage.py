"""Small shared helpers for token usage and API-cost accounting."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def extract_token_usage(result: Any) -> dict[str, int | None]:
    usage = getattr(result, "usage_metadata", None) or {}
    response_metadata = getattr(result, "response_metadata", None) or {}
    token_usage = response_metadata.get("token_usage") or {}

    input_tokens = _first_present(
        usage.get("input_tokens"),
        usage.get("prompt_tokens"),
        token_usage.get("prompt_tokens"),
    )
    output_tokens = _first_present(
        usage.get("output_tokens"),
        usage.get("completion_tokens"),
        token_usage.get("completion_tokens"),
    )
    total_tokens = _first_present(
        usage.get("total_tokens"), token_usage.get("total_tokens")
    )
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    return {
        "input_tokens": _int_or_none(input_tokens),
        "output_tokens": _int_or_none(output_tokens),
        "total_tokens": _int_or_none(total_tokens),
    }


def calculate_api_cost(
    usage: Mapping[str, int | None],
    *,
    input_cost_per_million_tokens: float | None,
    output_cost_per_million_tokens: float | None,
) -> tuple[float | None, bool]:
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    if (
        input_tokens is None
        or output_tokens is None
        or input_cost_per_million_tokens is None
        or output_cost_per_million_tokens is None
    ):
        return None, False
    return (
        input_tokens / 1_000_000 * input_cost_per_million_tokens
        + output_tokens / 1_000_000 * output_cost_per_million_tokens,
        True,
    )


def _first_present(*values: Any) -> Any:
    return next((value for value in values if value is not None), None)


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
