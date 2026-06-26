from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping
from typing import Any

from langchain_core.messages import AIMessage

from placement_compiler.models import ModelEndpoint, WorkflowMetadata, WorkflowNode
from placement_compiler.profile_models import ModelInvocation, PlacementPlan


class PlacementResolutionError(RuntimeError):
    pass


class TraceCollector:
    def __init__(self) -> None:
        self.invocations: list[ModelInvocation] = []

    def add(self, invocation: ModelInvocation) -> None:
        self.invocations.append(invocation)


class EndpointRegistry:
    def __init__(self, endpoints: list[ModelEndpoint]) -> None:
        self.endpoints = {endpoint.id: endpoint for endpoint in endpoints}
        if len(self.endpoints) != len(endpoints):
            raise ValueError("duplicate endpoint IDs in endpoint registry")
        self._client_cache: dict[tuple[str, str, str], Any] = {}

    def endpoint(self, endpoint_id: str) -> ModelEndpoint:
        try:
            return self.endpoints[endpoint_id]
        except KeyError as exc:
            raise PlacementResolutionError(f"unknown endpoint ID {endpoint_id!r}") from exc

    def build_model(
        self,
        *,
        endpoint: ModelEndpoint,
        placement_unit_id: str,
        trace_collector: TraceCollector,
    ) -> "InstrumentedModel":
        base_model = self._base_model(endpoint)
        return InstrumentedModel(
            base_model=base_model,
            endpoint=endpoint,
            placement_unit_id=placement_unit_id,
            trace_collector=trace_collector,
        )

    def _base_model(self, endpoint: ModelEndpoint) -> Any:
        cache_key = (
            endpoint.provider,
            endpoint.model,
            json.dumps(endpoint.model_kwargs, sort_keys=True, default=str),
        )
        if cache_key in self._client_cache:
            return self._client_cache[cache_key]

        if endpoint.provider == "mock":
            model = MockPlacementModel(endpoint.model, dict(endpoint.model_kwargs))
        else:
            from langchain.chat_models import init_chat_model

            model = init_chat_model(
                endpoint.model,
                model_provider=endpoint.provider,
                **endpoint.model_kwargs,
            )

        self._client_cache[cache_key] = model
        return model


class ModelResolver:
    def __init__(
        self,
        *,
        plan: PlacementPlan,
        endpoint_registry: EndpointRegistry,
        workflow: WorkflowMetadata,
        trace_collector: TraceCollector,
    ) -> None:
        self.plan = plan
        self.endpoint_registry = endpoint_registry
        self.workflow = workflow
        self.trace_collector = trace_collector
        self._placement_units = {node.id: node for node in workflow.placement_units()}
        self.validate_plan()

    def validate_plan(self) -> None:
        expected = set(self._placement_units)
        actual = set(self.plan.assignments)
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        if missing:
            raise PlacementResolutionError(
                f"plan {self.plan.id!r} is missing placement units {missing}"
            )
        if unknown:
            raise PlacementResolutionError(
                f"plan {self.plan.id!r} contains unknown placement units {unknown}"
            )

        for unit_id, endpoint_id in sorted(self.plan.assignments.items()):
            endpoint = self.endpoint_registry.endpoint(endpoint_id)
            _validate_endpoint_capabilities(unit_id, self._placement_units[unit_id], endpoint)

    def get_model(self, node_name: str) -> "InstrumentedModel":
        if node_name not in self.plan.assignments:
            raise PlacementResolutionError(
                f"placement unit {node_name!r} is not assigned in plan {self.plan.id!r}"
            )
        endpoint = self.endpoint_registry.endpoint(self.plan.assignments[node_name])
        _validate_endpoint_capabilities(node_name, self._placement_units[node_name], endpoint)
        return self.endpoint_registry.build_model(
            endpoint=endpoint,
            placement_unit_id=node_name,
            trace_collector=self.trace_collector,
        )


class InstrumentedModel:
    def __init__(
        self,
        *,
        base_model: Any,
        endpoint: ModelEndpoint,
        placement_unit_id: str,
        trace_collector: TraceCollector,
    ) -> None:
        self._base_model = base_model
        self._endpoint = endpoint
        self._placement_unit_id = placement_unit_id
        self._trace_collector = trace_collector

    def bind_tools(self, tools: list[Any]) -> "InstrumentedModel":
        return self._wrap(self._base_model.bind_tools(tools))

    def with_structured_output(self, schema: Any) -> "InstrumentedModel":
        return self._wrap(self._base_model.with_structured_output(schema))

    def invoke(self, messages: Any) -> Any:
        started = time.perf_counter()
        error = None
        result = None
        try:
            result = self._base_model.invoke(messages)
            return result
        except Exception as exc:
            error = repr(exc)
            raise
        finally:
            elapsed = time.perf_counter() - started
            usage = _extract_usage(result)
            cloud_cost, cost_known = _cloud_api_cost(self._endpoint, usage)
            self._trace_collector.add(
                ModelInvocation(
                    placement_unit_id=self._placement_unit_id,
                    endpoint_id=self._endpoint.id,
                    provider=self._endpoint.provider,
                    location=self._endpoint.location,
                    model=self._endpoint.model,
                    latency_seconds=elapsed,
                    input_tokens=usage.get("input_tokens"),
                    output_tokens=usage.get("output_tokens"),
                    total_tokens=usage.get("total_tokens"),
                    cloud_api_cost=cloud_cost,
                    cost_known=cost_known,
                    error=error,
                )
            )

    def _wrap(self, base_model: Any) -> "InstrumentedModel":
        return InstrumentedModel(
            base_model=base_model,
            endpoint=self._endpoint,
            placement_unit_id=self._placement_unit_id,
            trace_collector=self._trace_collector,
        )


class MockPlacementModel:
    def __init__(self, model: str, model_kwargs: dict[str, Any]) -> None:
        self.model = model
        self.model_kwargs = model_kwargs
        self._structured_schema: Any = None

    def bind_tools(self, tools: list[Any]) -> "MockPlacementModel":
        return self

    def with_structured_output(self, schema: Any) -> "MockPlacementModel":
        clone = MockPlacementModel(self.model, self.model_kwargs)
        clone._structured_schema = schema
        return clone

    def invoke(self, messages: Any) -> Any:
        content = self._content_for(messages)
        usage = _mock_usage(messages, content)
        if self._structured_schema is not None:
            data = self._structured_payload()
            return self._structured_schema.model_validate(data)
        return AIMessage(content=content, usage_metadata=usage)

    def _structured_payload(self) -> dict[str, str]:
        return self.model_kwargs.get("structured_payload") or {"decision": "answer"}

    def _content_for(self, messages: Any) -> str:
        text = _messages_text(messages)
        unit_outputs = self.model_kwargs.get("unit_outputs") or {}
        for unit_hint, output in unit_outputs.items():
            if str(unit_hint) in text:
                return str(output)

        if "Review" in text or "generated_test_result" in text:
            return json.dumps({"decision": "final", "comments": "mock review"})
        if "Generate tests" in text or "test" in text.lower():
            entry_point = _entry_point(text)
            return f"assert {entry_point}(1, 2) == 3"
        if "Entry point:" in text or "Solve this MBPP" in text:
            entry_point = _entry_point(text)
            return f"def {entry_point}(a, b):\n    return a + b"
        if "plan" in text.lower() or "programming task" in text.lower():
            return "Use a direct implementation."
        return str(self.model_kwargs.get("default_answer", "Paris"))


def _validate_endpoint_capabilities(
    unit_id: str,
    node: WorkflowNode,
    endpoint: ModelEndpoint,
) -> None:
    if node.tool_use and not endpoint.tool_calling:
        raise PlacementResolutionError(
            f"placement unit {unit_id!r} requires tool calling but endpoint "
            f"{endpoint.id!r} does not support it"
        )
    if node.structured_output_required and not endpoint.structured_output:
        raise PlacementResolutionError(
            f"placement unit {unit_id!r} requires structured output but endpoint "
            f"{endpoint.id!r} does not support it"
        )
    if (
        node.required_context_window is not None
        and endpoint.context_window is not None
        and endpoint.context_window < node.required_context_window
    ):
        raise PlacementResolutionError(
            f"placement unit {unit_id!r} requires context window "
            f"{node.required_context_window} but endpoint {endpoint.id!r} only "
            f"declares {endpoint.context_window}"
        )


def _cloud_api_cost(
    endpoint: ModelEndpoint,
    usage: Mapping[str, int | None],
) -> tuple[float | None, bool]:
    if endpoint.location == "local":
        return 0.0, True
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    if input_tokens is None or output_tokens is None:
        return None, False
    if (
        endpoint.input_cost_per_million_tokens is None
        or endpoint.output_cost_per_million_tokens is None
    ):
        return None, False
    return (
        input_tokens / 1_000_000 * endpoint.input_cost_per_million_tokens
        + output_tokens / 1_000_000 * endpoint.output_cost_per_million_tokens,
        True,
    )


def _extract_usage(result: Any) -> dict[str, int | None]:
    usage = getattr(result, "usage_metadata", None) or {}
    response_metadata = getattr(result, "response_metadata", None) or {}
    token_usage = response_metadata.get("token_usage") or {}

    input_tokens = (
        usage.get("input_tokens")
        or usage.get("prompt_tokens")
        or token_usage.get("prompt_tokens")
    )
    output_tokens = (
        usage.get("output_tokens")
        or usage.get("completion_tokens")
        or token_usage.get("completion_tokens")
    )
    total_tokens = usage.get("total_tokens") or token_usage.get("total_tokens")
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    return {
        "input_tokens": _int_or_none(input_tokens),
        "output_tokens": _int_or_none(output_tokens),
        "total_tokens": _int_or_none(total_tokens),
    }


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _mock_usage(messages: Any, content: str) -> dict[str, int]:
    input_tokens = max(1, len(_messages_text(messages).split()))
    output_tokens = max(1, len(content.split()))
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def _messages_text(messages: Any) -> str:
    if isinstance(messages, list):
        return "\n".join(str(getattr(message, "content", message)) for message in messages)
    return str(messages)


def _entry_point(text: str) -> str:
    match = re.search(r"Entry point:\s*([A-Za-z_][A-Za-z0-9_]*)", text)
    if match:
        return match.group(1)
    return "add"
