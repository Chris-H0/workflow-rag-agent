"""Resolve placement plans to instrumented model endpoints at runtime."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any, Callable

from placement_compiler.core.models import ModelEndpoint, PlacementPlan, WorkflowMetadata
from placement_compiler.core.usage import calculate_api_cost, extract_token_usage
from placement_compiler.core.validation import PlanValidationError, validate_plan
from placement_compiler.profiling.models import ModelInvocation


class PlacementResolutionError(RuntimeError):
    pass


class TraceCollector:
    def __init__(self) -> None:
        self.invocations: list[ModelInvocation] = []

    def add(self, invocation: ModelInvocation) -> None:
        self.invocations.append(invocation)


class EndpointRegistry:
    def __init__(
        self,
        endpoints: list[ModelEndpoint],
        model_factory: Callable[[ModelEndpoint], Any] | None = None,
    ) -> None:
        self.endpoints = {endpoint.id: endpoint for endpoint in endpoints}
        if len(self.endpoints) != len(endpoints):
            raise ValueError("duplicate endpoint IDs in endpoint registry")
        self._model_factory = model_factory or _build_langchain_model
        self._client_cache: dict[str, Any] = {}

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
        cache_key: str | None = None,
    ) -> "InstrumentedModel":
        base_model = self._base_model(endpoint, cache_key=cache_key)
        return InstrumentedModel(
            base_model=base_model,
            endpoint=endpoint,
            placement_unit_id=placement_unit_id,
            trace_collector=trace_collector,
        )

    def _base_model(self, endpoint: ModelEndpoint, *, cache_key: str | None = None) -> Any:
        key = cache_key or endpoint.id
        if key in self._client_cache:
            return self._client_cache[key]

        model = self._model_factory(endpoint)
        self._client_cache[key] = model
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
        try:
            validate_plan(
                self.plan,
                self.workflow,
                list(self.endpoint_registry.endpoints.values()),
            )
        except PlanValidationError as exc:
            raise PlacementResolutionError(f"plan {self.plan.id!r}: {exc}") from exc

    def get_model(self, node_name: str) -> "InstrumentedModel":
        if node_name not in self.plan.assignments:
            raise PlacementResolutionError(
                f"placement unit {node_name!r} is not assigned in plan {self.plan.id!r}"
            )
        endpoint = self.endpoint_registry.endpoint(self.plan.assignments[node_name])
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
        result_transform: Any | None = None,
        usage_source: Any | None = None,
    ) -> None:
        self._base_model = base_model
        self._endpoint = endpoint
        self._placement_unit_id = placement_unit_id
        self._trace_collector = trace_collector
        self._result_transform = result_transform
        self._usage_source = usage_source

    def bind_tools(self, tools: list[Any]) -> "InstrumentedModel":
        return self._wrap(self._base_model.bind_tools(tools))

    def with_structured_output(self, schema: Any) -> "InstrumentedModel":
        try:
            structured = self._base_model.with_structured_output(
                schema,
                include_raw=True,
            )
        except TypeError:
            return self._wrap(self._base_model.with_structured_output(schema))

        return self._wrap(
            structured,
            result_transform=_structured_parsed_result,
            usage_source=_structured_raw_result,
        )

    def invoke(self, messages: Any) -> Any:
        started = time.perf_counter()
        error = None
        result = None
        returned_result = None
        try:
            result = self._base_model.invoke(messages)
            returned_result = (
                self._result_transform(result)
                if self._result_transform is not None
                else result
            )
            return returned_result
        except Exception as exc:
            error = repr(exc)
            raise
        finally:
            elapsed = time.perf_counter() - started
            usage_result = (
                self._usage_source(result)
                if self._usage_source is not None and result is not None
                else result
            )
            usage = extract_token_usage(usage_result)
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

    def _wrap(
        self,
        base_model: Any,
        *,
        result_transform: Any | None = None,
        usage_source: Any | None = None,
    ) -> "InstrumentedModel":
        return InstrumentedModel(
            base_model=base_model,
            endpoint=self._endpoint,
            placement_unit_id=self._placement_unit_id,
            trace_collector=self._trace_collector,
            result_transform=result_transform,
            usage_source=usage_source,
        )


def _cloud_api_cost(
    endpoint: ModelEndpoint,
    usage: Mapping[str, int | None],
) -> tuple[float | None, bool]:
    if endpoint.location == "local":
        return 0.0, True
    return calculate_api_cost(
        usage,
        input_cost_per_million_tokens=endpoint.input_cost_per_million_tokens,
        output_cost_per_million_tokens=endpoint.output_cost_per_million_tokens,
    )


def _structured_parsed_result(result: Any) -> Any:
    if isinstance(result, Mapping):
        parsing_error = result.get("parsing_error")
        if parsing_error is not None:
            raise ValueError(f"structured output parsing failed: {parsing_error}")
        return result.get("parsed")
    return result


def _structured_raw_result(result: Any) -> Any:
    if isinstance(result, Mapping):
        return result.get("raw") or result
    return result


def _build_langchain_model(endpoint: ModelEndpoint) -> Any:
    from langchain.chat_models import init_chat_model

    return init_chat_model(
        endpoint.model,
        model_provider=endpoint.provider,
        **endpoint.model_kwargs,
    )
