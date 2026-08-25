from __future__ import annotations

import json
import unittest

from langchain_core.messages import AIMessage
from pydantic import BaseModel

from placement_compiler.core.models import (
    ModelEndpoint,
    PlacementPlan,
    PlacementUnit,
    WorkflowEdge,
    WorkflowMetadata,
)
from placement_compiler.experiments.runtime_router import (
    _runtime_metrics,
    load_runtime_routing_config,
)
from placement_compiler.profiling.models import RunRecord, RunTrace
from placement_compiler.profiling.runner import aggregate_plan_result
from placement_compiler.runtime.dynamic_router import (
    ROUTER_TRACE_PREFIX,
    RouteDecision,
    RuntimeRouterResolver,
)
from placement_compiler.runtime.endpoint_registry import EndpointRegistry, TraceCollector


class StructuredAnswer(BaseModel):
    decision: str


class FakeState:
    def __init__(self, *router_choices: str) -> None:
        self.router_choices = list(router_choices)
        self.calls: list[dict] = []


class FakeModel:
    def __init__(
        self,
        endpoint_id: str,
        state: FakeState,
        *,
        schema=None,
        include_raw: bool = False,
        tools_bound: bool = False,
    ) -> None:
        self.endpoint_id = endpoint_id
        self.state = state
        self.schema = schema
        self.include_raw = include_raw
        self.tools_bound = tools_bound

    def bind_tools(self, tools):
        return FakeModel(
            self.endpoint_id,
            self.state,
            schema=self.schema,
            include_raw=self.include_raw,
            tools_bound=bool(tools),
        )

    def with_structured_output(self, schema, *, include_raw=False, **kwargs):
        return FakeModel(
            self.endpoint_id,
            self.state,
            schema=schema,
            include_raw=include_raw,
            tools_bound=self.tools_bound,
        )

    def invoke(self, messages):
        text = "\n".join(str(getattr(message, "content", message)) for message in messages)
        self.state.calls.append(
            {
                "endpoint_id": self.endpoint_id,
                "text": text,
                "tools_bound": self.tools_bound,
                "schema": self.schema,
            }
        )
        usage = {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}
        if self.schema is RouteDecision:
            data = {
                "endpoint_id": self.state.router_choices.pop(0),
                "reason": "Selected by the fake judge.",
            }
            parsed = RouteDecision.model_validate(data)
            raw = AIMessage(content=json.dumps(data), usage_metadata=usage)
            return (
                {"raw": raw, "parsed": parsed, "parsing_error": None}
                if self.include_raw
                else parsed
            )
        if self.schema is not None:
            parsed = self.schema.model_validate({"decision": "answer"})
            raw = AIMessage(
                content=json.dumps({"decision": "answer"}),
                usage_metadata=usage,
            )
            return (
                {"raw": raw, "parsed": parsed, "parsing_error": None}
                if self.include_raw
                else parsed
            )
        return AIMessage(content=f"response from {self.endpoint_id}", usage_metadata=usage)


def endpoint(
    endpoint_id: str,
    *,
    location: str,
    structured_output: bool,
) -> ModelEndpoint:
    return ModelEndpoint(
        id=endpoint_id,
        model=endpoint_id,
        provider="mock",
        location=location,
        tool_calling=True,
        structured_output=structured_output,
        input_cost_per_million_tokens=1.0,
        output_cost_per_million_tokens=2.0,
    )


def workflow() -> WorkflowMetadata:
    edge = WorkflowEdge(
        source="tool_node",
        target="structured_node",
        conditional=True,
    )
    return WorkflowMetadata(
        workflow_id="runtime-test",
        placement_units=[
            PlacementUnit(id="tool_node", tool_use=True),
            PlacementUnit(id="structured_node", structured_output_required=True),
        ],
        edges=[edge],
        entry_nodes=["tool_node"],
        terminal_nodes=["structured_node"],
        conditional_edges=[edge],
    )


def resolver(
    *choices: str,
    include_full_workflow_metadata: bool = False,
    local_structured_output: bool = False,
):
    state = FakeState(*choices)
    endpoints = [
        endpoint(
            "qwen-local",
            location="local",
            structured_output=local_structured_output,
        ),
        endpoint("mini-cloud", location="cloud", structured_output=True),
        endpoint("strong-cloud", location="cloud", structured_output=True),
    ]
    collector = TraceCollector()
    registry = EndpointRegistry(
        endpoints,
        model_factory=lambda item: FakeModel(item.id, state),
    )
    return (
        RuntimeRouterResolver(
            workflow=workflow(),
            endpoint_registry=registry,
            router_endpoint_id="mini-cloud",
            fallback_endpoint_id="strong-cloud",
            router_temperature=0.0,
            hardware_context="Apple M4 Pro with 24 GB unified memory",
            trace_collector=collector,
            include_full_workflow_metadata=include_full_workflow_metadata,
        ),
        state,
        collector,
    )


class RuntimeRouterTests(unittest.TestCase):
    def test_routes_tool_call_and_records_router_and_target_overhead(self):
        runtime_resolver, state, collector = resolver("qwen-local")

        response = runtime_resolver.get_model("tool_node").bind_tools([object()]).invoke(
            [{"role": "user", "content": "simple retrieval control"}]
        )

        self.assertIn("qwen-local", response.content)
        self.assertEqual(len(collector.invocations), 2)
        self.assertTrue(
            collector.invocations[0].placement_unit_id.startswith(ROUTER_TRACE_PREFIX)
        )
        self.assertEqual(collector.invocations[1].placement_unit_id, "tool_node")
        self.assertEqual(collector.invocations[1].endpoint_id, "qwen-local")
        self.assertTrue(state.calls[-1]["tools_bound"])
        self.assertIn("Apple M4 Pro", state.calls[0]["text"])
        self.assertNotIn('"workflow":', state.calls[0]["text"])

        run = RunRecord(
            plan_id="runtime-router",
            plan_source="candidate",
            example_id="one",
            repeat=0,
            metrics={"quality": 1.0},
            latency_seconds=0.1,
            trace=RunTrace(
                invocations=collector.invocations,
                system_metrics={"runtime_routing": runtime_resolver.decisions},
            ),
        )
        metrics = _runtime_metrics([run])
        aggregate = aggregate_plan_result(
            PlacementPlan(
                id="runtime-router",
                assignments={"tool_node": "runtime-selected"},
            ),
            [run],
            primary_metric_name="quality",
        )
        self.assertEqual(metrics["router_call_count"], 1)
        self.assertGreater(metrics["router_cloud_api_cost"], 0)
        self.assertGreater(metrics["router_latency_seconds"], 0)
        self.assertEqual(metrics["selected_endpoint_counts"], {"qwen-local": 1})
        self.assertEqual(
            aggregate.metrics["cloud_api_cost"], metrics["router_cloud_api_cost"]
        )

    def test_structured_node_excludes_local_endpoint(self):
        runtime_resolver, state, collector = resolver("mini-cloud")

        response = runtime_resolver.get_model("structured_node").with_structured_output(
            StructuredAnswer
        ).invoke([{"role": "user", "content": "choose an action"}])

        self.assertEqual(response.decision, "answer")
        decision = runtime_resolver.decisions[0]
        self.assertEqual(
            decision["compatible_endpoint_ids"], ["mini-cloud", "strong-cloud"]
        )
        self.assertEqual(collector.invocations[-1].endpoint_id, "mini-cloud")
        self.assertNotIn('"id": "qwen-local"', state.calls[0]["text"])

    def test_structured_node_can_route_to_capable_local_endpoint(self):
        runtime_resolver, state, collector = resolver(
            "qwen-local",
            local_structured_output=True,
        )

        response = runtime_resolver.get_model("structured_node").with_structured_output(
            StructuredAnswer
        ).invoke([{"role": "user", "content": "choose an action"}])

        self.assertEqual(response.decision, "answer")
        self.assertEqual(
            runtime_resolver.decisions[0]["compatible_endpoint_ids"],
            ["qwen-local", "mini-cloud", "strong-cloud"],
        )
        self.assertEqual(collector.invocations[-1].endpoint_id, "qwen-local")
        self.assertIn('"id": "qwen-local"', state.calls[0]["text"])

    def test_invalid_judge_choice_falls_back_once_to_strong_model(self):
        runtime_resolver, _, collector = resolver("not-an-endpoint")

        response = runtime_resolver.get_model("tool_node").invoke(
            [{"role": "user", "content": "do the task"}]
        )

        self.assertIn("strong-cloud", response.content)
        self.assertEqual(len(collector.invocations), 2)
        decision = runtime_resolver.decisions[0]
        self.assertTrue(decision["fallback_used"])
        self.assertEqual(decision["selected_endpoint_id"], "strong-cloud")
        self.assertIn("not-an-endpoint", decision["router_error"])

    def test_full_workflow_context_uses_compiler_metadata_object(self):
        runtime_resolver, state, _ = resolver(
            "mini-cloud", include_full_workflow_metadata=True
        )

        runtime_resolver.get_model("tool_node").invoke(
            [{"role": "user", "content": "route this invocation"}]
        )

        prompt = state.calls[0]["text"]
        self.assertIn('"workflow":', prompt)
        self.assertIn('"workflow_id": "runtime-test"', prompt)
        self.assertIn('"structured_node"', prompt)
        self.assertIn('"conditional_edges"', prompt)
        self.assertIn("downstream importance", prompt)
        self.assertTrue(
            runtime_resolver.decisions[0]["full_workflow_metadata_included"]
        )

    def test_full_configs_request_two_fifty_example_generations(self):
        for workflow_name in ("qa", "code"):
            config = load_runtime_routing_config(
                f"placement_compiler/examples/{workflow_name}_runtime_router.yaml"
            )
            self.assertEqual(config.workflow, workflow_name)
            self.assertEqual(config.sample_size, 50)
            self.assertEqual(config.repeats, 2)
            self.assertEqual(config.router_endpoint, "gpt-4.1-mini-cloud")
            self.assertIn("Apple M4 Pro", config.hardware_context)
            self.assertFalse(config.include_full_workflow_metadata)

            graph_config = load_runtime_routing_config(
                "placement_compiler/examples/"
                f"{workflow_name}_runtime_router_workflow_context.yaml"
            )
            self.assertEqual(graph_config.workflow, workflow_name)
            self.assertEqual(graph_config.sample_size, 50)
            self.assertEqual(graph_config.repeats, 2)
            self.assertTrue(graph_config.include_full_workflow_metadata)
            self.assertIn("runtime-routing-workflow-context", str(graph_config.output))

    def test_structured_qa_runtime_configs_use_seed_56_final_outputs(self):
        current = load_runtime_routing_config(
            "placement_compiler/examples/qa_runtime_router_structured_v2.yaml"
        )
        graph = load_runtime_routing_config(
            "placement_compiler/examples/"
            "qa_runtime_router_workflow_context_structured_v2.yaml"
        )

        for config in (current, graph):
            self.assertEqual(config.workflow, "qa")
            self.assertEqual(config.sample_size, 50)
            self.assertEqual(config.repeats, 2)
            self.assertEqual(config.holdout_seed, 56)
            self.assertEqual(config.output.parent.name, "qa")
            self.assertEqual(config.source_results.parent.parent.name, "qa")
        self.assertFalse(current.include_full_workflow_metadata)
        self.assertTrue(graph.include_full_workflow_metadata)


if __name__ == "__main__":
    unittest.main()
