"""Verify local QA structured decisions before running paid experiments."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from placement_compiler.adapters.workflows import load_workflow_driver
from placement_compiler.core.catalogue import load_model_catalogue
from placement_compiler.core.models import PlacementPlan
from placement_compiler.pipeline.config import load_pipeline_config
from placement_compiler.profiling.runner import PlanEvaluator, deterministic_sample
from placement_compiler.runtime.endpoint_registry import EndpointRegistry, TraceCollector
from workflows.qa.agent.nodes import RetrievalDecision
from workflows.qa.agent.prompts import RETRIEVAL_DECISION_PROMPT
from workflows.qa.evaluation import prepare_runtime


REPO_ROOT = Path(__file__).resolve().parents[2]


def run_preflight(*, config_path: str | Path, decision_calls: int, workflow_calls: int) -> dict:
    if decision_calls < 0 or workflow_calls <= 0:
        raise ValueError("decision calls must be non-negative and workflow calls positive")

    _load_dotenvs()
    config = load_pipeline_config(config_path)
    if config.workflow != "qa":
        raise ValueError("the structured-decision preflight requires a QA config")

    driver = load_workflow_driver("qa")
    endpoints = load_model_catalogue(config.models)
    collector = TraceCollector()
    decisions: Counter[str] = Counter()
    if decision_calls:
        examples = deterministic_sample(
            driver.load_examples(config),
            sample_size=decision_calls,
            seed=config.seed,
            example_id=driver.example_id,
        )
        retriever_tool = prepare_runtime(examples)["retriever_tool"]
        registry = EndpointRegistry(endpoints)
        decision_model = registry.build_model(
            endpoint=registry.endpoint("qwen-local"),
            placement_unit_id="decide_after_retrieval",
            trace_collector=collector,
        ).with_structured_output(RetrievalDecision)

        for index, example in enumerate(examples, start=1):
            question = str(example["question"])
            context = str(retriever_tool.invoke({"query": question}))
            prompt = RETRIEVAL_DECISION_PROMPT.format(
                question=question,
                context=context,
            )
            parsed = decision_model.invoke([{"role": "user", "content": prompt}])
            decisions[parsed.decision] += 1
            print(
                f"Structured decision {index}/{decision_calls}: {parsed.decision}",
                flush=True,
            )

    decision_errors = [item.error for item in collector.invocations if item.error]
    if decision_errors:
        raise RuntimeError(f"local structured-decision errors: {decision_errors}")

    preflight_settings = config.model_copy(
        update={"sample_size": workflow_calls, "repeats": 1, "warmup_runs": 0}
    )
    evaluator = PlanEvaluator(
        workflow=driver.metadata(),
        endpoint_registry=EndpointRegistry(endpoints),
        driver=driver,
        settings=preflight_settings,
    )
    evaluator.prepare()
    plan = PlacementPlan(
        id="discarded-structured-decision-preflight",
        source="candidate",
        description="Discarded MQMG QA structured-decision preflight",
        assignments={
            "generate_query_or_respond": "gpt-4.1-mini-cloud",
            "decide_after_retrieval": "qwen-local",
            "rewrite_question": "gpt-4.1-mini-cloud",
            "generate_answer": "gpt-5.5-cloud",
        },
    )
    result, runs = evaluator.evaluate(plan)
    decision_invocations = [
        invocation
        for run in runs
        for invocation in run.trace.invocations
        if invocation.placement_unit_id == "decide_after_retrieval"
    ]
    if result.metrics["failed_run_count"]:
        errors = [run.error for run in runs if run.error]
        raise RuntimeError(f"end-to-end preflight failures: {errors}")
    if not decision_invocations:
        raise RuntimeError("end-to-end preflight never invoked the QA decision unit")
    if any(invocation.error for invocation in decision_invocations):
        raise RuntimeError("end-to-end preflight recorded a decision-unit error")

    return {
        "direct_decision_calls": len(collector.invocations),
        "direct_decision_errors": len(decision_errors),
        "decisions": dict(sorted(decisions.items())),
        "workflow_calls": len(runs),
        "workflow_failures": int(result.metrics["failed_run_count"]),
        "workflow_decision_invocations": len(decision_invocations),
    }


def _load_dotenvs() -> None:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env", override=True)
    load_dotenv(REPO_ROOT / "workflows" / "qa" / ".env", override=True)
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--decision-calls", type=int, default=50)
    parser.add_argument("--workflow-calls", type=int, default=10)
    args = parser.parse_args(argv)
    summary = run_preflight(
        config_path=args.config,
        decision_calls=args.decision_calls,
        workflow_calls=args.workflow_calls,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
