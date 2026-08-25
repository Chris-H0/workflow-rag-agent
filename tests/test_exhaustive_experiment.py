import tempfile
import unittest
from pathlib import Path

from placement_compiler.adapters.workflows import load_workflow_driver
from placement_compiler.core.catalogue import load_model_catalogue
from placement_compiler.core.models import PlacementPlan
from placement_compiler.core.validation import validate_plan
from placement_compiler.experiments.exhaustive import (
    _load_or_seed_results,
    enumerate_valid_plans,
)
from placement_compiler.pipeline.config import load_pipeline_config


class ExhaustiveExperimentTests(unittest.TestCase):
    def test_fresh_exhaustive_starts_without_reused_results(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            results, runs, reused_count = _load_or_seed_results(
                output_dir=Path(temporary_directory),
                source_results_path=None,
                selected_example_ids=["example-1"],
                plans=[],
                expected_runs_per_plan=1,
            )

        self.assertEqual(results, [])
        self.assertEqual(runs, [])
        self.assertEqual(reused_count, 0)

    def test_qa_space_contains_81_unique_valid_plans(self):
        config = load_pipeline_config(
            "placement_compiler/examples/qa_pipeline_initial.yaml"
        )
        workflow = load_workflow_driver(config.workflow).metadata()
        endpoints = load_model_catalogue(config.models)

        plans = enumerate_valid_plans(
            workflow,
            endpoints,
            config.strongest_cloud_endpoint,
        )

        self.assertEqual(len(plans), 81)
        self.assertEqual(plans[0].id, "placement-001")
        self.assertEqual(plans[0].source, "baseline")
        self.assertEqual(
            set(plans[0].assignments.values()),
            {config.strongest_cloud_endpoint},
        )
        self.assertEqual(
            len({tuple(sorted(plan.assignments.items())) for plan in plans}),
            81,
        )
        qwen_decision_plans = [
            plan
            for plan in plans
            if plan.assignments["decide_after_retrieval"] == "qwen-local"
        ]
        self.assertEqual(len(qwen_decision_plans), 27)
        for plan in plans:
            validate_plan(plan, workflow, endpoints)

        validate_plan(
            PlacementPlan(
                id="qwen-decision",
                assignments={
                    unit.id: (
                        "qwen-local"
                        if unit.id == "decide_after_retrieval"
                        else config.strongest_cloud_endpoint
                    )
                    for unit in workflow.placement_units
                },
            ),
            workflow,
            endpoints,
        )

    def test_code_space_contains_81_unique_valid_plans(self):
        config = load_pipeline_config(
            "placement_compiler/examples/code_pipeline_initial.yaml"
        )
        workflow = load_workflow_driver(config.workflow).metadata()
        endpoints = load_model_catalogue(config.models)

        plans = enumerate_valid_plans(
            workflow,
            endpoints,
            config.strongest_cloud_endpoint,
        )

        self.assertEqual(len(plans), 81)
        self.assertEqual(plans[0].source, "baseline")
        self.assertEqual(
            len({tuple(sorted(plan.assignments.items())) for plan in plans}),
            81,
        )
        for plan in plans:
            validate_plan(plan, workflow, endpoints)


if __name__ == "__main__":
    unittest.main()
