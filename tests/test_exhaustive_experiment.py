import unittest

from placement_compiler.adapters.workflows import load_workflow_driver
from placement_compiler.core.catalogue import load_model_catalogue
from placement_compiler.core.validation import validate_plan
from placement_compiler.experiments.exhaustive import enumerate_valid_plans
from placement_compiler.pipeline.config import load_pipeline_config


class ExhaustiveExperimentTests(unittest.TestCase):
    def test_qa_space_contains_54_unique_valid_plans(self):
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

        self.assertEqual(len(plans), 54)
        self.assertEqual(plans[0].id, "placement-001")
        self.assertEqual(plans[0].source, "baseline")
        self.assertEqual(
            set(plans[0].assignments.values()),
            {config.strongest_cloud_endpoint},
        )
        self.assertEqual(
            len({tuple(sorted(plan.assignments.items())) for plan in plans}),
            54,
        )
        for plan in plans:
            validate_plan(plan, workflow, endpoints)

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
