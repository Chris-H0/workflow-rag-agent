from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from placement_compiler.adapters.workflows import load_workflow_driver
from placement_compiler.core.catalogue import load_model_catalogue
from placement_compiler.core.models import PlacementProposal
from placement_compiler.experiments.unprofiled_search import (
    build_prompt_context,
    run_unprofiled_search,
)
from placement_compiler.pipeline.config import load_pipeline_config


class FakeCompilerLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.messages = []

    def generate(self, messages, output_schema):
        self.messages.append(messages)
        return self.responses.pop(0)


class UnprofiledSearchTests(unittest.TestCase):
    def setUp(self):
        self.config = load_pipeline_config(
            "placement_compiler/examples/qa_pipeline_structured_v2_01.yaml"
        )
        self.workflow = load_workflow_driver("qa").metadata()
        self.endpoints = load_model_catalogue(self.config.models)

    def test_no_metadata_only_strips_workflow_metadata(self):
        full = build_prompt_context(self.workflow, self.endpoints, "full")
        without = build_prompt_context(self.workflow, self.endpoints, "none")

        self.assertEqual(full.endpoints, without.endpoints)
        self.assertEqual(
            [unit.id for unit in full.workflow.placement_units],
            [unit.id for unit in without.workflow.placement_units],
        )
        self.assertTrue(full.workflow.edges)
        self.assertFalse(without.workflow.edges)
        for unit in without.workflow.placement_units:
            self.assertIsNone(unit.semantic_role)
            self.assertIsNone(unit.description)
            self.assertFalse(unit.tool_use)
            self.assertFalse(unit.structured_output_required)

    def test_search_has_no_baseline_or_feedback_and_allows_gggg(self):
        unit_ids = [unit.id for unit in self.workflow.placement_units]
        gggg = {unit_id: "gpt-5.5-cloud" for unit_id in unit_ids}
        mmmm = {unit_id: "gpt-4.1-mini-cloud" for unit_id in unit_ids}
        llm = FakeCompilerLLM(
            [
                PlacementProposal(assignments=gggg),
                PlacementProposal(assignments=gggg),
                PlacementProposal(assignments=mmmm),
            ]
        )

        with tempfile.TemporaryDirectory() as directory:
            output = run_unprofiled_search(
                config=self.config,
                metadata_mode="full",
                output_dir=Path(directory),
                sequence_count=1,
                proposal_count=2,
                compiler_llm=llm,
            )
            artifact = json.loads(output.read_text())

        self.assertFalse(artifact["baseline_supplied"])
        self.assertFalse(artifact["evaluated_results_supplied"])
        self.assertEqual(
            artifact["sequences"][0]["proposals"][0]["assignments"], gggg
        )
        self.assertEqual(
            artifact["sequences"][0]["proposals"][1]["assignments"], mmmm
        )
        self.assertEqual(
            artifact["compiler_metrics"]["duplicate_proposal_count"], 1
        )
        self.assertEqual(artifact["compiler_metrics"]["workflow_execution_count"], 0)
        for messages in llm.messages:
            payload = messages[1]["content"]
            self.assertIn('"evaluated_results": []', payload)
        self.assertIn(
            "excluded assignment maps",
            llm.messages[1][1]["content"],
        )
        self.assertIn(
            "using the measured evaluation results",
            llm.messages[0][0]["content"],
        )


if __name__ == "__main__":
    unittest.main()
