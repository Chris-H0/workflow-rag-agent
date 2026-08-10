import unittest

from placement_compiler.experiments.holdout import select_disjoint_examples


class HoldoutExperimentTests(unittest.TestCase):
    def test_selected_examples_exclude_development_ids(self):
        examples = [{"id": str(index)} for index in range(20)]

        selected = select_disjoint_examples(
            examples,
            excluded_ids={"0", "1", "2"},
            sample_size=5,
            seed=43,
            example_id=lambda example: example["id"],
        )

        selected_ids = {example["id"] for example in selected}
        self.assertEqual(len(selected_ids), 5)
        self.assertTrue(selected_ids.isdisjoint({"0", "1", "2"}))


if __name__ == "__main__":
    unittest.main()
