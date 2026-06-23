import unittest

from main.structured_report import (
    aggregate_structure_metrics,
    force_seven_slot_template,
    parse_slot_descriptions,
    pick_best_candidate,
    structure_metrics_for_text,
)


class StructuredReportTests(unittest.TestCase):
    def test_force_template_fills_missing_slots(self) -> None:
        raw = "Breathing (Mild): Reduced breath support during speech."
        forced = force_seven_slot_template(raw)
        metrics = structure_metrics_for_text(forced)
        self.assertTrue(metrics["all_7_slots"])
        self.assertEqual(metrics["category_coverage"], 1.0)
        self.assertIn("Breathing (Mild): Reduced breath support during speech.", forced)
        self.assertIn("Lips (Normal): Within normal limits.", forced)

    def test_parse_slot_descriptions_extracts_multiple_categories(self) -> None:
        text = (
            "Breathing (Mild): A. "
            "Lips (Normal): B. "
            "Palate (Moderate): C."
        )
        slots = parse_slot_descriptions(text)
        self.assertEqual(slots["Breathing"], ("Mild", "A."))
        self.assertEqual(slots["Lips"], ("Normal", "B."))
        self.assertEqual(slots["Palate"], ("Moderate", "C."))

    def test_pick_best_candidate_prefers_more_slots(self) -> None:
        candidates = [
            "Breathing (Mild): only one slot.",
            (
                "Breathing (Mild): A. Lips (Normal): B. Palate (Normal): C. "
                "Larynx (Normal): D. Monotonicity (Normal): E. Tongue (Normal): F. "
                "Intelligibility (Normal): G."
            ),
        ]
        idx, chosen, metrics = pick_best_candidate(candidates)
        self.assertEqual(idx, 1)
        self.assertTrue(metrics["all_7_slots"])
        self.assertEqual(chosen, candidates[1])

    def test_aggregate_structure_metrics_by_group(self) -> None:
        texts = [
            force_seven_slot_template("Breathing (Mild): x."),
            "Breathing (Mild): only one.",
        ]
        groups = ["PD", "HC"]
        report = aggregate_structure_metrics(texts, groups)
        self.assertEqual(report["n"], 2)
        self.assertAlmostEqual(report["category_coverage"], 0.5714, places=3)
        self.assertIn("by_group", report)
        self.assertEqual(report["by_group"]["PD"]["all_7_slots_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
