import unittest

from main.structured_targets import (
    aggregate_label_metrics,
    parse_seven_label_target,
    prose_to_seven_label_target,
    seven_label_target_to_prose,
)


FULL_PROSE = (
    "Breathing (Mild): A. Lips (Normal): B. Palate (Normal): C. "
    "Larynx (Moderate): D. Monotonicity (Mild): E. Tongue (Normal): F. "
    "Intelligibility (Mild): G."
)


class StructuredTargetsTests(unittest.TestCase):
    def test_prose_to_seven_label_roundtrip_labels(self) -> None:
        labels = prose_to_seven_label_target(FULL_PROSE)
        parsed = parse_seven_label_target(labels)
        self.assertEqual(parsed["Breathing"], "Mild")
        self.assertEqual(parsed["Intelligibility"], "Mild")
        self.assertEqual(len(labels.splitlines()), 7)

    def test_verbalize_has_all_slots(self) -> None:
        labels = prose_to_seven_label_target(FULL_PROSE)
        prose = seven_label_target_to_prose(labels)
        self.assertIn("Breathing (Mild):", prose)
        self.assertIn("Intelligibility (Mild):", prose)

    def test_label_metrics_perfect_match(self) -> None:
        labels = prose_to_seven_label_target(FULL_PROSE)
        agg = aggregate_label_metrics([(labels, labels)])
        self.assertEqual(agg["exact_match_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
