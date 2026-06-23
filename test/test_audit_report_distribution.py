import unittest

from main.audit_report_distribution import (
    _category_presence_rates,
    _compare_splits,
    _severity_histogram,
    audit_dataframe,
)
import pandas as pd


FULL_REPORT = (
    "Breathing (Mild): A. Lips (Normal): B. Palate (Normal): C. "
    "Larynx (Moderate): D. Monotonicity (Mild): E. Tongue (Normal): F. "
    "Intelligibility (Mild): G."
)
PARTIAL = "Breathing (Mild): reduced support."


class AuditReportDistributionTests(unittest.TestCase):
    def test_full_report_has_full_coverage(self) -> None:
        rates = _category_presence_rates([FULL_REPORT])
        self.assertEqual(rates["Breathing"], 1.0)
        self.assertEqual(rates["Intelligibility"], 1.0)

    def test_partial_report_presence(self) -> None:
        rates = _category_presence_rates([PARTIAL])
        self.assertEqual(rates["Breathing"], 1.0)
        self.assertEqual(rates["Lips"], 0.0)

    def test_compare_splits_flags_weak_synthetic(self) -> None:
        synthetic = {
            "mean_category_coverage": 0.14,
            "all_7_slots_rate": 0.0,
            "category_presence_rate": {"Breathing": 1.0, "Lips": 0.0},
            "duplicates": {"duplicate_rate": 0.5},
        }
        real = {
            "mean_category_coverage": 1.0,
            "all_7_slots_rate": 1.0,
            "category_presence_rate": {"Breathing": 1.0, "Lips": 1.0},
            "duplicates": {"duplicate_rate": 0.0},
        }
        cmp = _compare_splits(synthetic, real)
        self.assertLess(cmp["coverage_delta_synthetic_minus_real"], 0)
        self.assertIn("Lips", cmp["categories_well_in_real_but_weak_in_synthetic"])

    def test_audit_dataframe_structure(self) -> None:
        df = pd.DataFrame(
            {
                "target_text": [FULL_REPORT, PARTIAL],
                "is_real": [True, True],
                "group": ["PD", "HC"],
            }
        )
        out = audit_dataframe(df, max_duplicate_examples=2)
        self.assertEqual(out["n"], 2)
        self.assertGreater(out["mean_category_coverage"], 0.5)
        self.assertIn("by_group", out)


if __name__ == "__main__":
    unittest.main()
