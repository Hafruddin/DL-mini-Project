"""
Unit tests for the disposal and recycling recommendation module.
"""

import unittest
from config import TARGET_CLASSES
from recommendations import RECOMMENDATIONS, get_recommendation


class TestRecommendations(unittest.TestCase):

    def test_all_six_target_classes_present(self):
        for cname in TARGET_CLASSES:
            self.assertIn(cname, RECOMMENDATIONS, f"Missing recommendation entry for {cname}")

    def test_recommendation_structure(self):
        for cname, data in RECOMMENDATIONS.items():
            self.assertIn("display_name", data)
            self.assertIn("icon", data)
            self.assertIn("short_description", data)
            self.assertIn("examples", data)
            self.assertIn("preparation_instructions", data)
            self.assertIn("disposal_instructions", data)
            self.assertIn("recycling_guidance", data)
            self.assertIn("safety_precautions", data)
            self.assertTrue(len(data["examples"]) > 0)
            self.assertTrue(len(data["preparation_instructions"]) > 0)

    def test_fallback_recommendation(self):
        fallback = get_recommendation("UnknownMaterial")
        self.assertEqual(fallback["display_name"], "UnknownMaterial")
        self.assertIn("disposal_instructions", fallback)


if __name__ == "__main__":
    unittest.main()
