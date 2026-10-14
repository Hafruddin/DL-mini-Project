"""
Unit tests for image verification and screenshot detection.
"""

import unittest
from pathlib import Path
from PIL import Image
import numpy as np

from image_verifier import (
    is_blank_or_uniform_image,
    is_digital_screenshot_or_ui,
    verify_waste_image
)
from config import TARGET_CLASSES


class TestImageVerifier(unittest.TestCase):

    def test_blank_white_image(self):
        img = Image.new("RGB", (100, 100), (255, 255, 255))
        is_blank, msg = is_blank_or_uniform_image(img)
        self.assertTrue(is_blank)
        self.assertIn("uniform", msg.lower())

    def test_blank_black_image(self):
        img = Image.new("RGB", (100, 100), (0, 0, 0))
        is_blank, msg = is_blank_or_uniform_image(img)
        self.assertTrue(is_blank)

    def test_solid_color_image(self):
        img = Image.new("RGB", (100, 100), (50, 120, 200))
        is_blank, msg = is_blank_or_uniform_image(img)
        self.assertTrue(is_blank)

    def test_screenshot_detection_on_dashboard(self):
        sc_path = Path("tests/cropped_user_screenshot.png")
        if not sc_path.exists():
            self.skipTest("Screenshot sample not found.")
        img = Image.open(sc_path)
        is_sc, msg = is_digital_screenshot_or_ui(img, file_name=sc_path.name)
        self.assertTrue(is_sc)
        self.assertIn("screenshot", msg.lower())

    def test_physical_waste_samples_pass_verification(self):
        for cname in TARGET_CLASSES:
            pattern = f"sample_{cname.lower().replace(' ', '_')}*.jpg"
            samples = list(Path("test_samples").glob(pattern))
            if samples:
                is_valid, status, msg = verify_waste_image(samples[0])
                self.assertTrue(is_valid, f"Failed for {samples[0].name}: {msg}")
                self.assertEqual(status, "valid_physical_item")


if __name__ == "__main__":
    unittest.main()
