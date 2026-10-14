"""
Unit and integration tests for the prediction service.
"""

import unittest
import shutil
from pathlib import Path
from PIL import Image

from config import MODEL_PATH, TARGET_CLASSES
from predictor import predict_waste, get_loaded_model


class TestPrediction(unittest.TestCase):

    def setUp(self):
        self.test_img_path = Path("tests/test_pred_img.jpg")
        self.test_img_path.parent.mkdir(parents=True, exist_ok=True)
        # Use valid test sample
        sample_src = Path("test_samples/sample_plastic_bottle.jpg")
        if sample_src.exists():
            shutil.copy(sample_src, self.test_img_path)
        else:
            # Fallback realistic sample
            img = Image.new("RGB", (128, 128), color=(70, 140, 200))
            img.save(self.test_img_path)

    def tearDown(self):
        if self.test_img_path.exists():
            self.test_img_path.unlink()

    def test_model_loading_and_architecture(self):
        if not MODEL_PATH.exists():
            self.skipTest("Trained model not yet present on disk.")
        model = get_loaded_model()
        self.assertIsNotNone(model)
        self.assertEqual(model.output_shape[-1], len(TARGET_CLASSES))

    def test_predict_waste_output_structure(self):
        if not MODEL_PATH.exists():
            self.skipTest("Trained model not yet present on disk.")
        result = predict_waste(self.test_img_path)

        self.assertIn("predicted_category", result)
        self.assertIn("confidence_score", result)
        self.assertIn("confidence_percentage", result)
        self.assertIn("is_uncertain", result)
        self.assertIn("class_probabilities", result)
        self.assertIn("recommendation", result)
        self.assertIn("is_waste", result)
        self.assertIn("is_identified", result)

        self.assertIn(result["predicted_category"], TARGET_CLASSES)
        self.assertTrue(0.0 <= result["confidence_score"] <= 1.0)
        self.assertEqual(len(result["class_probabilities"]), len(TARGET_CLASSES))

    def test_predict_waste_on_screenshot_returns_cannot_identify(self):
        """Verifies that digital screenshots are identified as Cannot Identify Item."""
        screenshot_path = Path("tests/cropped_user_screenshot.png")
        if not screenshot_path.exists():
            self.skipTest("Screenshot sample not found.")
        
        result = predict_waste(screenshot_path)
        self.assertFalse(result["is_waste"])
        self.assertFalse(result["is_identified"])
        self.assertEqual(result["predicted_category"], "Cannot Identify Item")
        self.assertEqual(result["status"], "screenshot_detected")
        self.assertIsNotNone(result["error_message"])
        self.assertIn("screenshot", result["error_message"].lower())

    def test_predict_waste_on_blank_image_returns_cannot_identify(self):
        """Verifies that completely blank/uniform images are identified as Cannot Identify Item."""
        blank_img = Image.new("RGB", (128, 128), (255, 255, 255))
        result = predict_waste(blank_img)
        self.assertFalse(result["is_waste"])
        self.assertFalse(result["is_identified"])
        self.assertEqual(result["predicted_category"], "Cannot Identify Item")
        self.assertEqual(result["status"], "blank_image")

    def test_all_preset_samples_predicted_correctly(self):
        """Verifies that all 6 preset physical waste samples predict their valid categories."""
        for cname in TARGET_CLASSES:
            pattern = f"sample_{cname.lower().replace(' ', '_')}*.jpg"
            sample_files = list(Path("test_samples").glob(pattern))
            if sample_files:
                res = predict_waste(sample_files[0])
                self.assertTrue(res["is_waste"])
                self.assertTrue(res["is_identified"])
                self.assertEqual(res["predicted_category"], cname)


if __name__ == "__main__":
    unittest.main()
