"""
Unit and integration tests for the prediction service.
"""

import unittest
from pathlib import Path
from PIL import Image

from config import MODEL_PATH, TARGET_CLASSES
from predictor import predict_waste, get_loaded_model


class TestPrediction(unittest.TestCase):

    def setUp(self):
        self.test_img_path = Path("tests/test_pred_img.jpg")
        self.test_img_path.parent.mkdir(parents=True, exist_ok=True)
        img = Image.new("RGB", (128, 128), color=(50, 120, 200))
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

        self.assertIn(result["predicted_category"], TARGET_CLASSES)
        self.assertTrue(0.0 <= result["confidence_score"] <= 1.0)
        self.assertEqual(len(result["class_probabilities"]), len(TARGET_CLASSES))


if __name__ == "__main__":
    unittest.main()
