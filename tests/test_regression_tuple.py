"""
Regression test specifically verifying fix for:
TypeError: '>' not supported between instances of 'tuple' and 'int'
"""
import unittest
from pathlib import Path
from PIL import Image
import numpy as np

from preprocessing import validate_image_file, prepare_for_inference
from predictor import predict_waste


class TestRegressionTupleVsInt(unittest.TestCase):

    def test_pil_image_size_tuple_handling(self):
        """
        Verify that passing PIL Images of various dimensions (where img.size is a tuple)
        never raises TypeError: '>' not supported between instances of 'tuple' and 'int'.
        """
        sizes_to_test = [(128, 128), (612, 408), (256, 512), (64, 64), (1024, 768)]
        for w, h in sizes_to_test:
            with self.subTest(width=w, height=h):
                img = Image.new("RGB", (w, h), color=(100, 150, 200))
                # Validate should return True without throwing
                is_valid, msg = validate_image_file(img)
                self.assertTrue(is_valid)
                self.assertEqual(msg, "Valid")

                # Inference should execute without throwing
                result = predict_waste(img)
                self.assertIn("predicted_category", result)
                self.assertIn("confidence_score", result)
                self.assertTrue(isinstance(result["confidence_score"], float))

    def test_rgba_and_grayscale_images(self):
        """Verify RGBA and grayscale PIL images are converted cleanly to RGB tensors."""
        # RGBA
        rgba_img = Image.new("RGBA", (150, 150), color=(100, 150, 200, 255))
        res_rgba = predict_waste(rgba_img)
        self.assertIn("predicted_category", res_rgba)

        # Grayscale
        gray_img = Image.new("L", (150, 150), color=128)
        res_gray = predict_waste(gray_img)
        self.assertIn("predicted_category", res_gray)


if __name__ == "__main__":
    unittest.main()
