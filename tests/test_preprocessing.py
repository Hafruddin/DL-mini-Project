"""
Unit tests for the image preprocessing pipeline.
"""

import unittest
from pathlib import Path
import numpy as np
from PIL import Image

from preprocessing import (
    validate_image_file,
    load_and_preprocess_image,
    prepare_for_inference,
    PreprocessingError
)
from config import IMG_HEIGHT, IMG_WIDTH, INPUT_SHAPE


class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        self.test_img_path = Path("tests/test_temp_img.jpg")
        self.test_img_path.parent.mkdir(parents=True, exist_ok=True)
        # Create a small dummy RGB test image
        img = Image.new("RGB", (64, 64), color=(200, 100, 50))
        img.save(self.test_img_path)

    def tearDown(self):
        if self.test_img_path.exists():
            self.test_img_path.unlink()

    def test_validate_image_file_valid(self):
        is_valid, msg = validate_image_file(self.test_img_path)
        self.assertTrue(is_valid)
        self.assertEqual(msg, "Valid")

    def test_validate_image_file_unsupported_extension(self):
        invalid_path = Path("tests/dummy.txt")
        invalid_path.touch()
        try:
            is_valid, msg = validate_image_file(invalid_path)
            self.assertFalse(is_valid)
            self.assertIn("Unsupported", msg)
        finally:
            if invalid_path.exists():
                invalid_path.unlink()

    def test_load_and_preprocess_image_output_shape(self):
        processed = load_and_preprocess_image(self.test_img_path)
        self.assertEqual(processed.shape, (IMG_HEIGHT, IMG_WIDTH, 3))
        self.assertTrue(processed.dtype == np.float32)
        # Verify range [0, 1]
        self.assertTrue(np.all(processed >= 0.0) and np.all(processed <= 1.0))

    def test_prepare_for_inference_batch_dimension(self):
        tensor = prepare_for_inference(self.test_img_path)
        self.assertEqual(tensor.shape, (1, IMG_HEIGHT, IMG_WIDTH, 3))


if __name__ == "__main__":
    unittest.main()
