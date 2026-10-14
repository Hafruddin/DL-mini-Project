"""
Inference and Prediction Service for Smart Waste Classification.
Handles model loading with singleton caching, input preprocessing, class mapping,
confidence scoring, and uncertainty detection.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import logging
from pathlib import Path
from typing import Dict, Any, Union
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

from config import (
    MODEL_PATH,
    CLASS_MAPPING_PATH,
    TARGET_CLASSES,
    CONFIDENCE_THRESHOLD
)
from preprocessing import (
    validate_image_file,
    prepare_for_inference,
    PreprocessingError
)
from recommendations import get_recommendation
from image_verifier import verify_waste_image

logger = logging.getLogger(__name__)

# Global cache for single-process memory reuse
_MODEL_CACHE = None
_CLASS_MAPPING_CACHE = None


def get_loaded_model() -> keras.Model:
    """Loads and caches the trained Keras model."""
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found at '{MODEL_PATH}'. "
                "Please run 'python train_model.py' to generate and train the model."
            )
        logger.info(f"Loading Keras model from {MODEL_PATH}...")
        _MODEL_CACHE = keras.models.load_model(MODEL_PATH)
    return _MODEL_CACHE


def get_class_mapping() -> Dict[int, str]:
    """Loads and caches the class mapping dictionary."""
    global _CLASS_MAPPING_CACHE
    if _CLASS_MAPPING_CACHE is None:
        if CLASS_MAPPING_PATH.exists():
            with open(CLASS_MAPPING_PATH, "r", encoding="utf-8") as f:
                raw_map = json.load(f)
                _CLASS_MAPPING_CACHE = {int(k): v for k, v in raw_map.items()}
        else:
            _CLASS_MAPPING_CACHE = {i: c for i, c in enumerate(TARGET_CLASSES)}
    return _CLASS_MAPPING_CACHE


def predict_waste(image_input: Union[str, Path, Image.Image, object]) -> Dict[str, Any]:
    """
    Executes input verification, screenshot/non-waste filtering, and CNN inference.

    Returns:
        Dict containing:
            - is_waste: bool (True if physical waste item, False if screenshot/blank/non-waste)
            - is_identified: bool (True if confident waste identification, False otherwise)
            - predicted_category: str
            - confidence_score: float (0.0 to 1.0)
            - confidence_percentage: float (0.0 to 100.0)
            - is_uncertain: bool (if below CONFIDENCE_THRESHOLD or unverified)
            - status: str ('success', 'screenshot_detected', 'blank_image', 'low_confidence', etc.)
            - error_message: Optional[str]
            - class_probabilities: Dict[str, float]
            - recommendation: Dict[str, Any]
    """
    # 1. Validate file format and size
    is_valid, err_msg = validate_image_file(image_input)
    if not is_valid:
        raise ValueError(err_msg)

    # Extract filename hint if available
    file_name = None
    if isinstance(image_input, (str, Path)):
        file_name = Path(image_input).name
    elif hasattr(image_input, "name"):
        file_name = getattr(image_input, "name", "")

    # 2. Visual & Structural Verification (Detect screenshots, UI, blanks, non-waste)
    is_waste_item, status_code, verification_msg = verify_waste_image(image_input, file_name=file_name)
    if not is_waste_item:
        rec = get_recommendation("Cannot Identify Item")
        return {
            "is_waste": False,
            "is_identified": False,
            "predicted_category": "Cannot Identify Item",
            "confidence_score": 0.0,
            "confidence_percentage": 0.0,
            "is_uncertain": True,
            "status": status_code,
            "error_message": f"Cannot identify the item: {verification_msg}",
            "class_probabilities": {c: 0.0 for c in TARGET_CLASSES},
            "recommendation": rec
        }

    # 3. Preprocess for CNN
    try:
        tensor = prepare_for_inference(image_input)
    except PreprocessingError as pe:
        raise ValueError(f"Image preprocessing failed: {pe}") from pe

    # 4. Model inference
    model = get_loaded_model()
    class_map = get_class_mapping()

    raw_preds = model.predict(tensor, verbose=0)[0]
    pred_idx = int(np.argmax(raw_preds))
    confidence_score = float(raw_preds[pred_idx])
    predicted_category = class_map.get(pred_idx, TARGET_CLASSES[pred_idx])

    # Class probability dictionary
    class_probabilities = {}
    for idx, prob in enumerate(raw_preds):
        cname = class_map.get(idx, TARGET_CLASSES[idx] if idx < len(TARGET_CLASSES) else f"Class_{idx}")
        class_probabilities[cname] = round(float(prob), 4)

    # Check uncertainty against configurable threshold
    is_uncertain = confidence_score < CONFIDENCE_THRESHOLD

    if is_uncertain:
        rec = get_recommendation("Cannot Identify Item")
        return {
            "is_waste": True,
            "is_identified": False,
            "predicted_category": "Cannot Identify Item",
            "confidence_score": round(confidence_score, 4),
            "confidence_percentage": round(confidence_score * 100.0, 2),
            "is_uncertain": True,
            "status": "low_confidence",
            "error_message": (
                f"Cannot identify the item: The model returned a low confidence score ({confidence_score * 100.0:.1f}%), "
                f"which is below the {int(CONFIDENCE_THRESHOLD * 100)}% certainty threshold. "
                "Please upload a clearer, well-lit photo of the physical waste item."
            ),
            "class_probabilities": class_probabilities,
            "recommendation": rec
        }

    # 5. Confident physical waste identification
    rec = get_recommendation(predicted_category)

    return {
        "is_waste": True,
        "is_identified": True,
        "predicted_category": predicted_category,
        "confidence_score": round(confidence_score, 4),
        "confidence_percentage": round(confidence_score * 100.0, 2),
        "is_uncertain": False,
        "status": "success",
        "error_message": None,
        "class_probabilities": class_probabilities,
        "recommendation": rec
    }
