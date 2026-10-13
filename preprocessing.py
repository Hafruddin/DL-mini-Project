"""
Image preprocessing and validation pipeline for training and inference.
Guarantees identical normalization and transformations between training and production.
"""

from pathlib import Path
from typing import Tuple, Union
import numpy as np
from PIL import Image

from config import (
    IMG_HEIGHT,
    IMG_WIDTH,
    CHANNELS,
    INPUT_SHAPE
)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB limit


class PreprocessingError(Exception):
    """Custom exception raised when image preprocessing fails."""
    pass


def validate_image_file(file_path_or_buffer: Union[str, Path, object]) -> Tuple[bool, str]:
    """
    Validates that the file exists, does not exceed the size limit,
    and has an allowed extension.
    """
    if isinstance(file_path_or_buffer, (str, Path)):
        path = Path(file_path_or_buffer)
        if not path.exists():
            return False, f"File does not exist: {path.name}"
        if path.suffix.lower() not in ALLOWED_EXTENSIONS:
            return False, f"Unsupported file format '{path.suffix}'. Allowed formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        if path.stat().st_size > MAX_FILE_SIZE_BYTES:
            return False, f"File exceeds maximum allowed size of {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB."
        return True, "Valid"
    
    # If already a PIL Image instance
    if isinstance(file_path_or_buffer, Image.Image):
        return True, "Valid"

    # Handle Streamlit UploadedFile or file-like buffer objects
    if hasattr(file_path_or_buffer, "name"):
        name = getattr(file_path_or_buffer, "name", "")
        if name:
            ext = Path(name).suffix.lower()
            if ext and ext not in ALLOWED_EXTENSIONS:
                return False, f"Unsupported file format '{ext}'. Allowed formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}"

    if hasattr(file_path_or_buffer, "size"):
        file_size = getattr(file_path_or_buffer, "size")
        # Ensure file_size is an integer byte length (e.g. UploadedFile.size), not a PIL dimension tuple
        if isinstance(file_size, int) and file_size > MAX_FILE_SIZE_BYTES:
            return False, f"Uploaded file exceeds maximum allowed size of {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB."
    
    return True, "Valid"


def load_and_preprocess_image(
    image_source: Union[str, Path, Image.Image, object],
    target_size: Tuple[int, int] = (IMG_HEIGHT, IMG_WIDTH)
) -> np.ndarray:
    """
    Loads an image from path, buffer, or PIL object, ensures RGB 3-channel mode,
    resizes to target dimensions using high-quality Lanczos resampling,
    and normalizes pixel values to [0.0, 1.0] float32.

    Returns:
        np.ndarray of shape (target_size[0], target_size[1], 3)
    """
    try:
        if isinstance(image_source, Image.Image):
            pil_img = image_source
        else:
            pil_img = Image.open(image_source)

        # Force convert to RGB (handles RGBA, grayscale L, palette P, CMYK)
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        # Resize
        pil_img = pil_img.resize(target_size, Image.Resampling.LANCZOS)

        # Convert to numpy array and scale to [0, 1]
        img_array = np.asarray(pil_img, dtype=np.float32) / 255.0

        if img_array.shape != (target_size[0], target_size[1], CHANNELS):
            raise PreprocessingError(f"Unexpected image shape: {img_array.shape}. Expected: {INPUT_SHAPE}")

        return img_array

    except Exception as e:
        raise PreprocessingError(f"Failed to process image: {str(e)}") from e


def prepare_for_inference(
    image_source: Union[str, Path, Image.Image, object]
) -> np.ndarray:
    """
    Prepares a single image for model prediction by preprocessing it and
    adding the batch dimension: returns shape (1, 128, 128, 3).
    """
    img_array = load_and_preprocess_image(image_source)
    return np.expand_dims(img_array, axis=0)
