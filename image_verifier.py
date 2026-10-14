"""
Image verification and Out-of-Distribution / Screenshot detection module.
Detects digital screenshots, computer/mobile user interfaces, blank images,
and non-waste graphics before or alongside deep learning inference.
"""

from pathlib import Path
from typing import Tuple, Union, Optional
import numpy as np
from PIL import Image

# Common keywords that strongly indicate digital screen capture
SCREENSHOT_KEYWORDS = {
    "screenshot",
    "screen shot",
    "screen_shot",
    "screen-shot",
    "capture",
    "cleanshot",
    "snip",
    "screen",
    "display",
    "dashboard"
}


def is_blank_or_uniform_image(img: Image.Image, std_threshold: float = 3.0) -> Tuple[bool, str]:
    """
    Checks if an image is completely uniform, blank, solid color, or has near-zero pixel variance.
    Camera photos always have spatial sensor noise; solid or blank images have spatial std near 0.
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    arr = np.asarray(img, dtype=np.float32)
    # Check spatial variance across image dimensions for each channel
    spatial_stds = np.std(arr, axis=(0, 1))
    max_spatial_std = float(np.max(spatial_stds))
    
    if max_spatial_std < std_threshold:
        return True, f"Image is completely uniform or solid color (spatial variance std: {max_spatial_std:.2f} < {std_threshold})."
    
    # Check if image is virtually pitch black or blown-out white
    mean_val = float(np.mean(arr))
    if mean_val < 5.0:
        return True, "Image is pitch black with no distinguishable visual features."
    if mean_val > 252.0 and max_spatial_std < 5.0:
        return True, "Image is solid white with no distinguishable visual features."
        
    return False, "Image has sufficient pixel variance."


def is_digital_screenshot_or_ui(
    img: Image.Image,
    file_name: Optional[str] = None
) -> Tuple[bool, str]:
    """
    Analyzes visual statistics to determine whether an image is a digital screenshot,
    software interface, or digital document rather than a photograph of physical waste.

    Key Indicators:
    1. Zero-variance / flat block patches (cameras always have sensor noise; digital screens have flat panels).
    2. Discrete color concentration (UI uses discrete hex colors; photos have natural gradients).
    3. Axis-aligned edge dominance (UI borders and text rows are strictly 0 and 90 degrees).
    4. Filename metadata cues (e.g. 'Screenshot', 'Screen Shot', 'Capture').
    """
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Check filename cues if available
    if file_name:
        name_lower = Path(file_name).name.lower()
        for kw in SCREENSHOT_KEYWORDS:
            if kw in name_lower:
                # Strong prior hint - combine with visual verification
                pass

    arr = np.asarray(img, dtype=np.float32)
    h, w, _ = arr.shape

    # 1. Pixel flatness in small blocks (8x8)
    # Camera sensors always exhibit optical photon and sensor noise (std >= 1.5).
    # Digital graphics, browser pages, UI screenshots have large regions where std < 0.8.
    patch_size = 8
    ph = h // patch_size
    pw = w // patch_size
    if ph >= 4 and pw >= 4:
        trimmed = arr[:ph * patch_size, :pw * patch_size]
        blocks = trimmed.reshape(ph, patch_size, pw, patch_size, 3).swapaxes(1, 2).reshape(-1, patch_size, patch_size, 3)
        block_stds = np.std(blocks, axis=(1, 2, 3))
        flat_ratio = float(np.mean(block_stds < 0.8))
    else:
        flat_ratio = 0.0

    # 2. Discrete color dominance (UI screens have large areas of exact identical colors)
    arr_uint = np.asarray(img, dtype=np.uint8).reshape(-1, 3)
    if len(arr_uint) > 40000:
        idx = np.random.RandomState(42).choice(len(arr_uint), 40000, replace=False)
        sub = arr_uint[idx]
    else:
        sub = arr_uint
    _, counts = np.unique(sub, axis=0, return_counts=True)
    counts_sorted = np.sort(counts)
    top1_ratio = float(counts_sorted[-1] / len(sub))
    top3_ratio = float(np.sum(counts_sorted[-3:]) / len(sub))

    # 3. Axis-aligned gradient edges (Sobel-like)
    # UI elements, browser tabs, dashboard panels have sharp horizontal/vertical edges.
    gray = np.dot(arr[..., :3], [0.2989, 0.5870, 0.1140])
    gx = np.abs(gray[:, 1:] - gray[:, :-1])
    gy = np.abs(gray[1:, :] - gray[:-1, :])
    min_h, min_w = min(gx.shape[0], gy.shape[0]), min(gx.shape[1], gy.shape[1])
    gx, gy = gx[:min_h, :min_w], gy[:min_h, :min_w]
    mag = np.sqrt(gx**2 + gy**2)
    sig = mag > 20.0
    if np.sum(sig) > 50:
        angles = np.arctan2(gy[sig], gx[sig]) * 180.0 / np.pi
        axis_aligned = (np.abs(angles) < 4.0) | (np.abs(angles - 90.0) < 4.0) | (np.abs(angles - 180.0) < 4.0)
        axis_ratio = float(np.mean(axis_aligned))
    else:
        axis_ratio = 0.0

    # Decision boundaries calibrated on real vs screenshot benchmarks
    if flat_ratio > 0.25:
        return True, f"Digital screenshot or document detected (flat color regions: {flat_ratio*100:.1f}%)."

    if flat_ratio > 0.08 and (top1_ratio > 0.35 or top3_ratio > 0.60):
        return True, f"Digital user interface or screenshot detected (flat background: {flat_ratio*100:.1f}%, top color: {top1_ratio*100:.1f}%)."

    if flat_ratio > 0.07 and axis_ratio > 0.55:
        return True, f"Digital UI layout detected (axis-aligned edges: {axis_ratio*100:.1f}%, flat panels: {flat_ratio*100:.1f}%)."

    if file_name:
        name_lower = Path(file_name).name.lower()
        if any(kw in name_lower for kw in SCREENSHOT_KEYWORDS) and (flat_ratio > 0.05 or top1_ratio > 0.25):
            return True, f"Screen capture detected from image attributes and metadata."

    return False, "Physical item photograph."


def verify_waste_image(
    image_input: Union[str, Path, Image.Image, object],
    file_name: Optional[str] = None
) -> Tuple[bool, str, str]:
    """
    Complete validation pipeline verifying that an image represents a physical item
    suitable for waste classification.

    Returns:
        (is_valid_waste: bool, status_code: str, user_friendly_message: str)
    """
    try:
        if isinstance(image_input, Image.Image):
            pil_img = image_input
        elif isinstance(image_input, (str, Path)):
            pil_img = Image.open(image_input)
            if not file_name:
                file_name = Path(image_input).name
        elif hasattr(image_input, "read"):
            pil_img = Image.open(image_input)
            if not file_name and hasattr(image_input, "name"):
                file_name = getattr(image_input, "name", "")
        else:
            return False, "invalid_input", "Unsupported image input format."

        # Check 1: Blank or uniform
        is_blank, blank_msg = is_blank_or_uniform_image(pil_img)
        if is_blank:
            return (
                False,
                "blank_image",
                "The uploaded image appears to be blank, solid color, or severely underexposed. "
                "Please upload a clear photograph of a physical discarded object."
            )

        # Check 2: Screenshot or digital UI
        is_screenshot, sc_msg = is_digital_screenshot_or_ui(pil_img, file_name=file_name)
        if is_screenshot:
            return (
                False,
                "screenshot_detected",
                "The uploaded image appears to be a digital screenshot, document, or software interface "
                "rather than a physical waste material. Please upload a camera photograph of actual "
                "packaging, containers, or discarded physical items."
            )

        return True, "valid_physical_item", "Physical item photograph verified."

    except Exception as e:
        return False, "processing_error", f"Could not analyze image for waste verification: {str(e)}"
