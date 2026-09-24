"""PhytoGATE Vision Preprocessing and Inspection Overlays Module.

Handles robust multi-format image decoding, RGB normalization, resize optimization,
and local computer vision inspection visualizations.

CRITICAL ARCHITECTURAL GUARANTEE:
All visual overlays generated herein are clearly documented as local computer vision
heuristics (HSV/color-space segmentation and contrast heatmaps). They are NOT claimed
to be Gemini neural attention maps.
"""

import base64
import hashlib
import io
from typing import Dict, Tuple, Any

import cv2
import numpy as np
from PIL import Image, ImageOps


def normalize_image(
    raw_image_bytes: bytes,
    max_long_edge: int = 1024,
    jpeg_quality: int = 85
) -> Tuple[bytes, int, int, str]:
    """Decodes, normalizes to RGB, resizes, and encodes as high-efficiency JPEG.

    Args:
        raw_image_bytes: Raw bytes from uploaded file (JPG, PNG, WEBP, etc.)
        max_long_edge: Maximum allowed dimension for height or width
        jpeg_quality: Quality parameter for normalized JPEG output

    Returns:
        (normalized_jpeg_bytes, width, height, sha256_hash)

    Raises:
        ValueError: If image cannot be decoded or is corrupted
    """
    if not raw_image_bytes:
        raise ValueError("Image bytes are empty.")

    try:
        image = Image.open(io.BytesIO(raw_image_bytes))
        # Handle EXIF orientation if present
        image = ImageOps.exif_transpose(image)
    except Exception as e:
        raise ValueError(f"Unsupported or corrupted image file: {str(e)}")

    # Convert to RGB (handles RGBA, Palette, Grayscale, CMYK, etc.)
    if image.mode != "RGB":
        image = image.convert("RGB")

    width, height = image.size

    # Resize so longest edge is <= max_long_edge, maintaining aspect ratio
    longest_edge = max(width, height)
    if longest_edge > max_long_edge:
        scale_factor = max_long_edge / float(longest_edge)
        new_width = int(round(width * scale_factor))
        new_height = int(round(height * scale_factor))
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        width, height = image.size

    # Encode as JPEG
    out_buffer = io.BytesIO()
    image.save(out_buffer, format="JPEG", quality=jpeg_quality, optimize=True)
    normalized_jpeg_bytes = out_buffer.getvalue()

    # Deterministic SHA-256 hash of normalized bytes
    sha256_hash = hashlib.sha256(normalized_jpeg_bytes).hexdigest()

    return normalized_jpeg_bytes, width, height, sha256_hash


def generate_cv_overlays(normalized_jpeg_bytes: bytes) -> Dict[str, Any]:
    """Generates local computer vision inspection overlays.

    These are heuristic inspection tools providing foliar lesion segmentation
    and contrast heatmaps. They are explicitly labeled as local CV overlays
    and NOT neural attention maps.
    """
    try:
        # Decode normalized JPEG with OpenCV
        nparr = np.frombuffer(normalized_jpeg_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return {}

        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        h, w = img_bgr.shape[:2]

        # 1. Foliar Lesion Segmentation (Local Heuristic)
        # Identify brown/yellow/necrotic lesions vs healthy green foliage
        # Healthy green range in HSV
        lower_green = np.array([25, 35, 35])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(img_hsv, lower_green, upper_green)

        # Foliage overall mask (green or yellow/brown plant parts)
        lower_foliage = np.array([10, 25, 25])
        upper_foliage = np.array([95, 255, 255])
        foliage_mask = cv2.inRange(img_hsv, lower_foliage, upper_foliage)

        # Lesions are non-green parts within foliage or high-contrast spots
        lesion_mask = cv2.bitwise_and(foliage_mask, cv2.bitwise_not(green_mask))

        # Create overlay: Highlight lesions in red/amber on original image
        overlay_lesion = img_bgr.copy()
        overlay_lesion[lesion_mask > 0] = [0, 0, 230]  # Red in BGR
        blended_lesion = cv2.addWeighted(img_bgr, 0.65, overlay_lesion, 0.35, 0)

        # 2. Chlorosis & Contrast Heatmap (Local Heuristic)
        # Calculate saturation and value gradient
        v_channel = img_hsv[:, :, 2]
        s_channel = img_hsv[:, :, 1]
        contrast_map = cv2.absdiff(v_channel, s_channel)
        contrast_norm = cv2.normalize(contrast_map, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap(contrast_norm, cv2.COLORMAP_JET)
        blended_heatmap = cv2.addWeighted(img_bgr, 0.55, heatmap, 0.45, 0)

        # Encode overlays to base64 JPEG
        _, lesion_jpg = cv2.imencode(".jpg", blended_lesion, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        _, heatmap_jpg = cv2.imencode(".jpg", blended_heatmap, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

        lesion_b64 = base64.b64encode(lesion_jpg).decode("utf-8")
        heatmap_b64 = base64.b64encode(heatmap_jpg).decode("utf-8")

        return {
            "lesion_segmentation": {
                "title": "Local CV Inspection — Foliar Lesion Segmentation",
                "type": "LOCAL_HEURISTIC_CV",
                "disclaimer": "Heuristic computer vision segmentation for leaf inspection. NOT Gemini neural attention weights.",
                "image_data": f"data:image/jpeg;base64,{lesion_b64}"
            },
            "contrast_heatmap": {
                "title": "Local CV Inspection — Chlorosis & Necrosis Contrast Heatmap",
                "type": "LOCAL_HEURISTIC_CV",
                "disclaimer": "Heuristic color contrast heatmap highlighting necrotic and chlorotic tissue gradients. NOT Gemini neural attention weights.",
                "image_data": f"data:image/jpeg;base64,{heatmap_b64}"
            }
        }
    except Exception:
        # If any CV processing fails, do not break the diagnostic pipeline
        return {}
