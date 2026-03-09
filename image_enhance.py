from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def _pil_to_bgr(img: Image.Image) -> np.ndarray:
    rgb = np.array(img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _bgr_to_pil(arr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _resize_keep_aspect(img: Image.Image, max_dim: int) -> Image.Image:
    w, h = img.size
    longest = max(w, h)
    if longest <= max_dim:
        return img
    scale = max_dim / float(longest)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return img.resize((new_w, new_h), Image.LANCZOS)


def make_enhanced_display_image(img: Image.Image, max_dim: int = 1600) -> Image.Image:
    """
    Conservative enhancement only.
    Goal:
    - improve readability slightly
    - DO NOT turn ghost/unlit segment templates into active-looking digits
    """
    img = _resize_keep_aspect(img, max_dim=max_dim)
    bgr = _pil_to_bgr(img)

    # Mild denoise only
    denoised = cv2.bilateralFilter(bgr, d=5, sigmaColor=20, sigmaSpace=20)

    # LAB luminance enhancement, but very conservative
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Lower CLAHE aggressiveness
    clahe = cv2.createCLAHE(clipLimit=1.4, tileGridSize=(8, 8))
    l2 = clahe.apply(l)

    # Blend original luminance with enhanced luminance to avoid over-boosting ghost slots
    l_blend = cv2.addWeighted(l, 0.72, l2, 0.28, 0)

    lab2 = cv2.merge([l_blend, a, b])
    out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    # NO global brightness boost
    # NO aggressive sharpening
    # NO post-enhancement upscale that exaggerates segment outlines

    # Very mild upscale only if image is genuinely small
    h, w = out.shape[:2]
    if max(h, w) < 900:
        out = cv2.resize(
            out,
            None,
            fx=1.25,
            fy=1.25,
            interpolation=cv2.INTER_CUBIC,
        )

    return _bgr_to_pil(out)