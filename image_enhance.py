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


def _apply_gamma(channel: np.ndarray, gamma: float) -> np.ndarray:
    gamma = max(0.1, float(gamma))
    table = np.array(
        [((index / 255.0) ** gamma) * 255.0 for index in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(channel, table)


def _unsharp_mask(
    arr: np.ndarray,
    *,
    sigma: float = 0.9,
    amount: float = 0.45,
) -> np.ndarray:
    blurred = cv2.GaussianBlur(arr, (0, 0), sigma)
    return cv2.addWeighted(arr, 1.0 + amount, blurred, -amount, 0)


def _normalize_gray(arr: np.ndarray) -> np.ndarray:
    return cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX)


def _color_dodge(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
    base_u16 = base.astype(np.uint16)
    blend_u16 = blend.astype(np.uint16)
    safe = np.maximum(1, 255 - blend_u16)
    dodged = np.minimum(255, (base_u16 * 255) // safe)
    return dodged.astype(np.uint8)


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
    l3 = _apply_gamma(l2, 0.94)

    # Blend original luminance with enhanced luminance to avoid over-boosting ghost slots
    l_blend = cv2.addWeighted(l, 0.62, l2, 0.20, 0)
    l_blend = cv2.addWeighted(l_blend, 0.74, l3, 0.26, 0)

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


def _to_three_channel(gray: np.ndarray) -> np.ndarray:
    if len(gray.shape) == 2:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return gray


def make_led_prelocalizer_image(img: Image.Image, max_dim: int = 1600) -> Image.Image:
    """
    LED-only pre-localization filter.
    Goal:
    - make the display strip stand out before crop selection
    - strengthen green active segments without turning the whole frame artificial
    - keep enough global structure for Gemini/local CV localization
    """
    img = _resize_keep_aspect(img, max_dim=max_dim)
    bgr = _pil_to_bgr(img)
    denoised = cv2.bilateralFilter(bgr, d=5, sigmaColor=18, sigmaSpace=18)
    hsv = cv2.cvtColor(denoised, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6))
    v_eq = clahe.apply(v)
    v_gamma = _apply_gamma(v_eq, 0.90)

    green_dominance = denoised[:, :, 1].astype(np.int16) - np.maximum(
        denoised[:, :, 0],
        denoised[:, :, 2],
    ).astype(np.int16)
    green_dominance = np.clip(green_dominance, 0, 255).astype(np.uint8)
    green_soft = cv2.GaussianBlur(green_dominance, (0, 0), 1.0)
    green_mask = cv2.threshold(green_soft, 14, 255, cv2.THRESH_BINARY)[1]
    green_mask = cv2.morphologyEx(
        green_mask,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), np.uint8),
        iterations=1,
    )

    boosted_v = cv2.addWeighted(v_eq, 0.60, v_gamma, 0.40, 0)
    boosted_v = cv2.addWeighted(boosted_v, 0.84, green_soft, 0.24, 0)
    background_v = cv2.addWeighted(v_eq, 0.88, v_gamma, 0.12, 0)
    boosted_v = np.where(green_mask > 0, boosted_v, background_v)

    dodge_support = cv2.convertScaleAbs(green_soft, alpha=0.95, beta=0)
    dodged_v = _color_dodge(v_eq, dodge_support)
    boosted_v = cv2.addWeighted(boosted_v, 0.84, dodged_v, 0.16, 0)

    s = cv2.addWeighted(s, 0.92, green_mask, 0.08, 0)
    out = cv2.cvtColor(cv2.merge([h, s, boosted_v.astype(np.uint8)]), cv2.COLOR_HSV2BGR)
    out = _unsharp_mask(out, sigma=0.8, amount=0.18)
    return _bgr_to_pil(out)


def make_led_risk_display_image(img: Image.Image, max_dim: int = 1600) -> Image.Image:
    """
    Stronger LED-specific variant used only for risky cases.
    Goal:
    - preserve narrow bright '1' on the left
    - increase contrast between active green segments and dim placeholders
    - avoid global over-sharpening
    """
    img = _resize_keep_aspect(img, max_dim=max_dim)
    bgr = _pil_to_bgr(img)

    denoised = cv2.bilateralFilter(bgr, d=5, sigmaColor=18, sigmaSpace=18)
    hsv = cv2.cvtColor(denoised, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    v_eq = clahe.apply(v)
    v_gamma = _apply_gamma(v_eq, 0.72)

    green_dominance = denoised[:, :, 1].astype(np.int16) - np.maximum(
        denoised[:, :, 0],
        denoised[:, :, 2],
    ).astype(np.int16)
    green_dominance = np.clip(green_dominance, 0, 255).astype(np.uint8)
    green_mask = cv2.threshold(green_dominance, 14, 255, cv2.THRESH_BINARY)[1]
    green_mask = cv2.morphologyEx(
        green_mask,
        cv2.MORPH_CLOSE,
        np.ones((2, 2), np.uint8),
        iterations=1,
    )
    green_soft = cv2.GaussianBlur(green_mask, (0, 0), 1.0)

    green_eq = clahe.apply(denoised[:, :, 1])
    segment_support = cv2.adaptiveThreshold(
        green_eq,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        -6,
    )
    segment_support = cv2.bitwise_and(segment_support, green_soft)

    boosted_v = cv2.addWeighted(v_eq, 0.28, v_gamma, 0.72, 0)
    boosted_v = cv2.addWeighted(boosted_v, 0.68, green_dominance, 0.52, 0)
    boosted_v = np.where(green_mask > 0, boosted_v, cv2.addWeighted(v_eq, 0.72, v_gamma, 0.28, 0))
    boosted_v = cv2.addWeighted(boosted_v, 0.80, segment_support, 0.24, 0)

    # LED-only color dodge: lift genuinely lit segments without globally brightening
    # the whole crop. This is intentionally fed by the green-dominance / segment
    # support maps so faint active slots get a local contrast boost.
    dodge_support = cv2.addWeighted(green_dominance, 0.92, segment_support, 0.58, 0)
    dodge_support = cv2.GaussianBlur(dodge_support, (0, 0), 0.8)
    dodge_support = cv2.convertScaleAbs(dodge_support, alpha=1.85, beta=14)
    dodge_support = cv2.max(dodge_support, green_soft)
    dodged_seed = cv2.addWeighted(v_eq, 0.18, v_gamma, 0.82, 0)
    dodged_v = _color_dodge(dodged_seed, dodge_support)
    boosted_v = cv2.addWeighted(boosted_v, 0.50, dodged_v, 0.50, 0)

    hsv2 = cv2.merge([h, s, boosted_v.astype(np.uint8)])
    out = cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)
    out = _unsharp_mask(out, sigma=0.8, amount=0.28)

    # Mild left-side emphasis helps the narrow leading 1 survive.
    left_w = max(1, out.shape[1] // 3)
    left_region = out[:, :left_w]
    left_hsv = cv2.cvtColor(left_region, cv2.COLOR_BGR2HSV)
    lh, ls, lv = cv2.split(left_hsv)
    lv = _apply_gamma(lv, 0.82)
    lv = cv2.convertScaleAbs(lv, alpha=1.04, beta=2)
    left_region = cv2.cvtColor(cv2.merge([lh, ls, lv]), cv2.COLOR_HSV2BGR)
    out[:, :left_w] = left_region

    return _bgr_to_pil(out)


def make_lcd_risk_display_image(img: Image.Image, max_dim: int = 1600) -> Image.Image:
    """
    Stronger LCD-specific variant used only for risky cases.
    Goal:
    - make dark LCD strokes stand out from the light background
    - help per-digit reading without introducing fake segments
    """
    img = _resize_keep_aspect(img, max_dim=max_dim)
    bgr = _pil_to_bgr(img)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 6, 7, 21)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6))
    gray_eq = clahe.apply(denoised)
    gray_gamma = _apply_gamma(gray_eq, 0.90)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    blackhat = cv2.morphologyEx(gray_gamma, cv2.MORPH_BLACKHAT, kernel)
    sharpened = _unsharp_mask(gray_gamma, sigma=0.9, amount=0.40)
    combined = cv2.addWeighted(sharpened, 0.58, blackhat, 0.95, 0)
    combined = _normalize_gray(combined)
    adaptive = cv2.adaptiveThreshold(
        combined,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        7,
    )
    adaptive = cv2.morphologyEx(
        adaptive,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
        iterations=1,
    )
    blended = cv2.addWeighted(gray_gamma, 0.42, combined, 0.40, 0)
    blended = cv2.addWeighted(blended, 0.72, adaptive, 0.28, 0)
    blended = _normalize_gray(blended)
    return _bgr_to_pil(_to_three_channel(blended))


def make_risky_display_variant(
    img: Image.Image,
    *,
    display_kind: str,
    max_dim: int = 1600,
) -> Image.Image:
    if display_kind == "led":
        return make_led_risk_display_image(img, max_dim=max_dim)
    if display_kind == "lcd":
        return make_lcd_risk_display_image(img, max_dim=max_dim)
    return make_enhanced_display_image(img, max_dim=max_dim)


def make_localizer_support_variant(
    img: Image.Image,
    *,
    display_kind: str,
    max_dim: int = 1600,
) -> Image.Image:
    img = _resize_keep_aspect(img, max_dim=max_dim)
    if display_kind == "led":
        return make_led_prelocalizer_image(img, max_dim=max_dim)

    if display_kind == "lcd":
        bgr = _pil_to_bgr(img)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.fastNlMeansDenoising(gray, None, 5, 7, 21)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6)).apply(gray)
        gray = _apply_gamma(gray, 0.88)
        blackhat = cv2.morphologyEx(
            gray,
            cv2.MORPH_BLACKHAT,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        )
        merged = cv2.addWeighted(gray, 0.68, blackhat, 0.95, 0)
        merged = _normalize_gray(merged)
        return _bgr_to_pil(_to_three_channel(merged))

    return make_enhanced_display_image(img, max_dim=max_dim)


def _expand_box(
    box: tuple[int, int, int, int],
    image_size: tuple[int, int],
    *,
    x_pad_frac: float,
    y_pad_frac: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    width, height = image_size
    bw = x2 - x1
    bh = y2 - y1
    x_pad = int(round(bw * x_pad_frac))
    y_pad = int(round(bh * y_pad_frac))
    return (
        max(0, x1 - x_pad),
        max(0, y1 - y_pad),
        min(width, x2 + x_pad),
        min(height, y2 + y_pad),
    )


def _box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_a + area_b - inter)


def _dedupe_boxes(
    candidates: list[tuple[float, tuple[int, int, int, int]]],
    *,
    max_candidates: int,
) -> list[tuple[int, int, int, int]]:
    chosen: list[tuple[int, int, int, int]] = []
    for _score, box in sorted(candidates, key=lambda item: item[0], reverse=True):
        if any(_box_iou(box, existing) >= 0.45 for existing in chosen):
            continue
        chosen.append(box)
        if len(chosen) >= max_candidates:
            break
    return chosen


def _propose_led_candidate_boxes(
    img: Image.Image,
    *,
    max_candidates: int,
) -> list[tuple[int, int, int, int]]:
    bgr = _pil_to_bgr(img)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    _, _, v = cv2.split(hsv)
    v = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6)).apply(v)
    green_dominance = bgr[:, :, 1].astype(np.int16) - np.maximum(
        bgr[:, :, 0],
        bgr[:, :, 2],
    ).astype(np.int16)
    green_dominance = np.clip(green_dominance, 0, 255).astype(np.uint8)
    green_mask = cv2.threshold(green_dominance, 10, 255, cv2.THRESH_BINARY)[1]
    green_mask = cv2.morphologyEx(
        green_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3)),
        iterations=2,
    )
    green_mask = cv2.dilate(green_mask, np.ones((3, 5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = green_mask.shape[:2]
    candidates: list[tuple[float, tuple[int, int, int, int]]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < width * 0.05 or h < height * 0.010:
            continue
        if w > width * 0.96 or h > height * 0.30:
            continue
        aspect = w / max(h, 1)
        if aspect < 1.6:
            continue
        box = _expand_box((x, y, x + w, y + h), (width, height), x_pad_frac=0.22, y_pad_frac=0.85)
        bx1, by1, bx2, by2 = box
        roi = bgr[by1:by2, bx1:bx2]
        if roi.size == 0:
            continue
        green_score = float(np.mean(green_dominance[by1:by2, bx1:bx2])) / 255.0
        darkness = 1.0 - (float(np.mean(v[by1:by2, bx1:bx2])) / 255.0)
        cy_frac = ((by1 + by2) / 2.0) / max(height, 1)
        score = (green_score * 0.58) + (darkness * 0.22) + (min(aspect, 8.0) / 8.0 * 0.10) + (cy_frac * 0.10)
        candidates.append((score, box))

    if not candidates:
        blurred_v = cv2.GaussianBlur(v, (0, 0), 1.2)
        dark_mask = cv2.threshold(blurred_v, 118, 255, cv2.THRESH_BINARY_INV)[1]
        dark_mask[: int(height * 0.22), :] = 0
        dark_mask = cv2.morphologyEx(
            dark_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (19, 7)),
            iterations=2,
        )
        contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < width * 0.18 or h < height * 0.035:
                continue
            if w > width * 0.96 or h > height * 0.26:
                continue
            aspect = w / max(h, 1)
            if aspect < 1.8 or aspect > 10.0:
                continue
            box = _expand_box((x, y, x + w, y + h), (width, height), x_pad_frac=0.08, y_pad_frac=0.35)
            bx1, by1, bx2, by2 = box
            roi_green = green_dominance[by1:by2, bx1:bx2]
            roi_v = blurred_v[by1:by2, bx1:bx2]
            if roi_green.size == 0 or roi_v.size == 0:
                continue
            green_score = float(np.mean(roi_green)) / 255.0
            darkness = 1.0 - (float(np.mean(roi_v)) / 255.0)
            cy_frac = ((by1 + by2) / 2.0) / max(height, 1)
            score = (darkness * 0.44) + (green_score * 0.26) + (min(aspect, 8.0) / 8.0 * 0.16) + (cy_frac * 0.14)
            candidates.append((score, box))
    return _dedupe_boxes(candidates, max_candidates=max_candidates)


def _propose_lcd_candidate_boxes(
    img: Image.Image,
    *,
    max_candidates: int,
) -> list[tuple[int, int, int, int]]:
    bgr = _pil_to_bgr(img)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6)).apply(gray)
    blackhat = cv2.morphologyEx(
        gray,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
    )
    binary = cv2.adaptiveThreshold(
        blackhat,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        -3,
    )
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (11, 5)),
        iterations=2,
    )

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = binary.shape[:2]
    candidates: list[tuple[float, tuple[int, int, int, int]]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < width * 0.07 or h < height * 0.02:
            continue
        if w > width * 0.75 or h > height * 0.26:
            continue
        aspect = w / max(h, 1)
        if aspect < 1.5 or aspect > 8.5:
            continue
        box = _expand_box((x, y, x + w, y + h), (width, height), x_pad_frac=0.18, y_pad_frac=0.55)
        bx1, by1, bx2, by2 = box
        roi = gray[by1:by2, bx1:bx2]
        if roi.size == 0:
            continue
        contrast = float(np.std(roi)) / 64.0
        brightness = float(np.mean(roi)) / 255.0
        cy_frac = ((by1 + by2) / 2.0) / max(height, 1)
        score = (min(contrast, 1.0) * 0.48) + ((1.0 - abs(brightness - 0.62)) * 0.22) + (min(aspect, 6.0) / 6.0 * 0.14) + ((1.0 - abs(cy_frac - 0.38)) * 0.16)
        candidates.append((score, box))
    return _dedupe_boxes(candidates, max_candidates=max_candidates)


def propose_display_candidate_boxes(
    img: Image.Image,
    *,
    display_kind: str,
    max_candidates: int = 2,
) -> list[tuple[int, int, int, int]]:
    if display_kind == "led":
        return _propose_led_candidate_boxes(img, max_candidates=max_candidates)
    if display_kind == "lcd":
        return _propose_lcd_candidate_boxes(img, max_candidates=max_candidates)

    combined: list[tuple[int, int, int, int]] = []
    for candidate in _propose_led_candidate_boxes(img, max_candidates=max_candidates):
        combined.append(candidate)
    for candidate in _propose_lcd_candidate_boxes(img, max_candidates=max_candidates):
        if any(_box_iou(candidate, existing) >= 0.45 for existing in combined):
            continue
        combined.append(candidate)
        if len(combined) >= max_candidates:
            break
    return combined[:max_candidates]
