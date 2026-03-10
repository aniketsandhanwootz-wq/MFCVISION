from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from PIL import Image


@dataclass
class CropDiagnostics:
    is_reliable: bool
    mode: str
    quality_score: float
    lit_ratio: float
    green_ratio: float
    component_count: int
    active_span_ratio: float
    active_band_height_ratio: float
    leading_blank_ratio: float
    reason: str


@dataclass
class LocalDecodeResult:
    ok: bool
    value_text: Optional[str]
    confidence: float
    digit_count: int
    decimal_count: int
    reason: str


SEGMENT_PATTERNS: dict[str, tuple[int, ...]] = {
    "0": (1, 1, 1, 0, 1, 1, 1),
    "1": (0, 0, 1, 0, 0, 1, 0),
    "2": (1, 0, 1, 1, 1, 0, 1),
    "3": (1, 0, 1, 1, 0, 1, 1),
    "4": (0, 1, 1, 1, 0, 1, 0),
    "5": (1, 1, 0, 1, 0, 1, 1),
    "6": (1, 1, 0, 1, 1, 1, 1),
    "7": (1, 0, 1, 0, 0, 1, 0),
    "8": (1, 1, 1, 1, 1, 1, 1),
    "9": (1, 1, 1, 1, 0, 1, 1),
}


def pil_to_bgr(img: Image.Image) -> np.ndarray:
    rgb = np.array(img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)



def resize_keep_aspect(img: Image.Image, max_dim: int = 1600) -> Image.Image:
    w, h = img.size
    longest = max(w, h)
    if longest <= max_dim:
        return img
    scale = max_dim / float(longest)
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    return img.resize((nw, nh), Image.LANCZOS)



def _build_lit_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    l = lab[:, :, 0]
    blue, green, red = cv2.split(bgr)

    clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    local_contrast = cv2.absdiff(l2, cv2.GaussianBlur(l2, (11, 11), 0))
    max_rb = cv2.max(red, blue)
    green_excess = cv2.subtract(green, max_rb)
    green_dominant = cv2.inRange(green_excess, 10, 255) & cv2.inRange(green, 70, 255)
    faint_green = cv2.inRange(green_excess, 4, 255) & cv2.inRange(green, 58, 255)

    green_led = (
        cv2.inRange(h, 35, 100)
        & cv2.inRange(s, 22, 255)
        & cv2.inRange(v, 80, 255)
        & cv2.inRange(local_contrast, 10, 255)
        & green_dominant
    )
    vivid_led = (
        cv2.inRange(s, 45, 255)
        & cv2.inRange(v, 110, 255)
        & cv2.inRange(local_contrast, 18, 255)
        & green_dominant
    )

    adap = cv2.adaptiveThreshold(
        l2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, -8
    )
    structural_led = (
        adap
        & cv2.inRange(local_contrast, 24, 255)
        & cv2.inRange(v, 72, 255)
        & cv2.inRange(h, 25, 110)
        & faint_green
    )

    mask = green_led | vivid_led | structural_led
    k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k1, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k1, iterations=2)
    mask = cv2.dilate(mask, k2, iterations=1)
    return mask



def _build_dark_digit_mask(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    g2 = clahe.apply(gray)

    kh = max(5, gray.shape[0] // 90)
    kw = max(15, gray.shape[1] // 35)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw | 1, kh | 1))
    blackhat = cv2.morphologyEx(g2, cv2.MORPH_BLACKHAT, kernel)

    grad_x = cv2.Sobel(blackhat, cv2.CV_32F, 1, 0, ksize=3)
    grad_x = np.absolute(grad_x)
    if grad_x.max() > 0:
        grad_x = (grad_x / grad_x.max()) * 255.0
    grad_x = grad_x.astype("uint8")

    blur = cv2.GaussianBlur(grad_x, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(9, gray.shape[1] // 40) | 1, max(3, gray.shape[0] // 150) | 1),
    )
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
    return mask



def _compute_green_ratio(bgr: np.ndarray) -> float:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    green_mask = (
        cv2.inRange(hsv[:, :, 0], 35, 100)
        & cv2.inRange(hsv[:, :, 1], 18, 255)
        & cv2.inRange(hsv[:, :, 2], 70, 255)
    )
    return float(np.count_nonzero(green_mask)) / float(max(green_mask.size, 1))



def _extract_components(mask: np.ndarray) -> list[dict[str, float]]:
    height, width = mask.shape[:2]
    min_area = max(10, int((height * width) * 0.000015))

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    components: list[dict[str, float]] = []
    for idx in range(1, num_labels):
        x, y, w, h, area = stats[idx]
        if area < min_area or w < 2 or h < 4:
            continue
        components.append(
            {
                "x": float(x),
                "y": float(y),
                "w": float(w),
                "h": float(h),
                "area": float(area),
                "cx": float(x + (w / 2.0)),
                "cy": float(y + (h / 2.0)),
            }
        )
    return components



def _horizontal_gap(a: dict[str, float], b: dict[str, float]) -> float:
    left, right = (a, b) if a["cx"] <= b["cx"] else (b, a)
    return max(0.0, right["x"] - (left["x"] + left["w"]))



def _group_components_into_rows(components: list[dict[str, float]]) -> list[list[dict[str, float]]]:
    if not components:
        return []

    components = sorted(components, key=lambda comp: comp["cx"])
    parent = list(range(len(components)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri = find(i)
        rj = find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            a = components[i]
            b = components[j]
            max_h = max(a["h"], b["h"])
            min_h = min(a["h"], b["h"])
            if min_h <= 0:
                continue
            if abs(a["cy"] - b["cy"]) > max_h * 0.55:
                continue
            if max_h / min_h > 2.6:
                continue
            if _horizontal_gap(a, b) > max_h * 1.7:
                continue
            union(i, j)

    groups: dict[int, list[dict[str, float]]] = {}
    for idx, component in enumerate(components):
        groups.setdefault(find(idx), []).append(component)
    return list(groups.values())



def _best_projection_run(values: np.ndarray, threshold_ratio: float, min_len: int) -> tuple[int, int] | None:
    if values.size == 0 or values.max() <= 0:
        return None

    threshold = max(1.0, values.max() * threshold_ratio)
    runs: list[tuple[float, int, int]] = []
    start: Optional[int] = None

    for idx, value in enumerate(values):
        if value >= threshold:
            if start is None:
                start = idx
        elif start is not None:
            if idx - start >= min_len:
                runs.append((float(values[start:idx].sum()), start, idx))
            start = None

    if start is not None and values.size - start >= min_len:
        runs.append((float(values[start:].sum()), start, values.size))

    if not runs:
        return None

    runs.sort(key=lambda item: item[0], reverse=True)
    _, run_start, run_end = runs[0]
    return run_start, run_end



def _full_projection_span(values: np.ndarray, threshold_ratio: float, max_gap: int) -> tuple[int, int] | None:
    if values.size == 0 or values.max() <= 0:
        return None

    threshold = max(1.0, values.max() * threshold_ratio)
    active = (values >= threshold).astype(np.uint8)
    if max_gap > 0:
        kernel = np.ones((max_gap,), dtype=np.uint8)
        active = cv2.morphologyEx(active[None, :], cv2.MORPH_CLOSE, kernel[None, :])[0]

    xs = np.where(active > 0)[0]
    if xs.size == 0:
        return None
    return int(xs[0]), int(xs[-1] + 1)



def _projection_runs(
    values: np.ndarray,
    threshold_ratio: float,
    max_gap: int,
    min_len: int,
) -> list[tuple[int, int]]:
    if values.size == 0 or values.max() <= 0:
        return []

    threshold = max(1.0, values.max() * threshold_ratio)
    active = (values >= threshold).astype(np.uint8)
    if max_gap > 0:
        kernel = np.ones((max_gap,), dtype=np.uint8)
        active = cv2.morphologyEx(active[None, :], cv2.MORPH_CLOSE, kernel[None, :])[0]

    xs = np.where(active > 0)[0]
    if xs.size == 0:
        return []

    runs: list[tuple[int, int]] = []
    start = int(xs[0])
    prev = int(xs[0])
    for x in xs[1:]:
        xi = int(x)
        if xi != prev + 1:
            if prev + 1 - start >= min_len:
                runs.append((start, prev + 1))
            start = xi
        prev = xi
    if prev + 1 - start >= min_len:
        runs.append((start, prev + 1))
    return runs



def _range_score(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    if low <= value <= high:
        return 1.0
    if value < low:
        return max(0.0, value / max(low, 1e-6))
    overshoot = (value - high) / max(high, 1e-6)
    return max(0.0, 1.0 - overshoot)



def _analyze_mask_structure(mask: np.ndarray, gray: np.ndarray, mode: str) -> dict[str, float | str | bool]:
    components = _extract_components(mask)
    groups = _group_components_into_rows(components)
    height, width = mask.shape[:2]

    best_group: list[dict[str, float]] | None = None
    best_group_score = -1e18
    for group in groups:
        x1 = int(min(comp["x"] for comp in group))
        y1 = int(min(comp["y"] for comp in group))
        x2 = int(max(comp["x"] + comp["w"] for comp in group))
        y2 = int(max(comp["y"] + comp["h"] for comp in group))
        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0:
            continue
        heights = np.array([comp["h"] for comp in group], dtype=np.float32)
        centers_y = np.array([comp["cy"] for comp in group], dtype=np.float32)
        height_cv = float(np.std(heights) / max(np.mean(heights), 1.0))
        row_cv = float(np.std(centers_y) / max(np.mean(heights), 1.0))
        span_ratio = float(w) / float(max(width, 1))
        band_ratio = float(h) / float(max(height, 1))
        score = 0.0
        score += min(len(group) / 4.0, 1.0) * 0.38
        score += _range_score(span_ratio, 0.18 if mode == "lcd" else 0.28, 0.95) * 0.24
        score += _range_score(band_ratio, 0.06 if mode == "lcd" else 0.12, 0.70) * 0.16
        score += max(0.0, 1.0 - height_cv) * 0.11
        score += max(0.0, 1.0 - row_cv) * 0.11
        if score > best_group_score:
            best_group_score = score
            best_group = group

    col_energy = np.count_nonzero(mask, axis=0).astype(np.float32)
    span = _full_projection_span(
        col_energy,
        threshold_ratio=0.10,
        max_gap=max(6, width // 28),
    )
    row_energy = np.count_nonzero(mask, axis=1).astype(np.float32)
    band = _best_projection_run(
        row_energy,
        threshold_ratio=0.24,
        min_len=max(4, height // 10),
    )

    lit_ratio = float(np.count_nonzero(mask)) / float(max(mask.size, 1))
    if span is None:
        active_span_ratio = 0.0
        leading_blank_ratio = 1.0
    else:
        sx1, sx2 = span
        active_span_ratio = float(sx2 - sx1) / float(max(width, 1))
        leading_blank_ratio = float(sx1) / float(max(width, 1))

    if band is None:
        active_band_height_ratio = 0.0
    else:
        by1, by2 = band
        active_band_height_ratio = float(by2 - by1) / float(max(height, 1))

    best_group_size = len(best_group) if best_group is not None else 0
    quality_score = 0.0
    quality_score += min(best_group_size / (3.0 if mode == "lcd" else 4.0), 1.0) * 0.34
    quality_score += _range_score(active_span_ratio, 0.18 if mode == "lcd" else 0.28, 0.95) * 0.20
    quality_score += _range_score(active_band_height_ratio, 0.07 if mode == "lcd" else 0.12, 0.58 if mode == "lcd" else 0.60) * 0.14
    quality_score += _range_score(lit_ratio, 0.004 if mode == "lcd" else 0.012, 0.34) * 0.10
    quality_score += _range_score(1.0 - leading_blank_ratio, 0.20, 1.00) * 0.06
    quality_score += max(0.0, best_group_score) * 0.16

    failures: list[str] = []
    min_group = 2 if mode == "lcd" else 3
    if best_group_size < min_group:
        failures.append("no coherent digit row")
    if active_span_ratio < (0.18 if mode == "lcd" else 0.28):
        failures.append("active row too narrow")
    if active_band_height_ratio < (0.07 if mode == "lcd" else 0.12):
        failures.append("active band too thin")
    if active_band_height_ratio > (0.58 if mode == "lcd" else 0.60):
        failures.append("active band too tall for a display row")
    if lit_ratio < (0.004 if mode == "lcd" else 0.012):
        failures.append("too little display structure")
    if leading_blank_ratio > 0.80:
        failures.append("signal starts too far to the right")

    return {
        "mode": mode,
        "quality_score": round(quality_score, 3),
        "lit_ratio": round(lit_ratio, 4),
        "component_count": len(components),
        "best_group_size": best_group_size,
        "active_span_ratio": round(active_span_ratio, 3),
        "active_band_height_ratio": round(active_band_height_ratio, 3),
        "leading_blank_ratio": round(leading_blank_ratio, 3),
        "is_reliable": (
            quality_score >= (0.54 if mode == "lcd" else 0.58)
            and best_group_size >= min_group
            and active_span_ratio >= (0.18 if mode == "lcd" else 0.28)
            and active_band_height_ratio >= (0.07 if mode == "lcd" else 0.12)
            and active_band_height_ratio <= (0.58 if mode == "lcd" else 0.60)
        ),
        "reason": "localized crop contains a coherent active display row"
        if quality_score >= (0.54 if mode == "lcd" else 0.58) and best_group_size >= min_group
        else (", ".join(failures) if failures else "localized crop is weak or partial"),
    }



def analyze_crop_diagnostics(crop_img: Image.Image) -> CropDiagnostics:
    crop_img = resize_keep_aspect(crop_img, max_dim=900)
    bgr = pil_to_bgr(crop_img)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lit_mask = _build_lit_mask(bgr)
    dark_mask = _build_dark_digit_mask(gray)
    green_ratio = _compute_green_ratio(bgr)
    blue_ratio = float(
        np.count_nonzero(
            cv2.inRange(hsv[:, :, 0], 85, 135)
            & cv2.inRange(hsv[:, :, 1], 35, 255)
            & cv2.inRange(hsv[:, :, 2], 30, 255)
        )
    ) / float(max(gray.size, 1))
    dark_ratio = float(np.count_nonzero(cv2.inRange(gray, 0, 105))) / float(max(gray.size, 1))

    led_metrics = _analyze_mask_structure(lit_mask, gray, "led")
    lcd_metrics = _analyze_mask_structure(dark_mask, gray, "lcd")
    selected = led_metrics if float(led_metrics["quality_score"]) >= float(lcd_metrics["quality_score"]) else lcd_metrics

    is_reliable = bool(selected["is_reliable"])
    quality_score = float(selected["quality_score"])
    reason = str(selected["reason"])
    mode = str(selected["mode"])

    if mode == "led" and green_ratio < 0.008:
        is_reliable = False
        quality_score = min(quality_score, 0.42)
        reason = "crop lacks enough green LED signal and is likely not a readable LED display row"
    if mode == "led" and blue_ratio > 0.20 and dark_ratio < 0.14:
        is_reliable = False
        quality_score = min(quality_score, 0.38)
        reason = "crop looks like green reflections on a blue machine panel, not a dark LED display window"

    return CropDiagnostics(
        is_reliable=is_reliable,
        mode=mode,
        quality_score=quality_score,
        lit_ratio=float(selected["lit_ratio"]),
        green_ratio=round(green_ratio, 4),
        component_count=int(selected["component_count"]),
        active_span_ratio=float(selected["active_span_ratio"]),
        active_band_height_ratio=float(selected["active_band_height_ratio"]),
        leading_blank_ratio=float(selected["leading_blank_ratio"]),
        reason=reason,
    )



def _render_digit_template(digit: str, width: int = 72, height: int = 120) -> np.ndarray:
    template = np.zeros((height, width), dtype=np.uint8)
    segments = {
        0: ((16, 8), (56, 20)),
        1: ((8, 18), (20, 54)),
        2: ((52, 18), (64, 54)),
        3: ((16, 52), (56, 64)),
        4: ((8, 64), (20, 100)),
        5: ((52, 64), (64, 100)),
        6: ((16, 100), (56, 112)),
    }
    pattern = SEGMENT_PATTERNS[digit]
    for idx, is_on in enumerate(pattern):
        if not is_on:
            continue
        (x1, y1), (x2, y2) = segments[idx]
        cv2.rectangle(template, (x1, y1), (x2, y2), 255, -1)
    return template


DIGIT_TEMPLATES: dict[str, np.ndarray] = {
    digit: _render_digit_template(digit) for digit in SEGMENT_PATTERNS
}



def _extract_active_boxes(mask: np.ndarray) -> list[dict[str, float]]:
    height, width = mask.shape[:2]
    col_energy = np.count_nonzero(mask, axis=0).astype(np.float32)
    runs = _projection_runs(
        col_energy,
        threshold_ratio=0.16,
        max_gap=max(4, width // 90),
        min_len=max(2, width // 220),
    )

    boxes: list[dict[str, float]] = []
    for x1, x2 in runs:
        roi = mask[:, x1:x2]
        rows = np.where(np.count_nonzero(roi, axis=1) > 0)[0]
        if rows.size == 0:
            continue
        y1 = int(rows[0])
        y2 = int(rows[-1] + 1)
        w = x2 - x1
        h = y2 - y1
        if w <= 1 or h <= 1:
            continue
        boxes.append(
            {
                "x1": float(x1),
                "x2": float(x2),
                "y1": float(y1),
                "y2": float(y2),
                "w": float(w),
                "h": float(h),
                "cx": float((x1 + x2) / 2.0),
                "cy": float((y1 + y2) / 2.0),
                "area": float(np.count_nonzero(mask[y1:y2, x1:x2])),
            }
        )
    return boxes



def _classify_decimal_candidates(boxes: list[dict[str, float]]) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    if not boxes:
        return [], []

    heights = np.array([box["h"] for box in boxes], dtype=np.float32)
    widths = np.array([box["w"] for box in boxes], dtype=np.float32)
    centers_y = np.array([box["cy"] for box in boxes], dtype=np.float32)

    median_h = float(np.median(heights))
    median_w = float(np.median(widths))
    baseline_y = float(np.median(centers_y))

    digit_boxes: list[dict[str, float]] = []
    decimal_boxes: list[dict[str, float]] = []
    for box in boxes:
        is_decimal = (
            box["w"] <= max(4.0, median_w * 0.42)
            and box["h"] <= max(5.0, median_h * 0.55)
            and box["cy"] >= baseline_y - (median_h * 0.10)
        )
        if is_decimal:
            decimal_boxes.append(box)
        else:
            digit_boxes.append(box)

    return digit_boxes, decimal_boxes



def _normalize_digit_boxes(digit_boxes: list[dict[str, float]], frame_width: int) -> list[dict[str, float]]:
    if not digit_boxes:
        return []

    ordered = sorted(digit_boxes, key=lambda box: box["x1"])
    widths = np.array([box["w"] for box in ordered], dtype=np.float32)
    median_w = float(np.median(widths))
    wide_widths = widths[widths >= median_w * 0.70]
    target_w = float(np.median(wide_widths)) if wide_widths.size else median_w

    normalized: list[dict[str, float]] = []
    for idx, box in enumerate(ordered):
        cx = float(box["cx"])
        expand_w = max(float(box["w"]), target_w * 0.88)
        proposed_x1 = int(round(cx - (expand_w / 2.0)))
        proposed_x2 = int(round(cx + (expand_w / 2.0)))

        left_limit = 0
        right_limit = frame_width
        if idx > 0:
            prev = ordered[idx - 1]
            left_limit = int((prev["x2"] + box["x1"]) / 2.0)
        if idx + 1 < len(ordered):
            nxt = ordered[idx + 1]
            right_limit = int((box["x2"] + nxt["x1"]) / 2.0)

        x1 = max(0, max(proposed_x1, left_limit))
        x2 = min(frame_width, min(proposed_x2, right_limit))
        if x2 <= x1:
            x1 = int(box["x1"])
            x2 = int(box["x2"])

        normalized.append(
            {
                **box,
                "x1": float(x1),
                "x2": float(x2),
                "w": float(x2 - x1),
                "cx": float((x1 + x2) / 2.0),
            }
        )

    return normalized



def _normalize_digit_roi(mask: np.ndarray) -> np.ndarray:
    roi = cv2.copyMakeBorder(mask, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
    roi = cv2.resize(roi, (72, 120), interpolation=cv2.INTER_NEAREST)
    roi = cv2.morphologyEx(
        roi,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )
    return roi



def _decode_digit_box(mask: np.ndarray, box: dict[str, float]) -> tuple[Optional[str], float]:
    x1 = int(box["x1"])
    x2 = int(box["x2"])
    y1 = int(box["y1"])
    y2 = int(box["y2"])
    roi = mask[y1:y2, x1:x2]
    if roi.size == 0:
        return None, 0.0

    roi = _normalize_digit_roi(roi)
    roi_bin = (roi > 0).astype(np.uint8)
    best_digit: Optional[str] = None
    best_score = -1e9
    second_score = -1e9

    roi_count = float(np.count_nonzero(roi_bin))
    if roi_count <= 0.0:
        return None, 0.0

    for digit, template in DIGIT_TEMPLATES.items():
        templ_bin = (template > 0).astype(np.uint8)
        templ_count = float(np.count_nonzero(templ_bin))
        inter = float(np.count_nonzero((roi_bin > 0) & (templ_bin > 0)))
        union = float(np.count_nonzero((roi_bin > 0) | (templ_bin > 0)))
        dice = (2.0 * inter) / max(roi_count + templ_count, 1.0)
        iou = inter / max(union, 1.0)
        score = (dice * 0.7) + (iou * 0.3)
        if score > best_score:
            second_score = best_score
            best_score = score
            best_digit = digit
        elif score > second_score:
            second_score = score

    if best_digit is None:
        return None, 0.0

    confidence = max(0.0, min(1.0, 0.15 + (best_score * 1.15) + ((best_score - second_score) * 0.55)))
    return best_digit, confidence



def decode_display_crop(crop_img: Image.Image) -> LocalDecodeResult:
    crop_img = resize_keep_aspect(crop_img, max_dim=900)
    bgr = pil_to_bgr(crop_img)
    mask = _build_lit_mask(bgr)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )

    boxes = _extract_active_boxes(mask)
    digit_boxes, decimal_boxes = _classify_decimal_candidates(boxes)
    digit_boxes = sorted(digit_boxes, key=lambda box: box["x1"])
    decimal_boxes = sorted(decimal_boxes, key=lambda box: box["x1"])
    digit_boxes = _normalize_digit_boxes(digit_boxes, mask.shape[1])

    if not digit_boxes:
        return LocalDecodeResult(
            ok=False,
            value_text=None,
            confidence=0.0,
            digit_count=0,
            decimal_count=len(decimal_boxes),
            reason="no active digit boxes found in localized crop",
        )

    digits: list[str] = []
    digit_confidences: list[float] = []
    for box in digit_boxes:
        digit, conf = _decode_digit_box(mask, box)
        if digit is None:
            return LocalDecodeResult(
                ok=False,
                value_text=None,
                confidence=0.0,
                digit_count=len(digit_boxes),
                decimal_count=len(decimal_boxes),
                reason="deterministic seven-segment decode failed for one or more slots",
            )
        digits.append(digit)
        digit_confidences.append(conf)

    decimal_index: Optional[int] = None
    if decimal_boxes:
        dot = decimal_boxes[0]
        index = 0
        for idx, box in enumerate(digit_boxes):
            if dot["cx"] > box["x2"]:
                index = idx + 1
        if 0 < index < len(digit_boxes):
            decimal_index = index

    text = "".join(digits)
    if decimal_index is not None:
        text = f"{text[:decimal_index]}.{text[decimal_index:]}"

    confidence = float(np.mean(digit_confidences)) if digit_confidences else 0.0
    if decimal_index is not None and len(decimal_boxes) == 1:
        confidence = min(1.0, confidence + 0.06)

    return LocalDecodeResult(
        ok=True,
        value_text=text,
        confidence=round(confidence, 3),
        digit_count=len(digit_boxes),
        decimal_count=len(decimal_boxes),
        reason="deterministic seven-segment decode from localized crop",
    )
