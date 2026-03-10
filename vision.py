from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import Optional


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
class DisplayCandidate:
    mode: str
    score: float
    group_size: int
    raw_x1: int
    raw_y1: int
    raw_x2: int
    raw_y2: int
    crop_x1: int
    crop_y1: int
    crop_x2: int
    crop_y2: int


def _crop_roi(
    arr: np.ndarray,
    candidate: DisplayCandidate,
) -> np.ndarray:
    return arr[candidate.crop_y1:candidate.crop_y2, candidate.crop_x1:candidate.crop_x2]


@dataclass
class LocalDecodeResult:
    ok: bool
    value_text: Optional[str]
    confidence: float
    digit_count: int
    decimal_count: int
    reason: str


def pil_to_bgr(img: Image.Image) -> np.ndarray:
    rgb = np.array(img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_pil(arr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def resize_keep_aspect(img: Image.Image, max_dim: int = 1600) -> Image.Image:
    w, h = img.size
    longest = max(w, h)
    if longest <= max_dim:
        return img
    scale = max_dim / float(longest)
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    return img.resize((nw, nh), Image.LANCZOS)


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = order_points(pts.astype("float32"))
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB), 1)

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB), 1)

    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32",
    )

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def enhance_display_crop(crop_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    l_blend = cv2.addWeighted(l, 0.76, l2, 0.24, 0)
    lab2 = cv2.merge([l_blend, a, b])
    out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    out = cv2.bilateralFilter(out, d=5, sigmaColor=16, sigmaSpace=16)
    out = cv2.resize(out, None, fx=1.25, fy=1.25, interpolation=cv2.INTER_CUBIC)
    return out


def _build_lit_mask(bgr: np.ndarray) -> np.ndarray:
    """
    Emphasize active LED segments while suppressing the machine body and buttons.
    """
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

    # Active display segments are usually colored, bright, and locally contrasted.
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


def _lcd_window_stats(
    group: list[dict[str, float]],
    gray: np.ndarray,
    dark_mask: np.ndarray,
    bgr: np.ndarray,
) -> dict[str, float]:
    height, width = gray.shape[:2]
    x1 = int(min(comp["x"] for comp in group))
    y1 = int(min(comp["y"] for comp in group))
    x2 = int(max(comp["x"] + comp["w"] for comp in group))
    y2 = int(max(comp["y"] + comp["h"] for comp in group))
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)

    pad_x = max(8, int(w * 0.30))
    pad_top = max(6, int(h * 0.75))
    pad_bottom = max(6, int(h * 0.65))
    wx1 = max(0, x1 - pad_x)
    wy1 = max(0, y1 - pad_top)
    wx2 = min(width, x2 + pad_x)
    wy2 = min(height, y2 + pad_bottom)

    region_bgr = bgr[wy1:wy2, wx1:wx2]
    region_gray = gray[wy1:wy2, wx1:wx2]
    region_mask = dark_mask[wy1:wy2, wx1:wx2]
    if region_bgr.size == 0 or region_gray.size == 0 or region_mask.size == 0:
        return {
            "sat_mean": 255.0,
            "low_sat_ratio": 0.0,
            "blue_ratio": 1.0,
            "dark_ratio": 0.0,
            "gray_mean": 255.0,
            "gray_std": 0.0,
            "window_x1": float(wx1),
            "window_y1": float(wy1),
            "window_x2": float(wx2),
            "window_y2": float(wy2),
        }

    hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    hue = hsv[:, :, 0]
    val = hsv[:, :, 2]

    low_sat_ratio = float(np.count_nonzero(sat <= 72)) / float(max(sat.size, 1))
    blue_ratio = float(
        np.count_nonzero(
            cv2.inRange(hue, 85, 135)
            & cv2.inRange(sat, 40, 255)
            & cv2.inRange(val, 35, 255)
        )
    ) / float(max(sat.size, 1))
    dark_ratio = float(np.count_nonzero(region_mask)) / float(max(region_mask.size, 1))

    return {
        "sat_mean": float(sat.mean()),
        "low_sat_ratio": low_sat_ratio,
        "blue_ratio": blue_ratio,
        "dark_ratio": dark_ratio,
        "gray_mean": float(region_gray.mean()),
        "gray_std": float(region_gray.std()),
        "window_x1": float(wx1),
        "window_y1": float(wy1),
        "window_x2": float(wx2),
        "window_y2": float(wy2),
    }


def _led_window_stats(
    group: list[dict[str, float]],
    gray: np.ndarray,
    lit_mask: np.ndarray,
    bgr: np.ndarray,
) -> dict[str, float]:
    height, width = gray.shape[:2]
    x1 = int(min(comp["x"] for comp in group))
    y1 = int(min(comp["y"] for comp in group))
    x2 = int(max(comp["x"] + comp["w"] for comp in group))
    y2 = int(max(comp["y"] + comp["h"] for comp in group))
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)

    pad_x = max(10, int(w * 0.22))
    pad_top = max(8, int(h * 1.10))
    pad_bottom = max(6, int(h * 0.80))
    wx1 = max(0, x1 - pad_x)
    wy1 = max(0, y1 - pad_top)
    wx2 = min(width, x2 + pad_x)
    wy2 = min(height, y2 + pad_bottom)

    region_bgr = bgr[wy1:wy2, wx1:wx2]
    region_gray = gray[wy1:wy2, wx1:wx2]
    region_lit = lit_mask[wy1:wy2, wx1:wx2]
    if region_bgr.size == 0 or region_gray.size == 0 or region_lit.size == 0:
        return {
            "dark_ratio": 0.0,
            "blue_ratio": 1.0,
            "green_on_dark_ratio": 0.0,
            "lit_fill_ratio": 0.0,
        }

    hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
    green_mask = (
        cv2.inRange(hsv[:, :, 0], 35, 100)
        & cv2.inRange(hsv[:, :, 1], 18, 255)
        & cv2.inRange(hsv[:, :, 2], 70, 255)
    )
    blue_mask = (
        cv2.inRange(hsv[:, :, 0], 85, 135)
        & cv2.inRange(hsv[:, :, 1], 35, 255)
        & cv2.inRange(hsv[:, :, 2], 30, 255)
    )
    dark_mask = cv2.inRange(region_gray, 0, 105)
    dark_support = cv2.dilate(dark_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)

    total = float(max(region_gray.size, 1))
    return {
        "dark_ratio": float(np.count_nonzero(dark_mask)) / total,
        "blue_ratio": float(np.count_nonzero(blue_mask)) / total,
        "green_on_dark_ratio": float(np.count_nonzero(green_mask & dark_support)) / total,
        "lit_fill_ratio": float(np.count_nonzero(region_lit)) / total,
    }


def _extract_components(mask: np.ndarray) -> list[dict[str, float]]:
    H, W = mask.shape[:2]
    min_area = max(10, int((H * W) * 0.000015))

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    comps: list[dict[str, float]] = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        if w < 2 or h < 4:
            continue
        comps.append(
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
    return comps


def _horizontal_gap(a: dict[str, float], b: dict[str, float]) -> float:
    if a["cx"] <= b["cx"]:
        left = a
        right = b
    else:
        left = b
        right = a
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
    for idx, comp in enumerate(components):
        groups.setdefault(find(idx), []).append(comp)
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


def _full_projection_span(
    values: np.ndarray,
    threshold_ratio: float,
    max_gap: int,
) -> tuple[int, int] | None:
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


def _build_window_edge_mask(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120)
    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=2,
    )
    edges = cv2.dilate(
        edges,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )
    return edges


def _candidate_from_rect(
    x: int,
    y: int,
    w: int,
    h: int,
    shape: tuple[int, int],
    mode: str,
) -> DisplayCandidate:
    height, width = shape[:2]
    if mode == "lcd":
        pad_left = max(6, int(w * 0.10))
        pad_right = max(6, int(w * 0.10))
        pad_top = max(6, int(h * 0.18))
        pad_bottom = max(6, int(h * 0.18))
    else:
        pad_left = max(10, int(w * 0.12))
        pad_right = max(8, int(w * 0.10))
        pad_top = max(8, int(h * 0.22))
        pad_bottom = max(6, int(h * 0.18))

    x1 = max(0, x - pad_left)
    y1 = max(0, y - pad_top)
    x2 = min(width, x + w + pad_right)
    y2 = min(height, y + h + pad_bottom)
    return DisplayCandidate(
        mode=mode,
        score=0.0,
        group_size=0,
        raw_x1=x,
        raw_y1=y,
        raw_x2=x + w,
        raw_y2=y + h,
        crop_x1=x1,
        crop_y1=y1,
        crop_x2=x2,
        crop_y2=y2,
    )


def _rect_contains(candidate: DisplayCandidate, x: int, y: int, w: int, h: int, slack: int = 6) -> bool:
    return (
        x <= candidate.raw_x1 + slack
        and y <= candidate.raw_y1 + slack
        and x + w >= candidate.raw_x2 - slack
        and y + h >= candidate.raw_y2 - slack
    )


def _score_lcd_window_rect(
    x: int,
    y: int,
    w: int,
    h: int,
    area: float,
    gray: np.ndarray,
    bgr: np.ndarray,
    dark_mask: np.ndarray,
    edge_mask: np.ndarray,
) -> float:
    height, width = gray.shape[:2]
    if w < max(36, int(width * 0.06)) or h < max(14, int(height * 0.02)):
        return -1e9
    if w > int(width * 0.60) or h > int(height * 0.25):
        return -1e9

    aspect = w / max(h, 1)
    if aspect < 1.2 or aspect > 5.2:
        return -1e9

    rect_area = float(max(w * h, 1))
    fill_ratio = float(area) / rect_area
    if fill_ratio < 0.08:
        return -1e9

    pad = max(2, min(w, h) // 12)
    ix1 = min(x + pad, x + w - 1)
    iy1 = min(y + pad, y + h - 1)
    ix2 = max(ix1 + 1, x + w - pad)
    iy2 = max(iy1 + 1, y + h - pad)

    region_gray = gray[y:y + h, x:x + w]
    region_bgr = bgr[y:y + h, x:x + w]
    region_dark = dark_mask[y:y + h, x:x + w]
    region_edge = edge_mask[y:y + h, x:x + w]
    inner_gray = gray[iy1:iy2, ix1:ix2]
    inner_dark = dark_mask[iy1:iy2, ix1:ix2]
    if region_gray.size == 0 or region_bgr.size == 0 or inner_gray.size == 0:
        return -1e9

    hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
    low_sat_ratio = float(np.count_nonzero(hsv[:, :, 1] <= 72)) / float(max(region_gray.size, 1))
    blue_ratio = float(
        np.count_nonzero(
            cv2.inRange(hsv[:, :, 0], 85, 135)
            & cv2.inRange(hsv[:, :, 1], 40, 255)
            & cv2.inRange(hsv[:, :, 2], 35, 255)
        )
    ) / float(max(region_gray.size, 1))
    dark_ratio = float(np.count_nonzero(inner_dark)) / float(max(inner_dark.size, 1))

    ring_mask = np.ones(region_gray.shape[:2], dtype=np.uint8)
    ring_mask[pad:max(pad, h - pad), pad:max(pad, w - pad)] = 0
    ring_values = region_gray[ring_mask > 0]
    border_mean = float(ring_values.mean()) if ring_values.size else float(region_gray.mean())
    inner_mean = float(inner_gray.mean())
    border_contrast = max(0.0, inner_mean - border_mean)

    edge_density = float(np.count_nonzero(region_edge)) / float(max(region_edge.size, 1))
    cx = (x + (w / 2.0)) / max(width, 1)
    cy = (y + (h / 2.0)) / max(height, 1)
    center_bias_x = 1.0 - abs(cx - 0.50) / 0.50
    center_bias_y = 1.0 - abs(cy - 0.34) / 0.34

    score = 0.0
    score += min(low_sat_ratio / 0.70, 1.0) * 26.0
    score += _range_score(dark_ratio, 0.010, 0.20) * 28.0
    score += min(border_contrast / 32.0, 1.0) * 24.0
    score += min(edge_density / 0.12, 1.0) * 18.0
    score += max(0.0, 1.0 - (blue_ratio / 0.20)) * 26.0
    score += max(0.0, center_bias_x) * 8.0
    score += max(0.0, center_bias_y) * 14.0
    score += min(fill_ratio / 0.45, 1.0) * 8.0

    if cy > 0.60:
        score -= 44.0
    if cy < 0.12:
        score -= 16.0
    if blue_ratio > 0.18:
        score -= 38.0
    if low_sat_ratio < 0.38:
        score -= 28.0
    if border_contrast < 6.0:
        score -= 22.0

    return score


def _score_led_window_rect(
    x: int,
    y: int,
    w: int,
    h: int,
    area: float,
    gray: np.ndarray,
    bgr: np.ndarray,
    lit_mask: np.ndarray,
    edge_mask: np.ndarray,
    anchor: DisplayCandidate,
) -> float:
    height, width = gray.shape[:2]
    anchor_w = anchor.raw_x2 - anchor.raw_x1
    anchor_h = anchor.raw_y2 - anchor.raw_y1
    if w < max(anchor_w, int(width * 0.08)) or h < max(anchor_h, int(height * 0.03)):
        return -1e9
    if w > int(anchor_w * 4.8) or h > int(anchor_h * 5.2):
        return -1e9
    if not _rect_contains(anchor, x, y, w, h, slack=max(6, anchor_h // 3)):
        return -1e9

    aspect = w / max(h, 1)
    if aspect < 2.0 or aspect > 8.0:
        return -1e9

    region_gray = gray[y:y + h, x:x + w]
    region_bgr = bgr[y:y + h, x:x + w]
    region_lit = lit_mask[y:y + h, x:x + w]
    region_edge = edge_mask[y:y + h, x:x + w]
    if region_gray.size == 0 or region_bgr.size == 0 or region_lit.size == 0:
        return -1e9

    hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
    green_ratio = float(
        np.count_nonzero(
            cv2.inRange(hsv[:, :, 0], 35, 100)
            & cv2.inRange(hsv[:, :, 1], 18, 255)
            & cv2.inRange(hsv[:, :, 2], 70, 255)
        )
    ) / float(max(region_gray.size, 1))
    blue_ratio = float(
        np.count_nonzero(
            cv2.inRange(hsv[:, :, 0], 85, 135)
            & cv2.inRange(hsv[:, :, 1], 35, 255)
            & cv2.inRange(hsv[:, :, 2], 30, 255)
        )
    ) / float(max(region_gray.size, 1))
    lit_ratio = float(np.count_nonzero(region_lit)) / float(max(region_lit.size, 1))
    dark_ratio = float(np.count_nonzero(cv2.inRange(region_gray, 0, 110))) / float(max(region_gray.size, 1))
    edge_density = float(np.count_nonzero(region_edge)) / float(max(region_edge.size, 1))
    cx = (x + (w / 2.0)) / max(width, 1)
    cy = (y + (h / 2.0)) / max(height, 1)
    center_bias_x = 1.0 - abs(cx - 0.50) / 0.50
    center_bias_y = 1.0 - abs(cy - 0.58) / 0.42
    left_margin_ratio = float(anchor.raw_x1 - x) / float(max(w, 1))

    score = 0.0
    score += min(green_ratio / 0.06, 1.0) * 28.0
    score += min(lit_ratio / 0.08, 1.0) * 24.0
    score += min(dark_ratio / 0.22, 1.0) * 22.0
    score += min(edge_density / 0.10, 1.0) * 14.0
    score += max(0.0, 1.0 - (blue_ratio / 0.24)) * 18.0
    score += _range_score(left_margin_ratio, 0.08, 0.38) * 20.0
    score += max(0.0, center_bias_x) * 6.0
    score += max(0.0, center_bias_y) * 8.0

    if cy > 0.78:
        score -= 30.0
    if dark_ratio < 0.10:
        score -= 24.0
    if green_ratio < 0.018:
        score -= 22.0
    if left_margin_ratio < 0.05:
        score -= 18.0

    return score


def _score_group(
    group: list[dict[str, float]],
    gray: np.ndarray,
    lit_mask: np.ndarray,
    bgr: np.ndarray,
) -> float:
    H, W = gray.shape[:2]
    if len(group) < 3:
        return -1e9

    x1 = int(min(comp["x"] for comp in group))
    y1 = int(min(comp["y"] for comp in group))
    x2 = int(max(comp["x"] + comp["w"] for comp in group))
    y2 = int(max(comp["y"] + comp["h"] for comp in group))
    w = x2 - x1
    h = y2 - y1

    if w < 50 or h < 12:
        return -1e9

    aspect = w / max(h, 1)
    if aspect < 2.0 or aspect > 14.0:
        return -1e9

    heights = np.array([comp["h"] for comp in group], dtype=np.float32)
    centers_y = np.array([comp["cy"] for comp in group], dtype=np.float32)
    height_cv = float(np.std(heights) / max(np.mean(heights), 1.0))
    row_cv = float(np.std(centers_y) / max(np.mean(heights), 1.0))

    region_gray = gray[y1:y2, x1:x2]
    region_mask = lit_mask[y1:y2, x1:x2]
    region_bgr = bgr[y1:y2, x1:x2]
    if region_gray.size == 0 or region_mask.size == 0:
        return -1e9

    mean_gray = float(region_gray.mean())
    lit_ratio = float(np.count_nonzero(region_mask)) / float(max(region_mask.size, 1))
    hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
    green_ratio = float(
        np.count_nonzero(
            cv2.inRange(hsv[:, :, 0], 35, 100)
            & cv2.inRange(hsv[:, :, 1], 18, 255)
            & cv2.inRange(hsv[:, :, 2], 70, 255)
        )
    ) / float(max(region_mask.size, 1))
    window_stats = _led_window_stats(group, gray, lit_mask, bgr)

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    center_bias_x = 1.0 - abs((cx / W) - 0.50)
    center_bias_y = 1.0 - abs((cy / H) - 0.60)

    score = 0.0
    score += min(len(group), 8) * 8.0
    score += max(0.0, 1.0 - height_cv) * 28.0
    score += max(0.0, 1.0 - row_cv) * 32.0
    score += min(lit_ratio / 0.14, 1.0) * 35.0
    score += min(green_ratio / 0.08, 1.0) * 26.0
    score += max(0.0, 180.0 - mean_gray) / 180.0 * 24.0
    score += center_bias_x * 8.0
    score += center_bias_y * 10.0
    score += min(window_stats["dark_ratio"] / 0.24, 1.0) * 24.0
    score += min(window_stats["green_on_dark_ratio"] / 0.05, 1.0) * 24.0
    score += max(0.0, 1.0 - (window_stats["blue_ratio"] / 0.20)) * 16.0

    if 4 <= len(group) <= 7:
        score += 20.0
    if w > W * 0.09:
        score += 10.0
    if y2 > H * 0.82:
        score -= 30.0
    if h > H * 0.12:
        score -= 12.0
    if aspect < 2.8:
        score -= 10.0
    if green_ratio < 0.015:
        score -= 45.0
    if green_ratio < 0.025 and cy > H * 0.55:
        score -= 24.0
    if window_stats["dark_ratio"] < 0.12:
        score -= 30.0
    if window_stats["green_on_dark_ratio"] < 0.010:
        score -= 26.0
    if window_stats["blue_ratio"] > 0.20 and window_stats["dark_ratio"] < 0.16:
        score -= 42.0

    return score


def _score_lcd_group(
    group: list[dict[str, float]],
    gray: np.ndarray,
    dark_mask: np.ndarray,
    bgr: np.ndarray,
) -> float:
    H, W = gray.shape[:2]
    if len(group) < 3:
        return -1e9

    x1 = int(min(comp["x"] for comp in group))
    y1 = int(min(comp["y"] for comp in group))
    x2 = int(max(comp["x"] + comp["w"] for comp in group))
    y2 = int(max(comp["y"] + comp["h"] for comp in group))
    w = x2 - x1
    h = y2 - y1

    if w < 32 or h < 10:
        return -1e9

    aspect = w / max(h, 1)
    if aspect < 1.4 or aspect > 7.0:
        return -1e9

    heights = np.array([comp["h"] for comp in group], dtype=np.float32)
    centers_y = np.array([comp["cy"] for comp in group], dtype=np.float32)
    height_cv = float(np.std(heights) / max(np.mean(heights), 1.0))
    row_cv = float(np.std(centers_y) / max(np.mean(heights), 1.0))

    region_gray = gray[y1:y2, x1:x2]
    region_mask = dark_mask[y1:y2, x1:x2]
    if region_gray.size == 0 or region_mask.size == 0:
        return -1e9

    mask_ratio = float(np.count_nonzero(region_mask)) / float(max(region_mask.size, 1))
    contrast = float(region_gray.std())
    window_stats = _lcd_window_stats(group, gray, dark_mask, bgr)

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    center_bias_x = 1.0 - abs((cx / W) - 0.50)
    center_bias_y = 1.0 - abs((cy / H) - 0.42)

    score = 0.0
    score += min(len(group), 8) * 8.0
    score += max(0.0, 1.0 - height_cv) * 28.0
    score += max(0.0, 1.0 - row_cv) * 28.0
    score += min(mask_ratio / 0.10, 1.0) * 20.0
    score += min(contrast / 28.0, 1.0) * 20.0
    score += center_bias_x * 8.0
    score += center_bias_y * 18.0
    score += min(window_stats["low_sat_ratio"] / 0.72, 1.0) * 28.0
    score += _range_score(window_stats["gray_mean"], 55.0, 190.0) * 8.0
    score += _range_score(window_stats["gray_std"], 8.0, 60.0) * 8.0
    score += _range_score(window_stats["dark_ratio"], 0.010, 0.18) * 14.0
    score += max(0.0, 1.0 - (window_stats["sat_mean"] / 110.0)) * 20.0
    score += max(0.0, 1.0 - (window_stats["blue_ratio"] / 0.18)) * 34.0

    if 3 <= len(group) <= 8:
        score += 14.0
    if y2 > H * 0.82:
        score -= 28.0
    if cy > H * 0.58:
        score -= 20.0
    if h > H * 0.12:
        score -= 8.0
    if window_stats["blue_ratio"] > 0.20:
        score -= 46.0
    if window_stats["sat_mean"] > 95.0 and window_stats["low_sat_ratio"] < 0.45:
        score -= 30.0
    if window_stats["dark_ratio"] < 0.008:
        score -= 22.0
    if y1 > H * 0.55:
        score -= 26.0

    return score


def _refine_bounds_from_mask(
    lit_mask: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> tuple[int, int, int, int]:
    roi_mask = lit_mask[y1:y2, x1:x2]
    if roi_mask.size == 0:
        return x1, y1, x2, y2

    row_energy = np.count_nonzero(roi_mask, axis=1).astype(np.float32)
    row_run = _best_projection_run(row_energy, threshold_ratio=0.28, min_len=max(3, roi_mask.shape[0] // 18))
    if row_run is not None:
        ry1, ry2 = row_run
        band_h = max(ry2 - ry1, 1)
        base_y1 = y1
        y1 = max(base_y1, base_y1 + ry1 - int(band_h * 0.45))
        y2 = min(y2, base_y1 + ry2 + int(band_h * 0.55))

    roi_mask = lit_mask[y1:y2, x1:x2]
    if roi_mask.size == 0:
        return x1, y1, x2, y2

    col_energy = np.count_nonzero(roi_mask, axis=0).astype(np.float32)
    col_span = _full_projection_span(
        col_energy,
        threshold_ratio=0.10,
        max_gap=max(8, roi_mask.shape[1] // 22),
    )
    if col_span is not None:
        rx1, rx2 = col_span
        band_w = max(rx2 - rx1, 1)
        base_x1 = x1
        x1 = max(base_x1, base_x1 + rx1 - int(band_w * 0.40))
        x2 = min(x2, base_x1 + rx2 + int(band_w * 0.16))

    return x1, y1, x2, y2


def _candidate_from_group(
    group: list[dict[str, float]],
    shape: tuple[int, int],
    mode: str,
    refine_mask: np.ndarray | None = None,
) -> DisplayCandidate:
    height, width = shape[:2]
    x = int(min(comp["x"] for comp in group))
    y = int(min(comp["y"] for comp in group))
    w = int(max(comp["x"] + comp["w"] for comp in group) - x)
    h = int(max(comp["y"] + comp["h"] for comp in group) - y)

    avg_comp_w = int(np.mean([comp["w"] for comp in group])) if group else max(8, w // 6)
    if mode == "lcd":
        pad_left = max(10, int(w * 0.22))
        pad_right = max(10, int(w * 0.22))
        pad_top = max(10, int(h * 0.85))
        pad_bottom = max(8, int(h * 0.75))
    else:
        pad_left = max(24, int(avg_comp_w * 2.35), int(w * 0.32))
        pad_right = max(12, int(avg_comp_w * 0.9), int(w * 0.12))
        pad_top = max(8, int(h * 0.50))
        pad_bottom = max(6, int(h * 0.28))

    x1 = max(0, x - pad_left)
    y1 = max(0, y - pad_top)
    x2 = min(width, x + w + pad_right)
    y2 = min(height, y + h + pad_bottom)

    if mode == "led" and refine_mask is not None:
        x1, y1, x2, y2 = _refine_bounds_from_mask(refine_mask, x1, y1, x2, y2)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

    return DisplayCandidate(
        mode=mode,
        score=0.0,
        group_size=len(group),
        raw_x1=x,
        raw_y1=y,
        raw_x2=x + w,
        raw_y2=y + h,
        crop_x1=x1,
        crop_y1=y1,
        crop_x2=x2,
        crop_y2=y2,
    )


def _find_led_candidate(bgr: np.ndarray, gray: np.ndarray, lit_mask: np.ndarray) -> DisplayCandidate | None:
    components = _extract_components(lit_mask)
    row_groups = _group_components_into_rows(components)

    best_group: list[dict[str, float]] | None = None
    best_score = -1e18
    for group in row_groups:
        sc = _score_group(group, gray, lit_mask, bgr)
        if sc > best_score:
            best_score = sc
            best_group = group

    if best_group is None:
        return None

    candidate = _candidate_from_group(best_group, gray.shape, "led", lit_mask)
    candidate.score = float(best_score)
    return candidate


def _find_lcd_candidate(bgr: np.ndarray, gray: np.ndarray, dark_mask: np.ndarray) -> DisplayCandidate | None:
    components = _extract_components(dark_mask)
    row_groups = _group_components_into_rows(components)

    best_group: list[dict[str, float]] | None = None
    best_score = -1e18
    for group in row_groups:
        sc = _score_lcd_group(group, gray, dark_mask, bgr)
        if sc > best_score:
            best_score = sc
            best_group = group

    if best_group is None:
        return None

    candidate = _candidate_from_group(best_group, gray.shape, "lcd", None)
    candidate.score = float(best_score)
    return candidate


def _find_led_window_candidate(
    bgr: np.ndarray,
    gray: np.ndarray,
    lit_mask: np.ndarray,
    anchor: DisplayCandidate | None,
) -> DisplayCandidate | None:
    if anchor is None:
        return None

    edge_mask = _build_window_edge_mask(gray)
    contours, _ = cv2.findContours(edge_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best_candidate: DisplayCandidate | None = None
    best_score = -1e18
    image_area = float(gray.shape[0] * gray.shape[1])

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < image_area * 0.0006:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        score = _score_led_window_rect(x, y, w, h, area, gray, bgr, lit_mask, edge_mask, anchor)
        if score > best_score:
            cand = _candidate_from_rect(x, y, w, h, gray.shape, "led")
            cand.group_size = anchor.group_size
            cand.score = float(score)
            best_candidate = cand
            best_score = score

    return best_candidate


def _find_lcd_window_candidate(
    bgr: np.ndarray,
    gray: np.ndarray,
    dark_mask: np.ndarray,
) -> DisplayCandidate | None:
    edge_mask = _build_window_edge_mask(gray)
    contours, _ = cv2.findContours(edge_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best_candidate: DisplayCandidate | None = None
    best_score = -1e18
    image_area = float(gray.shape[0] * gray.shape[1])

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < image_area * 0.0005:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        score = _score_lcd_window_rect(x, y, w, h, area, gray, bgr, dark_mask, edge_mask)
        if score > best_score:
            cand = _candidate_from_rect(x, y, w, h, gray.shape, "lcd")
            cand.score = float(score)
            best_candidate = cand
            best_score = score

    return best_candidate


def _rescore_led_candidate(
    candidate: DisplayCandidate,
    bgr: np.ndarray,
    gray: np.ndarray,
    lit_mask: np.ndarray,
) -> float:
    crop_bgr = _crop_roi(bgr, candidate)
    crop_gray = _crop_roi(gray, candidate)
    crop_mask = _crop_roi(lit_mask, candidate)
    if crop_bgr.size == 0 or crop_gray.size == 0 or crop_mask.size == 0:
        return -1e9

    metrics = _analyze_mask_structure(crop_mask, crop_gray, "led")
    green_ratio = _compute_green_ratio(crop_bgr)
    height, width = gray.shape[:2]
    cx = (candidate.raw_x1 + candidate.raw_x2) / 2.0 / max(width, 1)
    cy = (candidate.raw_y1 + candidate.raw_y2) / 2.0 / max(height, 1)

    score = float(candidate.score)
    score += float(metrics["quality_score"]) * 72.0
    score += min(green_ratio / 0.08, 1.0) * 40.0
    score += max(0.0, 1.0 - abs(cx - 0.5) / 0.5) * 8.0

    if not bool(metrics["is_reliable"]):
        score -= 46.0
    if green_ratio < 0.010:
        score -= 90.0
    elif green_ratio < 0.020:
        score -= 40.0
    if cy > 0.82:
        score -= 65.0
    elif cy > 0.74:
        score -= 32.0
    if cy < 0.22:
        score -= 18.0

    return score


def _rescore_lcd_candidate(
    candidate: DisplayCandidate,
    bgr: np.ndarray,
    gray: np.ndarray,
    dark_mask: np.ndarray,
) -> float:
    crop_bgr = _crop_roi(bgr, candidate)
    crop_gray = _crop_roi(gray, candidate)
    crop_mask = _crop_roi(dark_mask, candidate)
    if crop_bgr.size == 0 or crop_gray.size == 0 or crop_mask.size == 0:
        return -1e9

    metrics = _analyze_mask_structure(crop_mask, crop_gray, "lcd")
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    hue = hsv[:, :, 0]
    val = hsv[:, :, 2]
    low_sat_ratio = float(np.count_nonzero(sat <= 72)) / float(max(sat.size, 1))
    blue_ratio = float(
        np.count_nonzero(
            cv2.inRange(hue, 85, 135)
            & cv2.inRange(sat, 40, 255)
            & cv2.inRange(val, 35, 255)
        )
    ) / float(max(sat.size, 1))

    height, width = gray.shape[:2]
    cx = (candidate.raw_x1 + candidate.raw_x2) / 2.0 / max(width, 1)
    cy = (candidate.raw_y1 + candidate.raw_y2) / 2.0 / max(height, 1)

    score = float(candidate.score)
    score += float(metrics["quality_score"]) * 78.0
    score += min(low_sat_ratio / 0.75, 1.0) * 34.0
    score += max(0.0, 1.0 - (blue_ratio / 0.20)) * 28.0
    score += max(0.0, 1.0 - abs(cx - 0.5) / 0.5) * 8.0

    if not bool(metrics["is_reliable"]):
        score -= 42.0
    if blue_ratio > 0.22:
        score -= 80.0
    elif blue_ratio > 0.14:
        score -= 32.0
    if low_sat_ratio < 0.38:
        score -= 38.0
    if cy > 0.68:
        score -= 70.0
    elif cy > 0.58:
        score -= 28.0
    if cy < 0.12:
        score -= 20.0

    return score


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


def _normalize_digit_boxes(
    digit_boxes: list[dict[str, float]],
    frame_width: int,
) -> list[dict[str, float]]:
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


def _segment_fill_ratios(mask: np.ndarray) -> np.ndarray:
    roi = _normalize_digit_roi(mask)

    regions = [
        roi[6:22, 16:56],   # top
        roi[20:54, 4:22],   # upper-left
        roi[20:54, 50:68],  # upper-right
        roi[48:72, 16:56],  # middle
        roi[66:102, 4:22],  # lower-left
        roi[66:102, 50:68], # lower-right
        roi[98:116, 16:56], # bottom
    ]
    ratios = np.array(
        [float(np.count_nonzero(region)) / float(max(region.size, 1)) for region in regions],
        dtype=np.float32,
    )
    return ratios


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


def locate_display_crop(img: Image.Image, max_dim: int = 1600) -> tuple[Image.Image, Image.Image]:
    """
    Returns:
      crop_pil: localized display crop
      debug_pil: image with selected candidate drawn
    """
    img = resize_keep_aspect(img, max_dim=max_dim)
    bgr = pil_to_bgr(img)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    dbg = bgr.copy()

    lit_mask = _build_lit_mask(bgr)
    dark_mask = _build_dark_digit_mask(gray)

    led_row_candidate = _find_led_candidate(bgr, gray, lit_mask)
    led_window_candidate = _find_led_window_candidate(bgr, gray, lit_mask, led_row_candidate)
    lcd_row_candidate = _find_lcd_candidate(bgr, gray, dark_mask)
    lcd_window_candidate = _find_lcd_window_candidate(bgr, gray, dark_mask)

    led_candidate = led_window_candidate or led_row_candidate
    lcd_candidate = (
        lcd_window_candidate
        if lcd_window_candidate is not None and (
            lcd_row_candidate is None or lcd_window_candidate.score >= lcd_row_candidate.score - 8.0
        )
        else lcd_row_candidate
    )

    led_final_score = (
        _rescore_led_candidate(led_candidate, bgr, gray, lit_mask)
        if led_candidate is not None
        else -1e18
    )
    lcd_final_score = (
        _rescore_lcd_candidate(lcd_candidate, bgr, gray, dark_mask)
        if lcd_candidate is not None
        else -1e18
    )

    chosen: DisplayCandidate | None = None
    if led_candidate and lcd_candidate:
        chosen = led_candidate if led_final_score >= lcd_final_score else lcd_candidate
    else:
        chosen = led_candidate or lcd_candidate

    # Fallback if nothing reasonable
    chosen_score = (
        led_final_score if chosen is led_candidate else lcd_final_score
        if chosen is not None
        else -1e18
    )
    if chosen is None or chosen_score < 45:
        H, W = gray.shape[:2]
        x1 = int(W * 0.18)
        y1 = int(H * 0.48)
        x2 = int(W * 0.82)
        y2 = int(H * 0.72)
        cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 255), 2)
        crop = bgr[y1:y2, x1:x2]
        return bgr_to_pil(crop), bgr_to_pil(dbg)

    if led_row_candidate is not None and led_row_candidate is not led_candidate:
        cv2.rectangle(
            dbg,
            (led_row_candidate.crop_x1, led_row_candidate.crop_y1),
            (led_row_candidate.crop_x2, led_row_candidate.crop_y2),
            (255, 128, 0),
            1,
        )
    if led_candidate is not None:
        cv2.rectangle(
            dbg,
            (led_candidate.crop_x1, led_candidate.crop_y1),
            (led_candidate.crop_x2, led_candidate.crop_y2),
            (0, 255, 255),
            1,
        )
    if lcd_row_candidate is not None and lcd_row_candidate is not lcd_candidate:
        cv2.rectangle(
            dbg,
            (lcd_row_candidate.crop_x1, lcd_row_candidate.crop_y1),
            (lcd_row_candidate.crop_x2, lcd_row_candidate.crop_y2),
            (255, 0, 255),
            1,
        )
    if lcd_candidate is not None:
        cv2.rectangle(
            dbg,
            (lcd_candidate.crop_x1, lcd_candidate.crop_y1),
            (lcd_candidate.crop_x2, lcd_candidate.crop_y2),
            (255, 255, 0),
            1,
        )

    cv2.rectangle(
        dbg,
        (chosen.raw_x1, chosen.raw_y1),
        (chosen.raw_x2, chosen.raw_y2),
        (0, 0, 255) if chosen.mode == "lcd" else (0, 255, 255),
        2,
    )
    cv2.rectangle(
        dbg,
        (chosen.crop_x1, chosen.crop_y1),
        (chosen.crop_x2, chosen.crop_y2),
        (0, 255, 0),
        2,
    )

    crop = bgr[chosen.crop_y1:chosen.crop_y2, chosen.crop_x1:chosen.crop_x2]

    return bgr_to_pil(crop), bgr_to_pil(dbg)
