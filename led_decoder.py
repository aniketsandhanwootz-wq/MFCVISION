from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import re
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from vision import (
    LocalDecodeResult,
    _build_lit_mask,
    _classify_decimal_candidates,
    _extract_active_boxes,
    _normalize_digit_boxes,
    pil_to_bgr,
    resize_keep_aspect,
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

SEGMENT_NAMES = (
    "top",
    "upper_left",
    "upper_right",
    "middle",
    "lower_left",
    "lower_right",
    "bottom",
)


@dataclass
class LedCropScore:
    score: float
    green_ratio: float
    horizontal_band_strength: float
    slot_regularity: float
    decimal_presence: float
    active_density: float
    left_edge_activity: float
    slot_count: int
    reason: str


@dataclass
class LedSlotEvidence:
    index: int
    x1: int
    y1: int
    x2: int
    y2: int
    state: str
    digit: Optional[str]
    confidence: float
    top_candidates: list[str] = field(default_factory=list)
    segment_scores: list[float] = field(default_factory=list)


@dataclass
class LedDecimalInfo:
    visible: bool
    after_slot: Optional[int]
    confidence: float
    x1: Optional[int] = None
    y1: Optional[int] = None
    x2: Optional[int] = None
    y2: Optional[int] = None


@dataclass
class LedLeadingOneInfo:
    present: bool
    confidence: float
    x1: Optional[int] = None
    y1: Optional[int] = None
    x2: Optional[int] = None
    y2: Optional[int] = None
    reason: str = ""


@dataclass
class LedResolveResult:
    value_text: Optional[str]
    confidence: float
    reason: str
    top_candidates: list[str] = field(default_factory=list)


@dataclass
class LedDecodeArtifact:
    ok: bool
    value_text: Optional[str]
    confidence: float
    digit_count: int
    decimal_count: int
    reason: str
    crop_score: LedCropScore
    slots: list[LedSlotEvidence] = field(default_factory=list)
    decimal_info: LedDecimalInfo = field(
        default_factory=lambda: LedDecimalInfo(False, None, 0.0)
    )
    leading_one_info: LedLeadingOneInfo = field(
        default_factory=lambda: LedLeadingOneInfo(False, 0.0)
    )
    top_candidates: list[str] = field(default_factory=list)

    def to_local_decode_result(self) -> LocalDecodeResult:
        return LocalDecodeResult(
            ok=self.ok,
            value_text=self.value_text,
            confidence=round(self.confidence, 3),
            digit_count=self.digit_count,
            decimal_count=self.decimal_count,
            reason=self.reason,
        )

    def to_payload(self) -> dict:
        return {
            "available": True,
            "ok": self.ok,
            "value_text": self.value_text,
            "confidence": round(self.confidence, 3),
            "digit_count": self.digit_count,
            "decimal_count": self.decimal_count,
            "reason": self.reason,
            "crop_score": asdict(self.crop_score),
            "decimal_info": asdict(self.decimal_info),
            "leading_one_info": asdict(self.leading_one_info),
            "top_candidates": list(self.top_candidates),
            "slots": [asdict(slot) for slot in self.slots],
        }


def _digits_only(value_text: str | None) -> str:
    return value_text.replace(".", "") if value_text else ""


def _count_integer_digits(value_text: str | None) -> int:
    if not value_text:
        return 0
    return len(value_text.split(".", 1)[0])


def _range_score(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    if low <= value <= high:
        return 1.0
    if value < low:
        return max(0.0, value / max(low, 1e-6))
    overshoot = (value - high) / max(high, 1e-6)
    return max(0.0, 1.0 - overshoot)


def _green_ratio_from_bgr(bgr: np.ndarray) -> float:
    blue, green, red = cv2.split(bgr)
    green_dom = cv2.subtract(green, cv2.max(red, blue))
    return float(np.count_nonzero(green_dom > 8)) / float(max(green_dom.size, 1))


def _normalize_digit_roi(mask: np.ndarray) -> np.ndarray:
    roi = cv2.copyMakeBorder(mask, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=0)
    roi = cv2.resize(roi, (84, 140), interpolation=cv2.INTER_NEAREST)
    return cv2.morphologyEx(
        roi,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )


def build_led_active_mask(img: np.ndarray | Image.Image) -> np.ndarray:
    if isinstance(img, Image.Image):
        bgr = pil_to_bgr(resize_keep_aspect(img, max_dim=900))
    else:
        bgr = img

    base_mask = _build_lit_mask(bgr)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    blue, green, red = cv2.split(bgr)
    value = hsv[:, :, 2]
    luminance = lab[:, :, 0]

    gamma_lut = np.array([((idx / 255.0) ** 0.72) * 255 for idx in range(256)], dtype=np.uint8)
    gamma_value = cv2.LUT(value, gamma_lut)
    gamma_l = cv2.LUT(luminance, gamma_lut)
    green_dom = cv2.subtract(green, cv2.max(red, blue))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6))
    local_contrast = cv2.absdiff(clahe.apply(gamma_l), cv2.GaussianBlur(gamma_l, (9, 9), 0))

    green_mask = (
        cv2.inRange(green_dom, 8, 255)
        & cv2.inRange(green, 55, 255)
        & cv2.inRange(gamma_value, 64, 255)
    )
    structure_mask = (
        cv2.adaptiveThreshold(
            gamma_l,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            27,
            -8,
        )
        & cv2.inRange(local_contrast, 10, 255)
        & cv2.inRange(hsv[:, :, 0], 25, 110)
    )
    mask = base_mask | green_mask | structure_mask
    k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k1, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k1, iterations=2)
    mask = cv2.dilate(mask, k2, iterations=1)
    return mask


def _compute_slot_regularity(digit_boxes: list[dict[str, float]]) -> float:
    if len(digit_boxes) <= 1:
        return 0.0
    centers = np.array([box["cx"] for box in digit_boxes], dtype=np.float32)
    widths = np.array([box["w"] for box in digit_boxes], dtype=np.float32)
    gaps = np.diff(centers)
    if gaps.size == 0:
        return 0.0
    gap_cv = float(np.std(gaps) / max(np.mean(gaps), 1.0))
    width_cv = float(np.std(widths) / max(np.mean(widths), 1.0))
    return max(0.0, min(1.0, 1.0 - ((gap_cv * 0.65) + (width_cv * 0.35))))


def _row_band_strength(mask: np.ndarray) -> float:
    row_energy = np.count_nonzero(mask, axis=1).astype(np.float32)
    if row_energy.size == 0 or row_energy.max() <= 0:
        return 0.0
    threshold = max(1.0, row_energy.max() * 0.28)
    active_rows = np.where(row_energy >= threshold)[0]
    if active_rows.size == 0:
        return 0.0
    band_height = float(active_rows[-1] - active_rows[0] + 1) / float(max(mask.shape[0], 1))
    return _range_score(band_height, 0.08, 0.30)


def _left_edge_activity(mask: np.ndarray, digit_boxes: list[dict[str, float]]) -> float:
    if not digit_boxes:
        return 0.0
    first = min(digit_boxes, key=lambda box: box["x1"])
    slot_width = max(6, int(round(np.median([box["w"] for box in digit_boxes]))))
    x2 = int(first["x1"])
    x1 = max(0, x2 - int(slot_width * 0.85))
    if x2 <= x1:
        return 0.0
    roi = mask[:, x1:x2]
    active = np.where(roi > 0)
    if active[0].size == 0:
        return 0.0
    width = float(active[1].max() - active[1].min() + 1)
    height = float(active[0].max() - active[0].min() + 1)
    width_score = _range_score(width / float(max(slot_width, 1)), 0.08, 0.42)
    height_score = _range_score(height / float(max(mask.shape[0], 1)), 0.32, 0.90)
    return round((width_score * 0.45) + (height_score * 0.55), 3)


def score_led_crop(img: Image.Image | np.ndarray) -> LedCropScore:
    if isinstance(img, Image.Image):
        img = resize_keep_aspect(img, max_dim=900)
        bgr = pil_to_bgr(img)
    else:
        bgr = img

    mask = build_led_active_mask(bgr)
    boxes = _extract_active_boxes(mask)
    digit_boxes, decimal_boxes = _classify_decimal_candidates(boxes)
    digit_boxes = _normalize_digit_boxes(digit_boxes, mask.shape[1])

    active_density = float(np.count_nonzero(mask)) / float(max(mask.size, 1))
    band_strength = _row_band_strength(mask)
    slot_regularity = _compute_slot_regularity(digit_boxes)
    decimal_presence = 1.0 if decimal_boxes else 0.0
    left_activity = _left_edge_activity(mask, digit_boxes)
    green_ratio = _green_ratio_from_bgr(bgr)

    score = 0.0
    score += _range_score(green_ratio, 0.010, 0.18) * 0.22
    score += band_strength * 0.24
    score += slot_regularity * 0.23
    score += _range_score(active_density, 0.008, 0.18) * 0.16
    score += decimal_presence * 0.08
    score += left_activity * 0.07

    reasons: list[str] = []
    if slot_regularity >= 0.55:
        reasons.append("regular digit spacing")
    if band_strength >= 0.55:
        reasons.append("stable horizontal band")
    if green_ratio >= 0.010:
        reasons.append("strong green LED signal")
    if left_activity >= 0.45:
        reasons.append("left-edge activity present")
    if decimal_presence >= 1.0:
        reasons.append("decimal candidate present")
    reason = ", ".join(reasons) or "weak LED crop structure"

    return LedCropScore(
        score=round(score, 3),
        green_ratio=round(green_ratio, 4),
        horizontal_band_strength=round(band_strength, 3),
        slot_regularity=round(slot_regularity, 3),
        decimal_presence=round(decimal_presence, 3),
        active_density=round(active_density, 4),
        left_edge_activity=round(left_activity, 3),
        slot_count=len(digit_boxes),
        reason=reason,
    )


def _segment_regions(width: int, height: int) -> list[tuple[int, int, int, int]]:
    x_pad = max(3, width // 10)
    y_pad = max(4, height // 14)
    mid_y = height // 2
    seg_h = max(5, height // 10)
    seg_w = max(5, width // 6)
    return [
        (x_pad, y_pad, width - x_pad, y_pad + seg_h),
        (0, y_pad + seg_h // 2, seg_w, mid_y - seg_h // 3),
        (width - seg_w, y_pad + seg_h // 2, width, mid_y - seg_h // 3),
        (x_pad, mid_y - seg_h // 2, width - x_pad, mid_y + seg_h // 2),
        (0, mid_y + seg_h // 3, seg_w, height - y_pad - seg_h // 2),
        (width - seg_w, mid_y + seg_h // 3, width, height - y_pad - seg_h // 2),
        (x_pad, height - y_pad - seg_h, width - x_pad, height - y_pad),
    ]


def decode_seven_segment(slot_img: np.ndarray) -> dict[str, object]:
    roi = _normalize_digit_roi(slot_img)
    roi_bin = (roi > 0).astype(np.uint8)
    height, width = roi_bin.shape[:2]
    regions = _segment_regions(width, height)
    segment_scores: list[float] = []
    for x1, y1, x2, y2 in regions:
        patch = roi_bin[y1:y2, x1:x2]
        score = float(np.count_nonzero(patch)) / float(max(patch.size, 1))
        segment_scores.append(score)

    best_digit: Optional[str] = None
    best_score = -1e9
    ranked: list[tuple[float, str]] = []
    for digit, pattern in SEGMENT_PATTERNS.items():
        score = 0.0
        for segment_score, is_on in zip(segment_scores, pattern):
            target = 1.0 if is_on else 0.0
            score += 1.0 - abs(segment_score - target)
        score /= len(pattern)
        if digit == "8" and segment_scores[4] < 0.18:
            score -= 0.18
        if digit == "5" and segment_scores[4] < 0.18:
            score += 0.06
        if digit == "7" and segment_scores[0] < 0.18:
            score -= 0.14
        ranked.append((score, digit))
        if score > best_score:
            best_score = score
            best_digit = digit

    ranked.sort(reverse=True)
    second_score = ranked[1][0] if len(ranked) > 1 else 0.0
    confidence = max(0.0, min(1.0, 0.18 + (best_score * 0.78) + ((best_score - second_score) * 0.55)))
    return {
        "digit": best_digit,
        "confidence": round(confidence, 3),
        "top_candidates": [digit for _, digit in ranked[:2]],
        "segment_scores": [round(score, 3) for score in segment_scores],
    }


def find_led_slots(img: np.ndarray, mask: np.ndarray) -> tuple[list[LedSlotEvidence], list[dict[str, float]]]:
    boxes = _extract_active_boxes(mask)
    digit_boxes, decimal_boxes = _classify_decimal_candidates(boxes)
    digit_boxes = _normalize_digit_boxes(digit_boxes, mask.shape[1])

    slots: list[LedSlotEvidence] = []
    for index, box in enumerate(digit_boxes):
        x1 = int(box["x1"])
        x2 = int(box["x2"])
        y1 = int(box["y1"])
        y2 = int(box["y2"])
        decoded = decode_seven_segment(mask[y1:y2, x1:x2])
        confidence = float(decoded["confidence"])
        slots.append(
            LedSlotEvidence(
                index=index,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                state="active" if confidence >= 0.65 else "weak_active",
                digit=decoded["digit"],
                confidence=confidence,
                top_candidates=list(decoded["top_candidates"]),
                segment_scores=list(decoded["segment_scores"]),
            )
        )
    return slots, decimal_boxes


def detect_decimal_point(
    img: np.ndarray,
    mask: np.ndarray,
    slots: list[LedSlotEvidence],
    decimal_boxes: list[dict[str, float]],
) -> LedDecimalInfo:
    if not decimal_boxes:
        return LedDecimalInfo(False, None, 0.0)

    dot = sorted(decimal_boxes, key=lambda box: (box["x1"], box["y1"]))[-1]
    after_slot = 0
    for slot in slots:
        if dot["cx"] > slot.x2:
            after_slot = slot.index + 1
    if after_slot <= 0 or after_slot >= len(slots):
        return LedDecimalInfo(False, None, 0.0)

    return LedDecimalInfo(
        visible=True,
        after_slot=after_slot,
        confidence=0.86 if len(decimal_boxes) == 1 else 0.68,
        x1=int(dot["x1"]),
        y1=int(dot["y1"]),
        x2=int(dot["x2"]),
        y2=int(dot["y2"]),
    )


def detect_weak_leading_one(
    slots: list[LedSlotEvidence],
    mask: np.ndarray,
) -> LedLeadingOneInfo:
    if not slots:
        return LedLeadingOneInfo(False, 0.0, reason="no decoded slots")

    first = slots[0]
    slot_width = max(6, int(round(np.median([slot.x2 - slot.x1 for slot in slots]))))
    x2 = first.x1
    x1 = max(0, x2 - int(slot_width * 0.82))
    if x2 <= x1:
        return LedLeadingOneInfo(False, 0.0, reason="no room left of first slot")

    roi = mask[:, x1:x2]
    ys, xs = np.where(roi > 0)
    if ys.size == 0:
        return LedLeadingOneInfo(False, 0.0, reason="left of first slot is dark")

    width = float(xs.max() - xs.min() + 1)
    height = float(ys.max() - ys.min() + 1)
    width_score = _range_score(width / float(max(slot_width, 1)), 0.08, 0.36)
    height_score = _range_score(height / float(max(mask.shape[0], 1)), 0.34, 0.86)
    col_energy = np.count_nonzero(roi, axis=0).astype(np.float32)
    if col_energy.max() <= 0:
        return LedLeadingOneInfo(False, 0.0, reason="no structured left-edge activity")
    active_cols = np.where(col_energy >= max(1.0, col_energy.max() * 0.35))[0]
    if active_cols.size == 0:
        return LedLeadingOneInfo(False, 0.0, reason="no concentrated left-edge activity")
    groups = np.split(active_cols, np.where(np.diff(active_cols) != 1)[0] + 1)
    group_count = len([group for group in groups if group.size > 0])
    group_score = 1.0 if group_count <= 2 else max(0.0, 1.0 - ((group_count - 2) * 0.35))
    confidence = round((width_score * 0.35) + (height_score * 0.40) + (group_score * 0.25), 3)
    present = confidence >= 0.62
    return LedLeadingOneInfo(
        present=present,
        confidence=confidence,
        x1=x1 + int(xs.min()),
        y1=int(ys.min()),
        x2=x1 + int(xs.max() + 1),
        y2=int(ys.max() + 1),
        reason="weak leading 1 candidate detected" if present else "no convincing weak leading 1",
    )


def _compose_slot_text(slots: list[LedSlotEvidence], decimal_info: LedDecimalInfo) -> Optional[str]:
    digits = "".join(slot.digit or "" for slot in slots if slot.digit)
    if not digits:
        return None
    if decimal_info.visible and decimal_info.after_slot is not None:
        after = decimal_info.after_slot
        if 0 < after < len(digits):
            return f"{digits[:after]}.{digits[after:]}"
        if after == 0:
            return f"0.{digits}"
    return digits


def resolve_led_read(
    gemini_read: str | None,
    local_slots: list[LedSlotEvidence],
    decimal_info: LedDecimalInfo,
    leading_one_info: LedLeadingOneInfo,
) -> LedResolveResult:
    local_text = _compose_slot_text(local_slots, decimal_info)
    if not local_text:
        return LedResolveResult(None, 0.0, "no local LED slot reading")

    slot_confidence = float(np.mean([slot.confidence for slot in local_slots])) if local_slots else 0.0
    top_candidates = [local_text]
    resolved_text = local_text
    reason = "local LED slot decode"
    confidence = slot_confidence

    if (
        leading_one_info.present
        and not local_text.startswith("1")
        and _count_integer_digits(local_text) <= 1
    ):
        prefixed = f"1{local_text}"
        top_candidates.insert(0, prefixed)
        if gemini_read and re.fullmatch(r"\d+(\.\d+)?", gemini_read):
            if _count_integer_digits(gemini_read) <= 1 or gemini_read.startswith(("7", "8", "0")):
                resolved_text = prefixed
                confidence = min(0.94, max(confidence, leading_one_info.confidence * 0.85))
                reason = "weak leading 1 upgraded a short LED reading"

    if gemini_read and re.fullmatch(r"\d+(\.\d+)?", gemini_read):
        if _digits_only(gemini_read) == _digits_only(local_text):
            confidence = min(0.96, max(confidence, 0.82))
            reason = "local LED slot decode matched Gemini digits"

    return LedResolveResult(
        value_text=resolved_text,
        confidence=round(confidence, 3),
        reason=reason,
        top_candidates=top_candidates,
    )


def decode_led_display(crop_img: Image.Image) -> LedDecodeArtifact:
    crop_img = resize_keep_aspect(crop_img, max_dim=900)
    bgr = pil_to_bgr(crop_img)
    mask = build_led_active_mask(bgr)
    crop_score = score_led_crop(bgr)
    slots, decimal_boxes = find_led_slots(bgr, mask)
    decimal_info = detect_decimal_point(bgr, mask, slots, decimal_boxes)
    leading_one_info = detect_weak_leading_one(slots, mask)
    resolved = resolve_led_read(None, slots, decimal_info, leading_one_info)

    if not slots or not resolved.value_text:
        return LedDecodeArtifact(
            ok=False,
            value_text=None,
            confidence=0.0,
            digit_count=len(slots),
            decimal_count=1 if decimal_info.visible else 0,
            reason="no active LED slot sequence found",
            crop_score=crop_score,
            slots=slots,
            decimal_info=decimal_info,
            leading_one_info=leading_one_info,
            top_candidates=[],
        )

    slot_confidence = float(np.mean([slot.confidence for slot in slots])) if slots else 0.0
    confidence = min(
        0.96,
        max(
            resolved.confidence,
            (slot_confidence * 0.72) + (crop_score.score * 0.22) + (decimal_info.confidence * 0.06),
        ),
    )
    reason = resolved.reason
    if decimal_info.visible:
        reason = f"{reason}; decimal detected from LED slot structure"
    if leading_one_info.present:
        reason = f"{reason}; leading-one candidate available"

    return LedDecodeArtifact(
        ok=True,
        value_text=resolved.value_text,
        confidence=round(confidence, 3),
        digit_count=len(slots),
        decimal_count=1 if decimal_info.visible else 0,
        reason=reason,
        crop_score=crop_score,
        slots=slots,
        decimal_info=decimal_info,
        leading_one_info=leading_one_info,
        top_candidates=resolved.top_candidates,
    )
