from __future__ import annotations

import io
import json
import os
import re
import time
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field, ValidationError

from image_enhance import make_enhanced_display_image
from vision import (
    CropDiagnostics,
    LocalDecodeResult,
    analyze_crop_diagnostics,
    decode_display_crop,
    resize_keep_aspect,
)

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
PROMPT_PATH = BASE_DIR / "prompts" / "scale_reader.txt"
STATIC_DIR = BASE_DIR / "static"

MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-flash")
_raw_localizer = os.getenv("LOCALIZER_MODEL_NAME", "gemini-2.5-flash")
# Safety: never allow the lite model for localization — it consistently mislocalizes
LOCALIZER_MODEL_NAME = MODEL_NAME if "lite" in _raw_localizer.lower() else _raw_localizer

MAX_IMAGE_MB = int(os.getenv("MAX_IMAGE_MB", "10"))
MAX_IMAGE_BYTES = MAX_IMAGE_MB * 1024 * 1024
MAX_DIMENSION = int(os.getenv("MAX_DIMENSION", "1200"))
LOCALIZER_MAX_DIMENSION = int(os.getenv("LOCALIZER_MAX_DIMENSION", "1024"))
READ_TEMPERATURE = float(os.getenv("READ_TEMPERATURE", "0.0"))
LOCALIZER_TEMPERATURE = float(os.getenv("LOCALIZER_TEMPERATURE", "0.0"))
EXPECTED_DECIMALS = (os.getenv("EXPECTED_DECIMALS") or os.getenv("FIXED_DECIMALS") or "").strip()
LOCAL_DECODER_MIN_CONFIDENCE = float(os.getenv("LOCAL_DECODER_MIN_CONFIDENCE", "0.90"))
LOCALIZER_MIN_CONFIDENCE = float(os.getenv("LOCALIZER_MIN_CONFIDENCE", "0.45"))
# Confidence threshold above which we skip the region-refiner pass (saves ~1s latency)
LOCALIZER_SKIP_REFINE_THRESHOLD = float(os.getenv("LOCALIZER_SKIP_REFINE_THRESHOLD", "0.90"))

if not os.getenv("GEMINI_API_KEY"):
    raise RuntimeError("Missing GEMINI_API_KEY in environment.")

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI(title="Gemini Scale Reader", version="9.2.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

_LAST_PREVIEWS: dict[str, bytes] = {}

# ------------------------------------------------------------------ #
# Prompt cache — load once, never hit disk again                       #
# ------------------------------------------------------------------ #
_PROMPT_CACHE: str | None = None

def load_prompt_text() -> str:
    global _PROMPT_CACHE
    if _PROMPT_CACHE is None:
        if not PROMPT_PATH.exists():
            raise RuntimeError(f"Prompt file not found: {PROMPT_PATH}")
        _PROMPT_CACHE = PROMPT_PATH.read_text(encoding="utf-8").strip()
    return _PROMPT_CACHE


class ReadingResult(BaseModel):
    status: Literal["ok", "needs_review"]
    value_text: Optional[str] = None
    value_number: Optional[float] = None
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    ignored_text_present: bool


class LocalizationResult(BaseModel):
    found: bool
    x1: Optional[int] = Field(default=None, ge=0, le=1000)
    y1: Optional[int] = Field(default=None, ge=0, le=1000)
    x2: Optional[int] = Field(default=None, ge=0, le=1000)
    y2: Optional[int] = Field(default=None, ge=0, le=1000)
    confidence: float = Field(ge=0.0, le=1.0)
    display_kind: Literal["led", "lcd", "unknown"] = "unknown"
    reason: str


def validate_upload(file: UploadFile, data: bytes) -> None:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image uploads are allowed.")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"Image too large. Max allowed size is {MAX_IMAGE_MB} MB.",
        )


def open_image(data: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(data))
        return img.convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")


def pil_to_jpeg_bytes(img: Image.Image, quality: int = 92) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def make_placeholder_preview(size: tuple[int, int], message: str) -> Image.Image:
    width, height = size
    canvas = Image.new("RGB", (max(320, width), max(120, height)), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width - 1, canvas.height - 1), outline=(210, 210, 210), width=2)
    draw.text((18, max(18, canvas.height // 2 - 10)), message, fill=(120, 120, 120))
    return canvas


# ----------------------------
# Gemini localizer helpers
# ----------------------------

def _call_gemini_localizer_once(
    localizer_original_img: Image.Image,
    localizer_enhanced_img: Image.Image,
    instructions: str,
    *,
    model_name: str | None = None,
) -> LocalizationResult:
    selected_model = model_name or LOCALIZER_MODEL_NAME
    try:
        response = client.models.generate_content(
            model=selected_model,
            contents=[
                instructions,
                "PRIMARY image: ORIGINAL full photo. Use this as the source of truth for physical display location.",
                localizer_original_img,
                "CONTEXT image: ENHANCED full photo. Use this only to help visibility of the display window, not to invent a location.",
                localizer_enhanced_img,
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=LocalizationResult,
                temperature=LOCALIZER_TEMPERATURE,
            ),
        )
    except Exception:
        if selected_model != MODEL_NAME:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[
                    instructions,
                    "PRIMARY image: ORIGINAL full photo. Use this as the source of truth for physical display location.",
                    localizer_original_img,
                    "CONTEXT image: ENHANCED full photo. Use this only to help visibility of the display window, not to invent a location.",
                    localizer_enhanced_img,
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=LocalizationResult,
                    temperature=LOCALIZER_TEMPERATURE,
                ),
            )
        else:
            raise

    if getattr(response, "parsed", None) is not None:
        parsed = response.parsed
        if isinstance(parsed, LocalizationResult):
            return parsed
        if isinstance(parsed, dict):
            return LocalizationResult(**parsed)

    raw_text = getattr(response, "text", None)
    if not raw_text:
        raise HTTPException(status_code=502, detail="Gemini localizer returned empty response.")

    try:
        payload = json.loads(raw_text)
        return LocalizationResult(**payload)
    except (json.JSONDecodeError, ValidationError) as e:
        raise HTTPException(status_code=502, detail=f"Invalid Gemini localization JSON response: {e}")


def _primary_localizer_instructions() -> str:
    return (
        "You are a precision bounding-box locator. Your ONLY job is to find the numeric display window. "
        "Do NOT read or report any numbers.\n\n"
        "WHAT YOU ARE LOOKING FOR — the display window is ONE of these:\n"
        "  (A) LED display: a DARK rectangular panel/window with BRIGHT GREEN or orange digit segments "
        "glowing in a horizontal row. The dark background is part of the window. "
        "It is typically in the LOWER portion of the scale body, BELOW the weighing bowl/pan.\n"
        "  (B) LCD display: a rectangular screen with DARK digit segments on a GRAY/LIGHT background. "
        "Common on micrometers, calipers, gauges. Typically in the CENTER or UPPER part of the device body.\n\n"
        "HOW TO LOCATE IT:\n"
        "1. First identify what type of measuring device is in the image.\n"
        "2. For a PRECISION BALANCE / WEIGHING SCALE with a bowl or pan on top:\n"
        "   - The bowl/pan is at the TOP. Ignore it.\n"
        "   - The LED display is BELOW the bowl, embedded in the scale body.\n"
        "   - Look for a dark rectangular cutout in the scale body with glowing green digits.\n"
        "   - It is ABOVE the control buttons.\n"
        "   - It is in the LOWER half of the overall image, not the upper half.\n"
        "   - The box height should be ONLY the display panel — do NOT include the bowl or buttons.\n"
        "3. For a THICKNESS GAUGE / MICROMETER / CALIPER:\n"
        "   - The LCD screen is a small rectangle, usually in the upper or middle section of the device.\n"
        "   - It has dark digits on a light gray/silver background.\n"
        "   - Box ONLY the screen rectangle — do NOT include buttons below the screen.\n\n"
        "CRITICAL BOX SIZE RULES:\n"
        "  - The box width should NOT span the full image width (x1=0, x2=1000 is always wrong).\n"
        "  - The box height (y2-y1) should be roughly 8-20% of the image height for a typical display.\n"
        "  - A box taller than 30% of the image height is always wrong — it includes non-display content.\n"
        "  - Tightly box ONLY the display window rectangle, not the surrounding machine body.\n\n"
        "FORBIDDEN — your box must NOT cover:\n"
        "  - The weighing bowl, pan, dish, or its metal support ring\n"
        "  - Any blank white, gray, or light-colored strip without visible digit segments\n"
        "  - Physical push buttons (round/oval colored buttons)\n"
        "  - Brand name, model number, or label text areas\n"
        "  - The blue or black machine body without a screen\n\n"
        "OUTPUT RULES:\n"
        "  - Coordinates are integers 0..1000, normalized relative to the full image width/height.\n"
        "  - x1,y1 = top-left corner of the display window. x2,y2 = bottom-right corner.\n"
        "  - The box must be WIDER than it is tall (landscape rectangle).\n"
        "  - For LED scales: the display window center y-coordinate should be BELOW 400.\n"
        "  - For LCD gauges: box ONLY the screen, stop BEFORE any buttons below it.\n"
        "  - If you cannot find a clear display window, return found=false.\n"
        "  - display_kind: 'led' for glowing segments, 'lcd' for dark-on-light segments, 'unknown' if unsure.\n"
        "  - confidence: your confidence that the box correctly surrounds the DISPLAY WINDOW ONLY.\n"
        "  - reason: describe the physical location only.\n"
        "  - NEVER mention any numeric value in the reason field."
    )


def _refine_localizer_instructions(previous: LocalizationResult) -> str:
    return (
        "Role: display-window localizer refinement only. Do NOT read digits. "
        "You are verifying and correcting a previous display-window box proposal. "
        f"Previous proposal (normalized 0..1000): x1={previous.x1}, y1={previous.y1}, x2={previous.x2}, y2={previous.y2}, "
        f"kind={previous.display_kind}, confidence={previous.confidence}. "
        "Return a corrected box around the FULL PHYSICAL DISPLAY WINDOW / SCREEN only. "
        "If the previous box covers bowl rim, metal ring, blue strip, labels, branding, device face without screen, or only part of the display, correct it. "
        "For LED devices: box the full dark display panel/window with the whole row inside. "
        "For LCD devices: box the full inner LCD screen rectangle only — stop before any buttons below the screen. "
        "CRITICAL: The box must NOT span the full image width (x1=0, x2=1000 is always wrong). "
        "CRITICAL: The box height (y2-y1) must be roughly 8-20% of image height — never more than 30%. "
        "For LED scales, the display center should be in the lower half of the image (y center > 400 on 0-1000 scale). "
        "If the display is genuinely not localizable, return found=false rather than a bad partial box. "
        "Coordinates must be integers normalized 0..1000 relative to the full input image. "
        "display_kind must be one of: led, lcd, unknown. "
        "In reason, describe location only and never mention an inferred reading."
    )


def _region_refine_instructions(display_kind_hint: str, primary_cy_frac: float = 0.5) -> str:
    if display_kind_hint == "led":
        # If the primary box was in the upper half, warn the refiner explicitly
        position_hint = ""
        if primary_cy_frac < 0.55:
            position_hint = (
                "\nIMPORTANT: The previous localization attempt may have incorrectly "
                "landed on the BOWL, PAN, or PLATFORM at the top of the scale. "
                "The actual LED display is LOWER — look specifically in the BOTTOM "
                "60% of this image for the dark rectangular panel with green digits. "
                "Ignore the bowl/pan/rim at the top.\n"
            )
        what_to_find = (
            "You are looking for a DARK rectangular panel with BRIGHT GREEN or orange glowing digit segments. "
            "The dark background panel is part of the display — box it fully, not just the bright digits. "
            "It sits embedded in the scale BODY, below the weighing bowl/pan and above the control buttons."
            f"{position_hint}"
        )
    elif display_kind_hint == "lcd":
        what_to_find = (
            "You are looking for a rectangular LCD screen with DARK digit segments on a LIGHT/GRAY background. "
            "Box the full screen rectangle including its frame. "
            "Stop before any physical buttons below the screen."
        )
    else:
        what_to_find = (
            "You are looking for either: (A) a dark panel with bright glowing digit segments (LED), "
            "or (B) a light rectangle with dark digit segments (LCD). "
            "Scan the entire image carefully for either type."
        )

    return (
        "You are a precision bounding-box locator operating on a CROPPED SEARCH REGION. "
        "Your ONLY job is to find the numeric display window within this crop. "
        "Do NOT read or report any numbers.\n\n"
        f"{what_to_find}\n\n"
        "FORBIDDEN boxes — do NOT return a box around:\n"
        "  - Blank white or light areas with no digit segments\n"
        "  - Physical buttons or controls\n"
        "  - Weighing bowl, pan, or metal ring\n"
        "  - Brand labels or text-only areas\n"
        "  - Machine body without a screen\n\n"
        "OUTPUT RULES:\n"
        "  - Coordinates are integers 0..1000 relative to THIS CROPPED IMAGE (not the original).\n"
        "  - The box must be landscape (wider than tall).\n"
        "  - The box height should be 8-25% of this cropped image height.\n"
        "  - If no clear display window is visible here, return found=false.\n"
        "  - display_kind: 'led', 'lcd', or 'unknown'.\n"
        "  - NEVER mention any numeric value in the reason field."
    )


def call_gemini_localizer(
    localizer_original_img: Image.Image,
    localizer_enhanced_img: Image.Image,
    *,
    previous: LocalizationResult | None = None,
) -> LocalizationResult:
    instructions = (
        _primary_localizer_instructions()
        if previous is None
        else _refine_localizer_instructions(previous)
    )
    return _call_gemini_localizer_once(
        localizer_original_img,
        localizer_enhanced_img,
        instructions,
        model_name=LOCALIZER_MODEL_NAME,
    )


def call_gemini_region_localizer(
    region_original_img: Image.Image,
    region_enhanced_img: Image.Image,
    *,
    display_kind_hint: str = "unknown",
    primary_cy_frac: float = 0.5,
) -> LocalizationResult:
    return _call_gemini_localizer_once(
        region_original_img,
        region_enhanced_img,
        _region_refine_instructions(display_kind_hint, primary_cy_frac),
        model_name=LOCALIZER_MODEL_NAME,
    )


# ----------------------------
# Box / crop utilities
# ----------------------------

def localization_box_to_pixels(
    localization: LocalizationResult,
    original_size: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    if not localization.found:
        return None
    if None in (localization.x1, localization.y1, localization.x2, localization.y2):
        return None

    width, height = original_size
    x1 = int(round(width * (localization.x1 / 1000.0)))
    y1 = int(round(height * (localization.y1 / 1000.0)))
    x2 = int(round(width * (localization.x2 / 1000.0)))
    y2 = int(round(height * (localization.y2 / 1000.0)))
    return (x1, y1, x2, y2)


def pixels_to_norm1000(
    box: tuple[int, int, int, int],
    image_size: tuple[int, int],
) -> dict[str, int]:
    width, height = image_size
    x1, y1, x2, y2 = box
    return {
        "x1": int(round((x1 / max(width, 1)) * 1000)),
        "y1": int(round((y1 / max(height, 1)) * 1000)),
        "x2": int(round((x2 / max(width, 1)) * 1000)),
        "y2": int(round((y2 / max(height, 1)) * 1000)),
    }


def is_valid_localization_box(
    box: tuple[int, int, int, int] | None,
    image_size: tuple[int, int],
) -> bool:
    if box is None:
        return False
    x1, y1, x2, y2 = box
    width, height = image_size
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0 or bh <= 0:
        return False
    area_ratio = (bw * bh) / float(max(width * height, 1))
    aspect = bw / float(max(bh, 1))
    if bw < max(20, int(width * 0.04)) or bh < max(14, int(height * 0.02)):
        return False
    if area_ratio < 0.0008 or area_ratio > 0.40:
        return False
    if aspect < 0.8 or aspect > 9.5:
        return False
    return True


def _is_bad_localization_box(
    box: tuple[int, int, int, int],
    image_size: tuple[int, int],
    display_kind: str,
) -> bool:
    """
    Returns True if the box is almost certainly NOT a display window.
    Any single rule triggering = reject.
    """
    x1, y1, x2, y2 = box
    img_w, img_h = image_size
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0 or bh <= 0:
        return True

    w_frac = bw / max(img_w, 1)
    h_frac = bh / max(img_h, 1)
    cy_frac = ((y1 + y2) / 2.0) / max(img_h, 1)
    aspect = bw / max(bh, 1)

    # Rule 1: Spans nearly the full image width — never a tight display box
    if w_frac > 0.92:
        return True

    # Rule 2: Very wide + very thin + in top half = bowl rim or strip
    if w_frac > 0.60 and h_frac < 0.10 and cy_frac < 0.50:
        return True

    # Rule 3: Any box thinner than 4% of image height
    if h_frac < 0.04:
        return True

    # Rule 4: Box taller than 30% of image — includes non-display content
    if h_frac > 0.30:
        return True

    # Rule 5: For LED scales, display is never in the very top quarter
    if display_kind == "led" and cy_frac < 0.25:
        return True

    # Rule 5b: Wide LED box in the upper 45% = bowl platform, not display.
    # On bowl scales the display sits in the lower half; a box that is
    # wide (>65% of image) AND centred above the midpoint is the bowl/rim.
    if display_kind == "led" and w_frac > 0.65 and cy_frac < 0.45:
        return True

    # Rule 6: Extremely high aspect + top half = rim/strip
    if aspect > 6.5 and cy_frac < 0.45:
        return True

    # Rule 7: Box covers more than 30% of total image area
    area_frac = (bw * bh) / max(img_w * img_h, 1)
    if area_frac > 0.30:
        return True

    return False


def _is_high_quality_localization(
    box: tuple[int, int, int, int],
    image_size: tuple[int, int],
    confidence: float,
    display_kind: str,
) -> bool:
    """
    Returns True when the localization is trustworthy enough to skip the
    region-refiner pass (saves ~1 full Gemini API call).

    We are CONSERVATIVE about skipping — a second pass is only skipped when
    we are highly confident the box is correct.  In particular, for LED bowl
    scales the primary localizer consistently places the box 10-15% too high
    (onto the rim/base of the bowl rather than the display panel below it).
    We detect this by checking cy_frac: if the box centre is above 55% of the
    image height we force Pass 2 so the refiner can correct the position.
    """
    if confidence < LOCALIZER_SKIP_REFINE_THRESHOLD:
        return False

    x1, y1, x2, y2 = box
    img_w, img_h = image_size
    bw = x2 - x1
    bh = y2 - y1

    w_frac = bw / max(img_w, 1)
    h_frac = bh / max(img_h, 1)
    cy_frac = ((y1 + y2) / 2.0) / max(img_h, 1)
    aspect = bw / max(bh, 1)

    # Good boxes: reasonably sized, good aspect, not edge-to-edge
    if w_frac < 0.12 or w_frac > 0.85:
        return False
    if h_frac < 0.05 or h_frac > 0.22:
        return False
    if aspect < 1.5 or aspect > 8.0:
        return False

    # For LED bowl scales the display is ALWAYS in the lower portion of the
    # image. If the box centre is above 55% we must run Pass 2 — the primary
    # localizer has likely landed on the bowl platform rather than the display.
    if display_kind == "led" and cy_frac < 0.55:
        return False

    return True


def expand_localization_box(
    box: tuple[int, int, int, int],
    image_size: tuple[int, int],
    display_kind: str,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    width, height = image_size
    bw = x2 - x1
    bh = y2 - y1

    if display_kind == "led":
        left_pad = int(round(bw * 0.22))
        right_pad = max(int(round(bw * 0.18)), 12)
        # top_pad = 0: for LED bowl scales the bowl sits directly above the
        # display. Any upward expansion pulls the crop into the bowl rim.
        # The localizer already includes a small top margin in its box.
        top_pad = 0
        bottom_pad = max(int(round(bh * 0.26)), 8)
    elif display_kind == "lcd":
        left_pad = int(round(bw * 0.06))
        right_pad = max(int(round(bw * 0.06)), 6)
        top_pad = int(round(bh * 0.08))
        bottom_pad = max(int(round(bh * 0.08)), 5)
    else:
        left_pad = int(round(bw * 0.14))
        right_pad = max(int(round(bw * 0.16)), 10)
        top_pad = int(round(bh * 0.18))
        bottom_pad = max(int(round(bh * 0.24)), 8)

    return (
        max(0, x1 - left_pad),
        max(0, y1 - top_pad),
        min(width, x2 + right_pad),
        min(height, y2 + bottom_pad),
    )


def make_search_region_box(
    box: tuple[int, int, int, int],
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    width, height = image_size
    bw = x2 - x1
    bh = y2 - y1

    left_pad  = int(round(bw * 1.6))
    right_pad = int(round(bw * 1.6))
    top_pad    = int(round(bh * 4.5))
    bottom_pad = int(round(bh * 4.5))

    left_pad   = max(left_pad,   int(width  * 0.12))
    right_pad  = max(right_pad,  int(width  * 0.12))
    top_pad    = max(top_pad,    int(height * 0.18))
    bottom_pad = max(bottom_pad, int(height * 0.18))

    return (
        max(0, x1 - left_pad),
        max(0, y1 - top_pad),
        min(width,  x2 + right_pad),
        min(height, y2 + bottom_pad),
    )


def map_child_box_to_parent(
    parent_box: tuple[int, int, int, int],
    child_box: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    px1, py1, _, _ = parent_box
    cx1, cy1, cx2, cy2 = child_box
    return (px1 + cx1, py1 + cy1, px1 + cx2, py1 + cy2)


def crop_from_box(img: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    return img.crop(box)


def draw_localization_debug(
    img: Image.Image,
    box: tuple[int, int, int, int],
    display_kind: str,
    source: str,
) -> Image.Image:
    debug = img.copy()
    draw = ImageDraw.Draw(debug)
    color = {
        "led": (0, 220, 120),
        "lcd": (255, 196, 0),
        "unknown": (255, 64, 64),
    }.get(display_kind, (255, 64, 64))
    draw.rectangle(box, outline=color, width=max(2, img.width // 300))
    draw.text((box[0] + 4, max(0, box[1] - 18)), f"{source}:{display_kind}", fill=color)
    return debug


def draw_fallback_debug(img: Image.Image, text: str) -> Image.Image:
    debug = img.copy()
    draw = ImageDraw.Draw(debug)
    draw.text((12, 12), text, fill=(255, 64, 64))
    return debug


# ----------------------------
# Read validation / post-processing
# ----------------------------

def validate_numeric_shape(result: ReadingResult) -> ReadingResult:
    if result.value_text:
        result.value_text = result.value_text.strip().replace(" ", "")

    if result.status == "ok":
        if not result.value_text:
            result.status = "needs_review"
            result.value_number = None
            result.confidence = min(result.confidence, 0.2)
            result.reason = "Empty value_text."
            return result

        if not re.fullmatch(r"\d+(\.\d+)?", result.value_text):
            result.status = "needs_review"
            result.value_number = None
            result.confidence = min(result.confidence, 0.2)
            result.reason = "Display text was not a single valid numeric token."
            return result

        if result.value_text.count(".") > 1:
            result.status = "needs_review"
            result.value_number = None
            result.confidence = min(result.confidence, 0.2)
            result.reason = "Multiple decimal points found."
            return result

        if EXPECTED_DECIMALS:
            try:
                expected = int(EXPECTED_DECIMALS)
                if "." in result.value_text:
                    right = result.value_text.split(".", 1)[1]
                    if len(right) != expected:
                        result.status = "needs_review"
                        result.value_number = None
                        result.confidence = min(result.confidence, 0.25)
                        result.reason = f"Decimal precision did not match expected {expected} places."
                        return result
                elif expected > 0:
                    result.status = "needs_review"
                    result.value_number = None
                    result.confidence = min(result.confidence, 0.25)
                    result.reason = f"Expected {expected} decimal places but none were found."
                    return result
            except ValueError:
                pass

        try:
            result.value_number = float(result.value_text)
        except Exception:
            result.status = "needs_review"
            result.value_number = None
            result.confidence = min(result.confidence, 0.2)
            result.reason = "Could not parse numeric value."
            return result

    return result


def infer_fixed_decimal_text(value_text: str, decimals: int) -> str:
    if decimals <= 0 or "." in value_text or len(value_text) <= decimals:
        return value_text
    return f"{value_text[:-decimals]}.{value_text[-decimals:]}"


def count_numeric_digits(value_text: str) -> int:
    return len(value_text.replace(".", "")) if value_text else 0


def has_probable_missing_leading_digit(
    text: str,
    crop_diagnostics: CropDiagnostics | None = None,
    expected_digit_count: int | None = None,
) -> bool:
    if not text or "." not in text:
        return False
    left, _ = text.split(".", 1)
    digit_count = count_numeric_digits(text)
    if expected_digit_count is not None and expected_digit_count > 0:
        if (
            digit_count == expected_digit_count - 1
            and len(left) == 1
            and crop_diagnostics is not None
            and crop_diagnostics.mode == "led"
            and crop_diagnostics.leading_blank_ratio <= 0.26
            and crop_diagnostics.active_span_ratio >= 0.64
        ):
            return True
    if crop_diagnostics is None or crop_diagnostics.mode != "led":
        return False
    return (
        len(left) == 1
        and digit_count >= 4
        and crop_diagnostics.component_count >= digit_count + 1
        and crop_diagnostics.leading_blank_ratio <= 0.22
        and crop_diagnostics.active_span_ratio >= 0.70
        and crop_diagnostics.green_ratio >= 0.05
    )


def is_suspicious_read(
    result: ReadingResult,
    crop_diagnostics: CropDiagnostics | None = None,
    expected_digit_count: int | None = None,
) -> bool:
    if result.status != "ok":
        return True
    if not result.value_text:
        return True
    text = result.value_text.strip()
    if not re.fullmatch(r"\d+(\.\d+)?", text):
        return True
    try:
        expected = int(EXPECTED_DECIMALS) if EXPECTED_DECIMALS else 0
    except ValueError:
        expected = 0
    if expected > 0:
        if "." not in text:
            return True
        right = text.split(".", 1)[1]
        if len(right) != expected:
            return True
    if text.startswith("88") and "." not in text:
        return True
    if text.startswith("888"):
        return True
    # Catch placeholder-segment bleed-through like 887.530, 88.530, 881.xxx
    # Real scale readings virtually never start with 88 at all (88x implies
    # two placeholder cells were misread as lit 8s)
    if re.match(r"^88\d", text):
        return True
    if expected_digit_count is not None and expected_digit_count > 0:
        if count_numeric_digits(text) != expected_digit_count:
            return True
    if has_probable_missing_leading_digit(text, crop_diagnostics, expected_digit_count):
        return True
    if crop_diagnostics and crop_diagnostics.is_reliable:
        if text.startswith("88"):
            return True
        if text.replace(".", "").startswith("888"):
            return True
    return False


def post_process_result(result: ReadingResult) -> ReadingResult:
    if result.status == "ok" and result.value_text:
        try:
            expected = int(EXPECTED_DECIMALS) if EXPECTED_DECIMALS else 0
        except ValueError:
            expected = 0
        if expected > 0 and "." not in result.value_text and not result.value_text.startswith("88") and not re.match(r"^88\d", result.value_text):
            result.value_text = infer_fixed_decimal_text(result.value_text, expected)
            result.reason = f"{result.reason} Decimal inferred using fixed {expected}-decimal display format."
            result.confidence = min(result.confidence, 0.88)
    return validate_numeric_shape(result)


# ----------------------------
# Reader
# ----------------------------

def call_gemini_with_instructions(
    instructions: str,
    crop_img: Image.Image,
    original_img: Image.Image,
    *,
    model_name: str | None = None,
    primary_source: Literal["crop", "original"] = "crop",
    include_secondary: bool = True,
) -> ReadingResult:
    prompt = load_prompt_text()
    contents: list[object] = [prompt, instructions]

    if primary_source == "crop":
        contents.extend(["PRIMARY image: LOCALIZED display crop.", crop_img])
        if include_secondary:
            contents.extend(["CONTEXT image: ORIGINAL full photo.", original_img])
    else:
        contents.extend(["PRIMARY image: ORIGINAL full photo.", original_img])
        if include_secondary:
            contents.extend(["CONTEXT image: LOCALIZED display crop.", crop_img])

    selected_model = model_name or MODEL_NAME
    try:
        response = client.models.generate_content(
            model=selected_model,
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ReadingResult,
                temperature=READ_TEMPERATURE,
            ),
        )
    except Exception:
        if selected_model != MODEL_NAME:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ReadingResult,
                    temperature=READ_TEMPERATURE,
                ),
            )
        else:
            raise

    if getattr(response, "parsed", None) is not None:
        parsed = response.parsed
        if isinstance(parsed, ReadingResult):
            return post_process_result(parsed)
        if isinstance(parsed, dict):
            return post_process_result(ReadingResult(**parsed))

    raw_text = getattr(response, "text", None)
    if not raw_text:
        raise HTTPException(status_code=502, detail="Gemini returned empty response.")

    try:
        payload = json.loads(raw_text)
        return post_process_result(ReadingResult(**payload))
    except (json.JSONDecodeError, ValidationError) as e:
        raise HTTPException(status_code=502, detail=f"Invalid Gemini JSON response: {e}")


def mark_suspicious_for_review(result: ReadingResult, reason: str) -> ReadingResult:
    result.status = "needs_review"
    result.value_number = None
    result.confidence = min(result.confidence, 0.35)
    result.reason = reason
    return result


def call_gemini_on_full_image(
    original_img: Image.Image,
    fallback_reason: str,
    *,
    display_kind: str = "unknown",
) -> ReadingResult:
    kind_hint = f"The device likely uses a {display_kind.upper()} display. " if display_kind in {"led", "lcd"} else ""
    primary_result = call_gemini_with_instructions(
        (
            "No trustworthy localized ROI is available. "
            "Read the value directly from the ORIGINAL full photo only. "
            "Ignore any imagined crop or display box. "
            "Look carefully for a tiny isolated decimal dot near the numeric row baseline, especially near the right side of the display. "
            "CRITICAL — DIGIT 1: The digit 1 uses ONLY the two right-side vertical segments of a slot. "
            "It is narrow and bright green, not a dim full-width 8 placeholder. "
            "A narrow bright column between placeholders and the main number IS the digit 1 — do NOT skip it. "
            f"{kind_hint}"
            f"Localization fallback reason: {fallback_reason}"
        ),
        original_img,
        original_img,
        model_name=MODEL_NAME,
        primary_source="original",
        include_secondary=False,
    )
    if not is_suspicious_read(primary_result):
        return primary_result
    return mark_suspicious_for_review(
        primary_result, "Single-pass full-image fallback remained suspicious. Marked for review."
    )


@app.get("/")
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "model": MODEL_NAME,
        "localizer_model": LOCALIZER_MODEL_NAME,
        "localizer_min_confidence": LOCALIZER_MIN_CONFIDENCE,
        "localizer_skip_refine_threshold": LOCALIZER_SKIP_REFINE_THRESHOLD,
        "localizer_max_dimension": LOCALIZER_MAX_DIMENSION,
        "local_decoder_min_confidence": LOCAL_DECODER_MIN_CONFIDENCE,
        "expected_decimals": EXPECTED_DECIMALS or None,
    }


@app.get("/api/preview/{kind}")
def api_preview(kind: str):
    if kind not in ("original", "enhanced", "crop", "debug"):
        raise HTTPException(status_code=404, detail="Preview not found.")
    img_bytes = _LAST_PREVIEWS.get(kind)
    if img_bytes is None:
        raise HTTPException(status_code=404, detail="No preview available yet.")
    return Response(content=img_bytes, media_type="image/jpeg")


@app.post("/api/read-scale")
async def read_scale(file: UploadFile = File(...)) -> JSONResponse:
    t_start = time.monotonic()
    data = await file.read()
    validate_upload(file, data)

    original_img = open_image(data)

    # ------------------------------------------------------------------ #
    # Enhancement: do it at localizer resolution directly — avoids        #
    # enhancing a large image then shrinking it (bilateral filter is slow) #
    # ------------------------------------------------------------------ #
    localizer_img = resize_keep_aspect(original_img, max_dim=LOCALIZER_MAX_DIMENSION)
    localizer_enhanced_img = make_enhanced_display_image(localizer_img, max_dim=LOCALIZER_MAX_DIMENSION)
    # Full-res enhanced still needed for the reader context image
    enhanced_full_img = make_enhanced_display_image(original_img, max_dim=MAX_DIMENSION)

    crop_img: Image.Image = make_placeholder_preview((720, 180), "No localized ROI")
    debug_img: Image.Image = original_img.copy()
    result: ReadingResult
    localization_payload: dict[str, object]
    skipped_refine = False

    # ------------------------------------------------------------------ #
    # Pass 1: primary localization                                         #
    # ------------------------------------------------------------------ #
    best_box_pixels: tuple[int, int, int, int] | None = None
    best_localization: LocalizationResult | None = None
    best_confidence: float = 0.0
    primary_localization: LocalizationResult

    try:
        primary_localization = call_gemini_localizer(
            localizer_img,
            localizer_enhanced_img,
            previous=None,
        )

        if primary_localization.found and primary_localization.confidence >= LOCALIZER_MIN_CONFIDENCE:
            raw_box = localization_box_to_pixels(primary_localization, original_img.size)
            if raw_box and is_valid_localization_box(raw_box, original_img.size):
                if not _is_bad_localization_box(raw_box, original_img.size, primary_localization.display_kind):
                    expanded = expand_localization_box(raw_box, original_img.size, primary_localization.display_kind)
                    best_box_pixels = expanded
                    best_localization = primary_localization
                    best_confidence = primary_localization.confidence

        # ------------------------------------------------------------------ #
        # Pass 2: region-refiner — SKIP if pass 1 is already high quality     #
        # For LED boxes in the upper image half we ALWAYS run pass 2 because  #
        # the primary localizer consistently lands on the bowl, not the display#
        # ------------------------------------------------------------------ #
        if best_box_pixels is not None:
            # Compute cy_frac of the current best box for the position hint
            _bcy = ((best_box_pixels[1] + best_box_pixels[3]) / 2.0) / max(original_img.size[1], 1)

            if _is_high_quality_localization(
                best_box_pixels, original_img.size, best_confidence, primary_localization.display_kind
            ):
                skipped_refine = True
            else:
                search_box = make_search_region_box(best_box_pixels, original_img.size)
                search_original = crop_from_box(original_img, search_box)
                search_enhanced = crop_from_box(enhanced_full_img, search_box)

                refined = call_gemini_region_localizer(
                    search_original,
                    search_enhanced,
                    display_kind_hint=primary_localization.display_kind,
                    primary_cy_frac=_bcy,
                )
                if refined.found and refined.confidence >= LOCALIZER_MIN_CONFIDENCE:
                    refined_local_box = localization_box_to_pixels(refined, search_original.size)
                    if refined_local_box and is_valid_localization_box(refined_local_box, search_original.size):
                        mapped = map_child_box_to_parent(search_box, refined_local_box)
                        if not _is_bad_localization_box(mapped, original_img.size, refined.display_kind):
                            expanded_mapped = expand_localization_box(mapped, original_img.size, refined.display_kind)
                            if refined.confidence >= best_confidence:
                                best_box_pixels = expanded_mapped
                                best_localization = refined
                                best_confidence = refined.confidence

    except Exception as e:
        primary_localization = LocalizationResult(
            found=False, confidence=0.0, display_kind="unknown",
            reason=f"Localization failed: {e}"
        )
        best_localization = primary_localization

    # ------------------------------------------------------------------ #
    # Build crop and debug image from best box (if any)                   #
    # ------------------------------------------------------------------ #
    display_kind = best_localization.display_kind if best_localization else "unknown"

    if best_box_pixels is not None and best_localization is not None:
        crop_img = crop_from_box(original_img, best_box_pixels)
        debug_img = draw_localization_debug(
            original_img, best_box_pixels, display_kind, best_localization.reason[:40]
        )
        _box_cy = round(((best_box_pixels[1] + best_box_pixels[3]) / 2.0) / max(original_img.size[1], 1), 3)
        localization_payload = {
            "source": "vlm_pass1" if skipped_refine else "vlm_pass2",
            "model": LOCALIZER_MODEL_NAME,
            "found": True,
            "confidence": round(best_confidence, 3),
            "display_kind": display_kind,
            "reason": best_localization.reason,
            "skipped_refine": skipped_refine,
            "box_cy_frac": _box_cy,
            "box_norm_1000": pixels_to_norm1000(best_box_pixels, original_img.size),
            "box_pixels": {
                "x1": best_box_pixels[0], "y1": best_box_pixels[1],
                "x2": best_box_pixels[2], "y2": best_box_pixels[3],
            },
        }
    else:
        debug_img = draw_fallback_debug(original_img, "localization failed — full image read")
        localization_payload = {
            "source": "full_image_fallback",
            "model": LOCALIZER_MODEL_NAME,
            "found": False,
            "confidence": 0.0,
            "display_kind": "unknown",
            "reason": best_localization.reason if best_localization else "Localization not attempted",
            "skipped_refine": False,
        }

    # ------------------------------------------------------------------ #
    # Read: always send both crop and full image.                          #
    # ------------------------------------------------------------------ #
    crop_diagnostics = CropDiagnostics(
        is_reliable=False,
        mode=display_kind if display_kind in {"led", "lcd"} else "unknown",
        quality_score=0.0, lit_ratio=0.0, green_ratio=0.0, component_count=0,
        active_span_ratio=0.0, active_band_height_ratio=0.0, leading_blank_ratio=0.0,
        reason="Crop diagnostics skipped — reader uses full-image fallback.",
    )

    if best_box_pixels is not None:
        kind_hint = f"Display type: {display_kind.upper()}. " if display_kind in {"led", "lcd"} else ""
        result = call_gemini_with_instructions(
            (
                f"{kind_hint}"
                "You have TWO images: PRIMARY = localized display crop, CONTEXT = full original photo.\n"
                "\n"
                "STEP 1 — ASSESS THE PRIMARY CROP:\n"
                "Does the PRIMARY crop show a rectangular display window containing visible numeric digit segments? "
                "A valid crop shows: (a) bright green/orange glowing segments on a dark background (LED), "
                "or (b) dark numeric digit segments on a light/gray background (LCD).\n"
                "\n"
                "STEP 2 — CHOOSE YOUR SOURCE:\n"
                "  • VALID CROP → read from the crop (it is the authoritative source).\n"
                "  • INVALID CROP → DISCARD IT ENTIRELY and read from the CONTEXT photo instead.\n"
                "    An INVALID crop is any of:\n"
                "    - A blank, featureless, or nearly uniform area (no digit shapes visible)\n"
                "    - A weighing bowl, pan, dish, or metal rim/ring\n"
                "    - Physical control buttons (round colored buttons)\n"
                "    - Machine casing, body, or label area without an actual screen\n"
                "    - A blurry region with no identifiable digit structure\n"
                "\n"
                "STEP 3 — READ THE VALUE:\n"
                "  - LED: count ONLY bright illuminated segments. "
                "Dim gray 8-shaped outlines filling an entire slot are INACTIVE PLACEHOLDERS — ignore them.\n"
                "  - CRITICAL — DIGIT 1: The digit 1 uses ONLY the two right-side vertical bars of a slot. "
                "It is narrow and bright green — NOT a dim full-width 8 placeholder. "
                "Do NOT skip it. A narrow bright column to the left of the main number IS the digit 1. "
                "Example: if display shows [dim][dim][narrow-bright][8].[5][4][0] → read 18.540, NOT 8.540.\n"
                "  - LCD: read dark segments on the light screen. "
                "Do not include buttons, labels, or areas outside the screen rectangle.\n"
                "  - Include decimal point only if a small isolated dot is clearly visible.\n"
                "  - Return exactly one numeric token: digits and at most one decimal point."
            ),
            crop_img,
            original_img,
            model_name=MODEL_NAME,
            primary_source="crop",
            include_secondary=True,
        )
    else:
        result = call_gemini_on_full_image(
            original_img,
            localization_payload.get("reason", "localization failed"),
            display_kind=display_kind,
        )

    if is_suspicious_read(result):
        result = mark_suspicious_for_review(result, f"Read flagged as suspicious: {result.reason}")

    t_elapsed = round(time.monotonic() - t_start, 2)

    _LAST_PREVIEWS["original"] = pil_to_jpeg_bytes(original_img)
    _LAST_PREVIEWS["enhanced"] = pil_to_jpeg_bytes(enhanced_full_img)
    _LAST_PREVIEWS["crop"] = pil_to_jpeg_bytes(crop_img)
    _LAST_PREVIEWS["debug"] = pil_to_jpeg_bytes(debug_img)

    return JSONResponse(
        content={
            "final": result.model_dump(),
            "localization": localization_payload,
            "crop_diagnostics": {
                "is_reliable": crop_diagnostics.is_reliable,
                "mode": crop_diagnostics.mode,
                "quality_score": crop_diagnostics.quality_score,
                "lit_ratio": crop_diagnostics.lit_ratio,
                "green_ratio": crop_diagnostics.green_ratio,
                "component_count": crop_diagnostics.component_count,
                "active_span_ratio": crop_diagnostics.active_span_ratio,
                "active_band_height_ratio": crop_diagnostics.active_band_height_ratio,
                "leading_blank_ratio": crop_diagnostics.leading_blank_ratio,
                "reason": crop_diagnostics.reason,
            },
            "elapsed_seconds": t_elapsed,
            "preview_urls": {
                "original": "/api/preview/original",
                "enhanced": "/api/preview/enhanced",
                "crop": "/api/preview/crop",
                "debug": "/api/preview/debug",
            },
        }
    )


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host=host, port=port, reload=True)