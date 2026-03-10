from __future__ import annotations

import io
import json
import os
import re
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
    locate_display_crop,
    resize_keep_aspect,
)

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
PROMPT_PATH = BASE_DIR / "prompts" / "scale_reader.txt"
STATIC_DIR = BASE_DIR / "static"

MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-flash")
VERIFY_MODEL_NAME = os.getenv("VERIFY_MODEL_NAME", "gemini-2.5-pro")
LOCALIZER_MODEL_NAME = os.getenv("LOCALIZER_MODEL_NAME", "gemini-2.5-flash-lite")
MAX_IMAGE_MB = int(os.getenv("MAX_IMAGE_MB", "10"))
MAX_IMAGE_BYTES = MAX_IMAGE_MB * 1024 * 1024
MAX_DIMENSION = int(os.getenv("MAX_DIMENSION", "1200"))
LOCALIZER_MAX_DIMENSION = int(os.getenv("LOCALIZER_MAX_DIMENSION", "1024"))
EXPECTED_DECIMALS = (os.getenv("EXPECTED_DECIMALS") or os.getenv("FIXED_DECIMALS") or "").strip()
LOCAL_DECODER_MIN_CONFIDENCE = float(os.getenv("LOCAL_DECODER_MIN_CONFIDENCE", "0.90"))
LOCALIZER_MIN_CONFIDENCE = float(os.getenv("LOCALIZER_MIN_CONFIDENCE", "0.45"))

if not os.getenv("GEMINI_API_KEY"):
    raise RuntimeError("Missing GEMINI_API_KEY in environment.")

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI(title="Gemini Scale Reader", version="6.0.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

_LAST_PREVIEWS: dict[str, bytes] = {}


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


def load_prompt() -> str:
    if not PROMPT_PATH.exists():
        raise RuntimeError(f"Prompt file not found: {PROMPT_PATH}")
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


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


def call_gemini_localizer(localizer_img: Image.Image) -> LocalizationResult:
    instructions = (
        "Locate the numeric display window only. "
        "Return one bounding box tightly enclosing the full display screen/window that contains the numeric reading. "
        "Do not return the machine body, bowl, buttons, blue base, watermark/date text, brand text, or reflections. "
        "For LED displays, include the full dark display window including faint inactive left slots. "
        "For LCD displays, include the full rectangular LCD screen. "
        "Coordinates must be integers normalized from 0 to 1000 relative to the full input image. "
        "If no display window is visible, set found=false. "
        "display_kind must be one of: led, lcd, unknown. "
        "confidence is localization confidence only."
    )

    selected_model = LOCALIZER_MODEL_NAME
    try:
        response = client.models.generate_content(
            model=selected_model,
            contents=[instructions, localizer_img],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=LocalizationResult,
                temperature=0.0,
            ),
        )
    except Exception:
        if selected_model != MODEL_NAME:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[instructions, localizer_img],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=LocalizationResult,
                    temperature=0.0,
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
    if area_ratio < 0.0008 or area_ratio > 0.28:
        return False
    if aspect < 1.0 or aspect > 9.5:
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
        right_pad = int(round(bw * 0.12))
        top_pad = int(round(bh * 0.20))
        bottom_pad = int(round(bh * 0.20))
    elif display_kind == "lcd":
        left_pad = int(round(bw * 0.10))
        right_pad = int(round(bw * 0.10))
        top_pad = int(round(bh * 0.16))
        bottom_pad = int(round(bh * 0.16))
    else:
        left_pad = int(round(bw * 0.14))
        right_pad = int(round(bw * 0.12))
        top_pad = int(round(bh * 0.18))
        bottom_pad = int(round(bh * 0.18))

    return (
        max(0, x1 - left_pad),
        max(0, y1 - top_pad),
        min(width, x2 + right_pad),
        min(height, y2 + bottom_pad),
    )


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


def apply_localization_mode_hint(
    crop_diagnostics: CropDiagnostics,
    display_kind: str,
    confidence: float,
) -> CropDiagnostics:
    if display_kind not in {"led", "lcd"}:
        return crop_diagnostics

    is_reliable = crop_diagnostics.is_reliable
    reason = crop_diagnostics.reason

    if confidence >= 0.75 and crop_diagnostics.quality_score >= 0.30:
        is_reliable = True
        reason = f"{reason}; promoted by high-confidence VLM localization"

    return CropDiagnostics(
        is_reliable=is_reliable,
        mode=display_kind,
        quality_score=crop_diagnostics.quality_score,
        lit_ratio=crop_diagnostics.lit_ratio,
        green_ratio=crop_diagnostics.green_ratio,
        component_count=crop_diagnostics.component_count,
        active_span_ratio=crop_diagnostics.active_span_ratio,
        active_band_height_ratio=crop_diagnostics.active_band_height_ratio,
        leading_blank_ratio=crop_diagnostics.leading_blank_ratio,
        reason=reason,
    )


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
                        result.reason = (
                            f"Decimal precision did not match expected {expected} places."
                        )
                        return result
                elif expected > 0:
                    result.status = "needs_review"
                    result.value_number = None
                    result.confidence = min(result.confidence, 0.25)
                    result.reason = (
                        f"Expected {expected} decimal places but none were found."
                    )
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

        if expected > 0 and "." not in result.value_text and not result.value_text.startswith("88"):
            result.value_text = infer_fixed_decimal_text(result.value_text, expected)
            result.reason = (
                f"{result.reason} Decimal inferred using fixed {expected}-decimal display format."
            )
            result.confidence = min(result.confidence, 0.88)

    return validate_numeric_shape(result)


def local_result_to_reading_result(local_result: LocalDecodeResult) -> ReadingResult:
    if not local_result.ok or not local_result.value_text:
        return ReadingResult(
            status="needs_review",
            value_text=None,
            value_number=None,
            confidence=max(0.0, min(local_result.confidence, 1.0)),
            reason=local_result.reason,
            ignored_text_present=True,
        )

    return post_process_result(
        ReadingResult(
            status="ok",
            value_text=local_result.value_text,
            value_number=None,
            confidence=max(0.0, min(local_result.confidence, 1.0)),
            reason=local_result.reason,
            ignored_text_present=True,
        )
    )


def call_gemini_with_instructions(
    instructions: str,
    crop_img: Image.Image, 
    original_img: Image.Image,
    *,
    model_name: str | None = None,
    primary_source: Literal["crop", "original"] = "crop",
    include_secondary: bool = True,
) -> ReadingResult:
    prompt = load_prompt()
    contents: list[object] = [prompt, instructions]

    if primary_source == "crop":
        contents.extend(
            [
                "PRIMARY image: LOCALIZED display crop.",
                crop_img,
            ]
        )
        if include_secondary:
            contents.extend(
                [
                    "CONTEXT image: ORIGINAL full photo.",
                    original_img,
                ]
            )
    else:
        contents.extend(
            [
                "PRIMARY image: ORIGINAL full photo.",
                original_img,
            ]
        )
        if include_secondary:
            contents.extend(
                [
                    "CONTEXT image: LOCALIZED display crop.",
                    crop_img,
                ]
            )

    selected_model = model_name or MODEL_NAME
    try:
        response = client.models.generate_content(
            model=selected_model,
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ReadingResult,
                temperature=0.0,
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
                    temperature=0.0,
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
        result = ReadingResult(**payload)
        return post_process_result(result)
    except (json.JSONDecodeError, ValidationError) as e:
        raise HTTPException(status_code=502, detail=f"Invalid Gemini JSON response: {e}")


def choose_result(
    primary_result: ReadingResult,
    secondary_result: ReadingResult,
    crop_diagnostics: CropDiagnostics,
    expected_digit_count: int | None = None,
) -> ReadingResult:
    primary_suspicious = is_suspicious_read(primary_result, crop_diagnostics, expected_digit_count)
    secondary_suspicious = is_suspicious_read(secondary_result, crop_diagnostics, expected_digit_count)

    if primary_suspicious and not secondary_suspicious:
        return secondary_result
    if secondary_suspicious and not primary_suspicious:
        return primary_result

    if primary_result.status == "ok" and secondary_result.status != "ok":
        return primary_result
    if secondary_result.status == "ok" and primary_result.status != "ok":
        return secondary_result

    if secondary_result.confidence > primary_result.confidence + 0.12:
        return secondary_result
    return primary_result


def mark_suspicious_for_review(
    result: ReadingResult,
    reason: str,
) -> ReadingResult:
    result.status = "needs_review"
    result.value_number = None
    result.confidence = min(result.confidence, 0.35)
    result.reason = reason
    return result


def call_gemini_on_crop(
    crop_img: Image.Image,
    original_img: Image.Image,
    crop_diagnostics: CropDiagnostics,
) -> ReadingResult:
    led_local_result = (
        decode_display_crop(crop_img)
        if crop_diagnostics.mode == "led"
        else LocalDecodeResult(
            ok=False,
            value_text=None,
            confidence=0.0,
            digit_count=0,
            decimal_count=0,
            reason="local seven-segment decoder disabled for non-LED crop",
        )
    )

    if crop_diagnostics.is_reliable:
        use_local_decoder = crop_diagnostics.mode == "led"
        local_result = led_local_result
        local_reading = local_result_to_reading_result(local_result)
        expected_digit_count = local_result.digit_count if use_local_decoder and local_result.digit_count > 0 else None
        local_hint = f"Crop mode is {crop_diagnostics.mode}. "
        if use_local_decoder:
            local_hint += (
                f"The crop contains about {local_result.digit_count} active digit slots and "
                f"{local_result.decimal_count} visible decimal dots. "
            )
        local_value_hint = ""
        if use_local_decoder and local_result.ok and local_result.confidence >= LOCAL_DECODER_MIN_CONFIDENCE:
            local_value_hint = f"The deterministic decoder proposes {local_result.value_text!r}. "
        if use_local_decoder and local_result.ok and local_result.confidence >= LOCAL_DECODER_MIN_CONFIDENCE:
            return local_reading

        primary_result = call_gemini_with_instructions(
            (
                "Localized crop quality is HIGH and contains a coherent display row. "
                "Use the crop as the authoritative source. "
                "Count only bright illuminated segments as real digits. "
                "Dim gray placeholder 8-shapes on the left are inactive slots and must be ignored. "
                "Do not let the full photo override a clean crop. "
                "Read every active digit slot; do not drop a visible middle digit such as 18.730 into 18.30. "
                f"{local_hint}{local_value_hint}"
                f"Crop diagnostics: score={crop_diagnostics.quality_score}, reason={crop_diagnostics.reason}."
            ),
            crop_img,
            original_img,
            model_name=MODEL_NAME,
            primary_source="crop",
            include_secondary=False,
        )

        if use_local_decoder and local_result.ok and local_result.confidence >= LOCAL_DECODER_MIN_CONFIDENCE and not is_suspicious_read(local_reading, crop_diagnostics, expected_digit_count):
            if is_suspicious_read(primary_result, crop_diagnostics, expected_digit_count):
                return local_reading
            if local_reading.value_text == primary_result.value_text:
                primary_result.confidence = max(primary_result.confidence, local_reading.confidence)
                primary_result.reason = f"{primary_result.reason} Confirmed by deterministic seven-segment decoder."
                return primary_result
            if local_reading.confidence >= primary_result.confidence + 0.08:
                return local_reading

        if not is_suspicious_read(primary_result, crop_diagnostics, expected_digit_count):
            return primary_result

        verification_result = call_gemini_with_instructions(
            (
                "The crop-first answer looked suspicious. "
                "Re-read using the ORIGINAL full photo as primary and the crop as confirmation. "
                "If the crop shows dim gray placeholder 8-shapes before bright green digits, ignore the gray placeholders. "
                "Do not convert a visible display like 18.670 into 88.600 or 88830. "
                "Do not drop the middle digit from a five-digit read like 18.730 into 18.30. "
                f"{local_hint}{local_value_hint}"
                f"Previous answer was {primary_result.value_text!r}. "
                f"Crop diagnostics: score={crop_diagnostics.quality_score}, reason={crop_diagnostics.reason}."
            ),
            crop_img,
            original_img,
            model_name=VERIFY_MODEL_NAME,
            primary_source="original",
            include_secondary=True,
        )
        chosen = choose_result(primary_result, verification_result, crop_diagnostics, expected_digit_count)
        if use_local_decoder and local_result.ok and local_result.confidence >= LOCAL_DECODER_MIN_CONFIDENCE and not is_suspicious_read(local_reading, crop_diagnostics, expected_digit_count):
            if is_suspicious_read(chosen, crop_diagnostics, expected_digit_count):
                return local_reading
            if local_reading.confidence >= chosen.confidence + 0.08:
                return local_reading
            if local_reading.value_text == chosen.value_text:
                chosen.confidence = max(chosen.confidence, local_reading.confidence)
                chosen.reason = f"{chosen.reason} Confirmed by deterministic seven-segment decoder."
        if is_suspicious_read(chosen, crop_diagnostics, expected_digit_count):
            return mark_suspicious_for_review(
                chosen,
                "Both crop-first and full-photo verification remained suspicious. Marked for review.",
            )
        if chosen is verification_result and chosen.status == "ok":
            chosen.reason = f"{chosen.reason} Replaced suspicious crop-first read using full-photo verification."
        return chosen

    weak_crop_hint = (
        f"The weak crop still suggests about {led_local_result.digit_count} active digit slots and "
        f"{led_local_result.decimal_count} visible decimal dots. "
        if crop_diagnostics.mode == "led" and led_local_result.digit_count > 0
        else ""
    )

    primary_result = call_gemini_with_instructions(
        (
            "Localized crop quality is LOW or partial. "
            "Use the ORIGINAL full photo as the only source of truth for this pass. "
            "Ignore the localized crop unless a later verification step explicitly asks for it. "
            "If the crop looks like bezel, machine body, or a blue strip, ignore it completely. "
            "Do not drop a faint but aligned leading digit on the left edge of the display row. "
            "If the full display row is `18540` with a visible decimal after the second digit, return `18.540`, not `8.540`. "
            f"{weak_crop_hint}"
            f"Crop diagnostics: score={crop_diagnostics.quality_score}, reason={crop_diagnostics.reason}."
        ),
        crop_img,
        original_img,
        model_name=MODEL_NAME,
        primary_source="original",
        include_secondary=False,
    )

    expected_digit_count = (
        led_local_result.digit_count
        if crop_diagnostics.mode == "led" and led_local_result.digit_count > 0
        else None
    )

    if not is_suspicious_read(primary_result, crop_diagnostics, expected_digit_count):
        return primary_result

    verification_result = call_gemini_with_instructions(
        (
            "Second pass: the localized crop is unreliable, so ignore it unless it clearly shows the display window. "
            "Read the value from the ORIGINAL full photo only. "
            "Re-check for a faint but real leading `1` on the left of the bright digits. "
            "If the display row is visibly `18.540`, do not return `8.540`. "
            "Return needs_review if the display cannot be cleanly read. "
            f"Previous answer was {primary_result.value_text!r}. "
            f"{weak_crop_hint}"
            f"Crop diagnostics: score={crop_diagnostics.quality_score}, reason={crop_diagnostics.reason}."
        ),
        crop_img,
        original_img,
        model_name=VERIFY_MODEL_NAME,
        primary_source="original",
        include_secondary=False,
    )
    chosen = choose_result(primary_result, verification_result, crop_diagnostics, expected_digit_count)
    if is_suspicious_read(chosen, crop_diagnostics, expected_digit_count):
        return mark_suspicious_for_review(
            chosen,
            "Both context-first passes remained suspicious after rejecting the weak crop. Marked for review.",
        )
    if chosen is verification_result and chosen.status == "ok":
        chosen.reason = f"{chosen.reason} Used context-first fallback because crop was unreliable."
    return chosen


@app.get("/")
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "model": MODEL_NAME,
        "verify_model": VERIFY_MODEL_NAME,
        "localizer_model": LOCALIZER_MODEL_NAME,
        "localizer_min_confidence": LOCALIZER_MIN_CONFIDENCE,
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
    data = await file.read()
    validate_upload(file, data)

    original_img = open_image(data)
    enhanced_full_img = make_enhanced_display_image(original_img, max_dim=MAX_DIMENSION)
    localizer_img = resize_keep_aspect(original_img, max_dim=LOCALIZER_MAX_DIMENSION)
    localization_payload: dict[str, object] = {
        "source": "cv_fallback",
        "model": LOCALIZER_MODEL_NAME,
        "found": False,
        "confidence": 0.0,
        "display_kind": "unknown",
        "reason": "Gemini localizer was not used.",
    }

    crop_img: Image.Image
    debug_img: Image.Image
    try:
        localization = call_gemini_localizer(localizer_img)
        raw_box = localization_box_to_pixels(localization, original_img.size)
        if (
            localization.confidence >= LOCALIZER_MIN_CONFIDENCE
            and is_valid_localization_box(raw_box, original_img.size)
        ):
            expanded_box = expand_localization_box(
                raw_box,
                original_img.size,
                localization.display_kind,
            )
            crop_img = crop_from_box(original_img, expanded_box)
            debug_img = draw_localization_debug(
                original_img,
                expanded_box,
                localization.display_kind,
                "vlm",
            )
            localization_payload = {
                "source": "vlm",
                "model": LOCALIZER_MODEL_NAME,
                "found": localization.found,
                "confidence": round(localization.confidence, 3),
                "display_kind": localization.display_kind,
                "reason": localization.reason,
                "box_norm_1000": {
                    "x1": localization.x1,
                    "y1": localization.y1,
                    "x2": localization.x2,
                    "y2": localization.y2,
                },
                "box_pixels": {
                    "x1": expanded_box[0],
                    "y1": expanded_box[1],
                    "x2": expanded_box[2],
                    "y2": expanded_box[3],
                },
            }
        else:
            crop_img, debug_img = locate_display_crop(original_img, max_dim=MAX_DIMENSION)
            localization_payload = {
                "source": "cv_fallback",
                "model": LOCALIZER_MODEL_NAME,
                "found": localization.found,
                "confidence": round(localization.confidence, 3),
                "display_kind": localization.display_kind,
                "reason": (
                    "Gemini localization was missing or below threshold. "
                    f"Localizer reason: {localization.reason}"
                ),
            }
    except Exception as e:
        crop_img, debug_img = locate_display_crop(original_img, max_dim=MAX_DIMENSION)
        localization_payload = {
            "source": "cv_fallback",
            "model": LOCALIZER_MODEL_NAME,
            "found": False,
            "confidence": 0.0,
            "display_kind": "unknown",
            "reason": f"Gemini localization failed: {e}",
        }

    crop_diagnostics = analyze_crop_diagnostics(crop_img)
    crop_diagnostics = apply_localization_mode_hint(
        crop_diagnostics,
        str(localization_payload.get("display_kind", "unknown")),
        float(localization_payload.get("confidence", 0.0)),
    )

    _LAST_PREVIEWS["original"] = pil_to_jpeg_bytes(original_img)
    _LAST_PREVIEWS["enhanced"] = pil_to_jpeg_bytes(enhanced_full_img)
    _LAST_PREVIEWS["crop"] = pil_to_jpeg_bytes(crop_img)
    _LAST_PREVIEWS["debug"] = pil_to_jpeg_bytes(debug_img)

    result = call_gemini_on_crop(crop_img, original_img, crop_diagnostics)

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
