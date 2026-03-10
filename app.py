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
from PIL import Image
from pydantic import BaseModel, Field, ValidationError

from image_enhance import make_enhanced_display_image
from vision import (
    CropDiagnostics,
    LocalDecodeResult,
    analyze_crop_diagnostics,
    decode_display_crop,
    locate_display_crop,
)

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
PROMPT_PATH = BASE_DIR / "prompts" / "scale_reader.txt"
STATIC_DIR = BASE_DIR / "static"

MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-flash")
VERIFY_MODEL_NAME = os.getenv("VERIFY_MODEL_NAME", "gemini-2.5-pro")
MAX_IMAGE_MB = int(os.getenv("MAX_IMAGE_MB", "10"))
MAX_IMAGE_BYTES = MAX_IMAGE_MB * 1024 * 1024
MAX_DIMENSION = int(os.getenv("MAX_DIMENSION", "1200"))
EXPECTED_DECIMALS = (os.getenv("EXPECTED_DECIMALS") or os.getenv("FIXED_DECIMALS") or "").strip()
LOCAL_DECODER_MIN_CONFIDENCE = float(os.getenv("LOCAL_DECODER_MIN_CONFIDENCE", "0.90"))

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
    if crop_diagnostics.is_reliable:
        use_local_decoder = crop_diagnostics.mode == "led"
        local_result = (
            decode_display_crop(crop_img)
            if use_local_decoder
            else LocalDecodeResult(
                ok=False,
                value_text=None,
                confidence=0.0,
                digit_count=0,
                decimal_count=0,
                reason="local seven-segment decoder disabled for non-LED crop",
            )
        )
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

    primary_result = call_gemini_with_instructions(
        (
            "Localized crop quality is LOW or partial. "
            "Use the ORIGINAL full photo as the main source and use the crop only as weak context. "
            "If the crop looks like bezel, machine body, or a blue strip, ignore it. "
            f"Crop diagnostics: score={crop_diagnostics.quality_score}, reason={crop_diagnostics.reason}."
        ),
        crop_img,
        original_img,
        model_name=MODEL_NAME,
        primary_source="original",
        include_secondary=True,
    )

    if not is_suspicious_read(primary_result, crop_diagnostics):
        return primary_result

    verification_result = call_gemini_with_instructions(
        (
            "Second pass: the localized crop is unreliable, so ignore it unless it clearly shows the display window. "
            "Read the value from the ORIGINAL full photo only. "
            "Return needs_review if the display cannot be cleanly read. "
            f"Previous answer was {primary_result.value_text!r}. "
            f"Crop diagnostics: score={crop_diagnostics.quality_score}, reason={crop_diagnostics.reason}."
        ),
        crop_img,
        original_img,
        model_name=VERIFY_MODEL_NAME,
        primary_source="original",
        include_secondary=False,
    )
    chosen = choose_result(primary_result, verification_result, crop_diagnostics)
    if is_suspicious_read(chosen, crop_diagnostics):
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
    crop_img, debug_img = locate_display_crop(original_img, max_dim=MAX_DIMENSION)
    crop_diagnostics = analyze_crop_diagnostics(crop_img)

    _LAST_PREVIEWS["original"] = pil_to_jpeg_bytes(original_img)
    _LAST_PREVIEWS["enhanced"] = pil_to_jpeg_bytes(enhanced_full_img)
    _LAST_PREVIEWS["crop"] = pil_to_jpeg_bytes(crop_img)
    _LAST_PREVIEWS["debug"] = pil_to_jpeg_bytes(debug_img)

    result = call_gemini_on_crop(crop_img, original_img, crop_diagnostics)

    return JSONResponse(
        content={
            "final": result.model_dump(),
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
