from __future__ import annotations

import io
import json
import os
import re
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types
from PIL import Image
from pydantic import BaseModel, Field, ValidationError

from image_enhance import make_enhanced_display_image

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
PROMPT_PATH = BASE_DIR / "prompts" / "scale_reader.txt"
STATIC_DIR = BASE_DIR / "static"

MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-flash")
MAX_IMAGE_MB = int(os.getenv("MAX_IMAGE_MB", "10"))
MAX_IMAGE_BYTES = MAX_IMAGE_MB * 1024 * 1024
MAX_DIMENSION = int(os.getenv("MAX_DIMENSION", "1600"))
EXPECTED_DECIMALS = (os.getenv("EXPECTED_DECIMALS") or "").strip()

if not os.getenv("GEMINI_API_KEY"):
    raise RuntimeError("Missing GEMINI_API_KEY in environment.")

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI(title="Gemini Scale Reader", version="4.0.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


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
            detail=f"Image too large. Max allowed size is {MAX_IMAGE_MB} MB."
        )


def open_image(data: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(data))
        return img.convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")


def pil_to_base64_data_url(img: Image.Image, fmt: str = "JPEG", quality: int = 92) -> str:
    """
    Convert PIL image to base64 data URL for easy frontend preview.
    """
    import base64

    buffer = io.BytesIO()
    img.save(buffer, format=fmt, quality=quality)
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    mime = "image/jpeg" if fmt.upper() == "JPEG" else "image/png"
    return f"data:{mime};base64,{b64}"


def validate_numeric_shape(result: ReadingResult) -> ReadingResult:
    """
    Validate only numeric token shape, not numeric range.
    """
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
                else:
                    if expected > 0:
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


def apply_simple_sanity_guard(result: ReadingResult) -> ReadingResult:
    """
    Guard against current dominant failure mode:
    ghost left slots amplified into fake leading 8s.
    """
    if result.status != "ok" or not result.value_text:
        return result

    txt = result.value_text

    # Very suspicious hallucination pattern seen in current failures.
    if txt.startswith("888") or txt.startswith("88"):
        result.status = "needs_review"
        result.value_number = None
        result.confidence = min(result.confidence, 0.25)
        result.reason = "Suspicious leading 8 pattern likely caused by ghost segment amplification."
        return result

    return result


def post_process_result(result: ReadingResult) -> ReadingResult:
    result = validate_numeric_shape(result)
    result = apply_simple_sanity_guard(result)
    return result


def call_gemini(original_img: Image.Image, enhanced_img: Image.Image) -> ReadingResult:
    prompt = load_prompt()

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            prompt,
            "Image 1: ORIGINAL photo. Use this as the primary source of truth for whether segments are truly lit.",
            original_img,
            "Image 2: ENHANCED version of the same photo. Use this only as a secondary aid for readability, not for inventing new digits.",
            enhanced_img,
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ReadingResult,
            temperature=0.0,
        ),
    )

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


@app.get("/")
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "model": MODEL_NAME,
        "expected_decimals": EXPECTED_DECIMALS or None,
    }


@app.post("/api/read-scale")
async def read_scale(file: UploadFile = File(...)) -> JSONResponse:
    data = await file.read()
    validate_upload(file, data)

    original_img = open_image(data)
    enhanced_img = make_enhanced_display_image(original_img, max_dim=MAX_DIMENSION)

    result = call_gemini(original_img, enhanced_img)

    return JSONResponse(
        content={
            "final": result.model_dump(),
            "previews": {
                "original": pil_to_base64_data_url(original_img),
                "enhanced": pil_to_base64_data_url(enhanced_img),
            },
        }
    )


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host=host, port=port, reload=True)