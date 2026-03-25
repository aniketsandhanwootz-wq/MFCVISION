from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import re
import sys
import threading
import time
import requests
from pathlib import Path
from typing import Any, Literal, Optional
from urllib.parse import urlsplit
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types
from PIL import Image, ImageDraw
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError

from image_enhance import make_enhanced_display_image
from mfc_queue import MFCQueueClient, MFCQueueConfig
from vision import (
    CropDiagnostics,
    LocalDecodeResult,
    analyze_crop_diagnostics,
    decode_display_crop,
    resize_keep_aspect,
)

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = BASE_DIR / "prompts"
STATIC_DIR = BASE_DIR / "static"

LOG_LEVEL = (os.getenv("LOG_LEVEL") or "INFO").upper()


def _build_logger() -> logging.Logger:
    app_logger = logging.getLogger("mfcvision")
    if not app_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        app_logger.addHandler(handler)
    app_logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    app_logger.propagate = False
    return app_logger


logger = _build_logger()
_EMBEDDED_WORKER_THREAD: threading.Thread | None = None
_EMBEDDED_WORKER_STOP_EVENT: threading.Event | None = None

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
CLAPPIA_API_KEY = (os.getenv("CLAPPIA_API_KEY") or "").strip()
CLAPPIA_APP_ID = (os.getenv("CLAPPIA_APP_ID") or "MFC182090").strip()
CLAPPIA_WORKPLACE_ID = (os.getenv("CLAPPIA_WORKPLACE_ID") or "").strip()
CLAPPIA_BASE_URL = (os.getenv("CLAPPIA_BASE_URL") or "https://api-public-v3.clappia.com").strip().rstrip("/")
CLAPPIA_REQUEST_TIMEOUT_SECONDS = float(os.getenv("CLAPPIA_REQUEST_TIMEOUT_SECONDS", "30"))
CLAPPIA_ANALYZE_CONCURRENCY = max(1, int(os.getenv("CLAPPIA_ANALYZE_CONCURRENCY", "6")))
REDIS_URL = (os.getenv("REDIS_URL") or "").strip()
MFC_QUEUE_NAME = (os.getenv("MFC_QUEUE_NAME") or "mfc_clappia_jobs").strip() or "mfc_clappia_jobs"
MFC_FAILED_QUEUE_NAME = (os.getenv("MFC_FAILED_QUEUE_NAME") or "mfc_clappia_failed").strip() or "mfc_clappia_failed"
MFC_REDIS_NAMESPACE = (os.getenv("MFC_REDIS_NAMESPACE") or "mfc_clappia").strip() or "mfc_clappia"
MFC_WRITEBACK_MODE = (os.getenv("MFC_WRITEBACK_MODE") or "sync").strip().lower() or "sync"
if MFC_WRITEBACK_MODE not in {"sync", "async"}:
    logger.warning(
        "invalid_writeback_mode configured=%s fallback=sync",
        MFC_WRITEBACK_MODE,
    )
    MFC_WRITEBACK_MODE = "sync"

# TODO: verify these Clappia destination fields against the live form whenever
# the Clappia app schema changes. Keep writes explicit; do not fall back to
# auto-writing unknown AI keys into Clappia.
CLAPPIA_FIELD_CONFIG: dict[str, dict[str, Optional[str]]] = {
    "ai_pre_weight_1": {"value": "pre_weight", "status": None, "reason": None},
    "ai_pre_weight_2": {"value": "pre_weight_1", "status": None, "reason": None},
    "ai_pre_weight_3": {"value": "pre_weight_2", "status": None, "reason": None},
    "ai_pre_weight_4": {"value": "pre_weight_3", "status": None, "reason": None},
    "ai_post_weight_1": {"value": "pre_weight_4", "status": None, "reason": None},
    "ai_post_weight_2": {"value": "pre_weight_5", "status": None, "reason": None},
    "ai_post_weight_3": {"value": "pre_weight_6", "status": None, "reason": None},
    "ai_post_weight_4": {"value": "pre_weight_7", "status": None, "reason": None},
    "ai_side_wall_1": {"value": "pre_weight_8", "status": None, "reason": None},
    "ai_side_wall_2": {"value": "side_wall__1", "status": None, "reason": None},
    "ai_side_wall_3": {"value": "side_wall_", "status": None, "reason": None},
    "ai_side_wall_4": {"value": "side_wall__3", "status": None, "reason": None},
    "ai_centre_wall_1": {"value": "centre_wal", "status": None, "reason": None},
    "ai_centre_wall_2": {"value": "pre_weight_9", "status": None, "reason": None},
    "ai_centre_wall_3": {"value": "side_wall__2", "status": None, "reason": None},
    "ai_centre_wall_4": {"value": "centre_wal_1", "status": None, "reason": None},
}

MFC_QUEUE = MFCQueueClient(
    MFCQueueConfig(
        redis_url=REDIS_URL,
        queue_name=MFC_QUEUE_NAME,
        failed_queue_name=MFC_FAILED_QUEUE_NAME,
        namespace=MFC_REDIS_NAMESPACE,
    )
)

if not os.getenv("GEMINI_API_KEY"):
    raise RuntimeError("Missing GEMINI_API_KEY in environment.")

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI(title="Gemini Scale Reader", version="9.2.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

_LAST_PREVIEWS: dict[str, bytes] = {}

# ------------------------------------------------------------------ #
# Prompt cache — load once, never hit disk again                       #
# ------------------------------------------------------------------ #
_PROMPT_CACHE: dict[str, str] = {}


def load_prompt_text(filename: str = "scale_reader.txt") -> str:
    cached = _PROMPT_CACHE.get(filename)
    if cached is not None:
        return cached
    prompt_path = PROMPTS_DIR / filename
    if not prompt_path.exists():
        raise RuntimeError(f"Prompt file not found: {prompt_path}")
    prompt_text = prompt_path.read_text(encoding="utf-8").strip()
    _PROMPT_CACHE[filename] = prompt_text
    return prompt_text


def render_prompt_template(
    filename: str,
    replacements: dict[str, Any] | None = None,
) -> str:
    prompt_text = load_prompt_text(filename)
    if replacements:
        for key, value in replacements.items():
            prompt_text = prompt_text.replace(f"__{key}__", str(value))
    return prompt_text


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


class ReadVerificationResult(BaseModel):
    verdict: Literal["confirmed", "corrected", "uncertain"]
    suggested_value_text: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    agrees_with_primary: bool
    decimal_visible: bool
    leftmost_digit_fully_visible: bool
    crop_edge_clipped: bool
    reason: str


class DecimalVerificationResult(BaseModel):
    same_digits_except_decimal: bool
    decimal_visible: Literal["yes", "no", "uncertain"]
    decimal_after_digit: Optional[int] = Field(default=None, ge=1, le=12)
    dot_is_real_not_glare: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class ClappiaAnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    submission_id: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("submission_id", "submissionId"),
    )
    workplace_id: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("workplace_id", "workplaceId"),
    )
    requesting_user_email_address: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices(
            "requesting_user_email_address",
            "requestingUserEmailAddress",
        ),
    )
    targets: dict[str, Any] = Field(default_factory=dict)


class ClappiaSingleResult(BaseModel):
    value: Optional[float] = None
    value_text: Optional[str] = None
    status: str
    confidence: float
    reason: str
    ignored_text_present: bool


class ClappiaWritebackResult(BaseModel):
    enabled: bool
    attempted: bool
    success: bool
    submission_id: Optional[str] = None
    app_id: Optional[str] = None
    endpoint: Optional[str] = None
    payload: dict[str, Any] = Field(default_factory=dict)
    written_fields: dict[str, Any] = Field(default_factory=dict)
    target_field_map: dict[str, str] = Field(default_factory=dict)
    skipped_targets: dict[str, Any] = Field(default_factory=dict)
    response_status: Optional[int] = None
    response_text: Optional[str] = None
    response_body: Any = None
    error: Optional[str] = None


class ClappiaJobPayload(BaseModel):
    job_id: str
    trace_id: str
    submitted_at: float
    submission_id: Optional[str] = None
    workplace_id: Optional[str] = None
    requesting_user_email_address: Optional[str] = None
    targets: dict[str, str] = Field(default_factory=dict)
    clappia_input_payload: dict[str, Any] = Field(default_factory=dict)
    dedupe_hash: Optional[str] = None


_CLAPPIA_RESERVED_KEYS = {
    "submission_id",
    "submissionId",
    "workplace_id",
    "workplaceId",
    "requesting_user_email_address",
    "requestingUserEmailAddress",
    "targets",
}


def _trace_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:8]}"


def _log_url_summary(url: str) -> str:
    parts = urlsplit(url)
    path = parts.path or "/"
    return f"{parts.scheme}://{parts.netloc}{path}"


def _pipeline_log_context(
    *,
    trace_id: str | None,
    source: str,
    target_key: str | None = None,
) -> str:
    context = [f"trace_id={trace_id or '-'}", f"source={source}"]
    if target_key:
        context.append(f"target={target_key}")
    return " ".join(context)


def _normalize_remote_image_url(candidate: Any) -> Optional[str]:
    if isinstance(candidate, str):
        value = candidate.strip()
        if value.startswith(("http://", "https://")):
            return value
        return None

    if isinstance(candidate, dict):
        for key in ("publicurl", "publicUrl", "url", "downloadUrl", "download_url", "href"):
            normalized = _normalize_remote_image_url(candidate.get(key))
            if normalized:
                return normalized
        for key in ("value", "file", "image"):
            normalized = _normalize_remote_image_url(candidate.get(key))
            if normalized:
                return normalized
        return None

    if isinstance(candidate, (list, tuple)):
        for item in candidate:
            normalized = _normalize_remote_image_url(item)
            if normalized:
                return normalized

    return None


def _normalize_clappia_token(candidate: Any) -> Optional[str]:
    if not isinstance(candidate, str):
        return None

    value = candidate.strip()
    if not value:
        return None

    # Clappia workflow misconfiguration often sends unresolved placeholders literally.
    if re.fullmatch(r"\{[^{}]+\}", value) or re.fullmatch(r"\{\$[^{}]+\}", value):
        return None

    return value


def _mask_email(candidate: Any) -> Optional[str]:
    if not isinstance(candidate, str) or "@" not in candidate:
        return None
    local, domain = candidate.split("@", 1)
    if not local:
        return f"***@{domain}"
    if len(local) <= 2:
        return f"{local[0]}***@{domain}"
    return f"{local[:2]}***@{domain}"


def _sanitize_for_log(value: Any, *, key_hint: str | None = None) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _sanitize_for_log(item, key_hint=str(key))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_for_log(item, key_hint=key_hint) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_for_log(item, key_hint=key_hint) for item in value]
    if isinstance(value, str):
        if value.startswith(("http://", "https://")):
            return _log_url_summary(value)
        if (key_hint or "").lower() in {
            "requesting_user_email_address",
            "requestinguseremailaddress",
        }:
            return _mask_email(value)
        if "@" in value and " " not in value:
            masked = _mask_email(value)
            if masked:
                return masked
        if len(value) > 500:
            return f"{value[:497]}..."
    return value


def _build_sanitized_clappia_input_payload(
    *,
    payload: ClappiaAnalyzeRequest,
    targets: dict[str, str],
) -> dict[str, Any]:
    return _sanitize_for_log(
        {
            "submission_id": payload.submission_id,
            "workplace_id": payload.workplace_id,
            "requesting_user_email_address": payload.requesting_user_email_address,
            "targets": targets,
        }
    )


def extract_clappia_targets(payload: ClappiaAnalyzeRequest) -> dict[str, str]:
    normalized_targets: dict[str, str] = {}

    raw_targets = payload.targets if isinstance(payload.targets, dict) else {}
    for output_key, raw_value in raw_targets.items():
        safe_key = str(output_key).strip()
        if not safe_key:
            continue
        image_url = _normalize_remote_image_url(raw_value)
        if image_url:
            normalized_targets[safe_key] = image_url

    if normalized_targets:
        return normalized_targets

    extra_fields = payload.model_extra or {}
    for output_key, raw_value in extra_fields.items():
        if output_key in _CLAPPIA_RESERVED_KEYS:
            continue

        safe_key = str(output_key).strip()
        if not safe_key or safe_key in normalized_targets:
            continue

        image_url = _normalize_remote_image_url(raw_value)
        if image_url:
            normalized_targets[safe_key] = image_url

    return normalized_targets


def _field_config_for_target(target_key: str) -> dict[str, Optional[str]]:
    return CLAPPIA_FIELD_CONFIG.get(target_key, {})


def _compact_clappia_field_mapping() -> dict[str, Optional[str]]:
    return {
        source_key: field_config.get("value")
        for source_key, field_config in sorted(CLAPPIA_FIELD_CONFIG.items())
    }


def build_clappia_writeback_data(
    results: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    writeback_data: dict[str, Any] = {}
    target_field_map: dict[str, str] = {}
    destination_sources: dict[str, str] = {}
    skipped_targets: dict[str, Any] = {}

    for source_key, final in results.items():
        field_config = _field_config_for_target(source_key)
        destination_field = field_config.get("value")

        if not destination_field:
            skipped_targets[source_key] = {
                "skipped_reason": "no writeback field mapping configured",
            }
            continue

        if not isinstance(final, dict):
            skipped_targets[source_key] = {
                "destination_field": destination_field,
                "skipped_reason": "missing final analysis payload",
            }
            continue

        status = final.get("status")
        value_number = final.get("value_number", final.get("value"))
        reason = final.get("reason")

        if status != "ok":
            skipped_targets[source_key] = {
                "destination_field": destination_field,
                "status": status,
                "reason": reason,
                "skipped_reason": "result status was not ok",
            }
            continue

        if value_number is None:
            skipped_targets[source_key] = {
                "destination_field": destination_field,
                "status": status,
                "reason": reason,
                "skipped_reason": "numeric value missing",
            }
            continue

        existing_source_key = destination_sources.get(destination_field)
        if existing_source_key:
            skipped_targets[source_key] = {
                "destination_field": destination_field,
                "status": status,
                "reason": reason,
                "skipped_reason": "destination field already populated by another successful result",
                "existing_source_key": existing_source_key,
            }
            continue

        writeback_data[destination_field] = value_number
        target_field_map[source_key] = destination_field
        destination_sources[destination_field] = source_key

        status_field = field_config.get("status")
        if status_field:
            writeback_data[status_field] = status

        reason_field = field_config.get("reason")
        if reason_field and reason:
            writeback_data[reason_field] = reason

    return writeback_data, target_field_map, skipped_targets


def _writeback_status_from_result(result: ClappiaWritebackResult) -> str:
    if not result.enabled:
        return "disabled"
    if result.success:
        return "success"
    if result.attempted:
        return "failed"
    return "skipped"


def _build_clappia_writeback_summary(result: ClappiaWritebackResult) -> dict[str, Any]:
    return _sanitize_for_log(
        {
            "enabled": result.enabled,
            "attempted": result.attempted,
            "success": result.success,
            "response_status": result.response_status,
            "written_fields": sorted(result.written_fields.keys()),
            "target_field_map": result.target_field_map,
            "skipped_targets": result.skipped_targets,
            "error": result.error,
            "response_body": result.response_body if result.response_body is not None else result.response_text,
        }
    )


def _build_per_target_response(
    final_results: dict[str, dict[str, Any]],
    *,
    target_field_map: dict[str, str],
    skipped_targets: dict[str, Any],
) -> dict[str, Any]:
    response: dict[str, Any] = {}
    for target_key, final in final_results.items():
        field_config = _field_config_for_target(target_key)
        destination_field = field_config.get("value")
        enriched = dict(final)
        enriched["writeback_field"] = destination_field
        enriched["included_in_writeback"] = target_key in target_field_map
        if target_key in skipped_targets:
            enriched["writeback_skip_reason"] = skipped_targets[target_key].get("skipped_reason")
        response[target_key] = enriched
    return response


def _parse_json_response_or_none(response: requests.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return None


def _evaluate_clappia_application_success(
    response_body: Any,
    response_text: Optional[str],
) -> tuple[str, str]:
    if isinstance(response_body, dict):
        status_value = str(response_body.get("status", "")).strip().lower()
        message_value = str(response_body.get("message", "")).strip().lower()
        error_value = response_body.get("error")

        if (
            error_value
            or response_body.get("success") is False
            or response_body.get("ok") is False
            or status_value in {"error", "failed", "failure"}
        ):
            return "rejected", "explicit_failure_indicator"

        if (
            response_body.get("success") is True
            or response_body.get("ok") is True
            or status_value in {"success", "ok"}
            or message_value in {"success", "ok", "updated", "submitted"}
        ):
            return "confirmed", "explicit_success_indicator"

        if not response_body:
            return "uncertain", "empty_json_object"

        return "uncertain", "json_without_explicit_success_indicator"

    if response_text:
        lowered = response_text.lower()
        if "success" in lowered or '"ok"' in lowered:
            return "confirmed", "success_text_indicator"
        return "uncertain", "non_json_2xx_response"

    return "uncertain", "empty_2xx_response"


def update_clappia_submission(
    *,
    trace_id: str,
    submission_id: Optional[str],
    data: dict[str, Any],
    target_field_map: dict[str, str],
    skipped_targets: dict[str, Any],
    workplace_id: Optional[str] = None,
    requesting_user_email_address: Optional[str] = None,
) -> ClappiaWritebackResult:
    endpoint = f"{CLAPPIA_BASE_URL}/submissions/edit"
    normalized_submission_id = _normalize_clappia_token(submission_id)
    normalized_workplace_id = _normalize_clappia_token(workplace_id) or _normalize_clappia_token(CLAPPIA_WORKPLACE_ID)
    normalized_requesting_email = _normalize_clappia_token(requesting_user_email_address)

    payload: dict[str, Any] = {
        "appId": CLAPPIA_APP_ID or None,
        "submissionId": normalized_submission_id,
        "data": data,
    }
    if normalized_workplace_id:
        payload["workplaceId"] = normalized_workplace_id
    if normalized_requesting_email:
        payload["requestingUserEmailAddress"] = normalized_requesting_email

    result = ClappiaWritebackResult(
        enabled=bool(CLAPPIA_API_KEY and CLAPPIA_APP_ID),
        attempted=False,
        success=False,
        submission_id=normalized_submission_id,
        app_id=CLAPPIA_APP_ID or None,
        endpoint=endpoint,
        payload=payload,
        written_fields=dict(data),
        target_field_map=dict(target_field_map),
        skipped_targets=dict(skipped_targets),
    )

    if not CLAPPIA_API_KEY:
        result.error = "Clappia writeback disabled: missing CLAPPIA_API_KEY."
        logger.warning(
            "clappia_writeback_disabled trace_id=%s submission_id=%s reason=missing_api_key",
            trace_id,
            submission_id,
        )
        return result

    if not CLAPPIA_APP_ID:
        result.error = "Clappia writeback disabled: missing CLAPPIA_APP_ID."
        logger.warning(
            "clappia_writeback_disabled trace_id=%s submission_id=%s reason=missing_app_id",
            trace_id,
            submission_id,
        )
        return result

    if not normalized_submission_id:
        result.error = "Clappia writeback skipped: missing submission_id."
        logger.warning(
            "clappia_writeback_skipped trace_id=%s reason=missing_submission_id",
            trace_id,
        )
        return result

    if not normalized_workplace_id:
        result.error = "Clappia writeback skipped: missing workplaceId."
        logger.warning(
            "clappia_writeback_skipped trace_id=%s submission_id=%s reason=missing_workplace_id",
            trace_id,
            normalized_submission_id,
        )
        return result

    if not data:
        result.error = "Clappia writeback skipped: no successful numeric values to write back."
        logger.info(
            "clappia_writeback_skipped trace_id=%s submission_id=%s reason=no_successful_values skipped_targets=%s",
            trace_id,
            normalized_submission_id,
            sorted(skipped_targets.keys()),
        )
        return result

    logger.info(
        "clappia_writeback_start trace_id=%s submission_id=%s field_map=%s payload_keys=%s payload=%s",
        trace_id,
        normalized_submission_id,
        target_field_map,
        sorted(data.keys()),
        json.dumps(_sanitize_for_log(payload), sort_keys=True),
    )

    headers = {
        "x-api-key": CLAPPIA_API_KEY,
        "Content-Type": "application/json",
    }

    try:
        result.attempted = True
        response = requests.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=CLAPPIA_REQUEST_TIMEOUT_SECONDS,
        )
        result.response_status = response.status_code
        result.response_text = response.text[:4000] if response.text else None
        result.response_body = _parse_json_response_or_none(response)
        sanitized_response_body = _sanitize_for_log(
            result.response_body if result.response_body is not None else result.response_text
        )
        application_status, application_reason = _evaluate_clappia_application_success(
            result.response_body,
            result.response_text,
        )
        logger.info(
            "clappia_writeback_response trace_id=%s submission_id=%s response_status=%s application_status=%s application_reason=%s response_body=%s",
            trace_id,
            normalized_submission_id,
            response.status_code,
            application_status,
            application_reason,
            json.dumps(sanitized_response_body, sort_keys=True),
        )

        if 200 <= response.status_code < 300:
            if application_status == "rejected":
                result.error = "Clappia writeback returned HTTP 2xx with an application-level failure indicator."
                logger.warning(
                    "clappia_writeback_failed trace_id=%s submission_id=%s response_status=%s application_status=%s response_body=%s",
                    trace_id,
                    normalized_submission_id,
                    response.status_code,
                    application_status,
                    json.dumps(sanitized_response_body, sort_keys=True),
                )
                return result

            result.success = True
            logger.info(
                "clappia_writeback_success trace_id=%s submission_id=%s response_status=%s application_status=%s written_fields=%s response_body=%s",
                trace_id,
                normalized_submission_id,
                response.status_code,
                application_status,
                sorted(data.keys()),
                json.dumps(sanitized_response_body, sort_keys=True),
            )
            return result

        result.error = f"Clappia writeback returned HTTP {response.status_code}."
        logger.warning(
            "clappia_writeback_failed trace_id=%s submission_id=%s response_status=%s application_status=%s response_body=%s",
            trace_id,
            normalized_submission_id,
            response.status_code,
            application_status,
            json.dumps(sanitized_response_body, sort_keys=True),
        )
        return result

    except requests.RequestException as e:
        result.attempted = True
        result.error = str(e)
        logger.exception(
            "clappia_writeback_exception trace_id=%s submission_id=%s error=%s",
            trace_id,
            normalized_submission_id,
            e,
        )
        return result


def _make_clappia_target_error_analysis(error: Exception) -> dict[str, Any]:
    return {
        "final": {
            "status": "needs_review",
            "value_text": None,
            "value_number": None,
            "confidence": 0.0,
            "reason": str(error),
            "ignored_text_present": False,
        },
        "localization": {
            "source": "clappia_url_error",
            "found": False,
            "confidence": 0.0,
            "display_kind": "unknown",
            "reason": str(error),
        },
    }


async def _analyze_clappia_target(
    *,
    trace_id: str,
    submission_id: Optional[str],
    target_key: str,
    image_url: str,
    semaphore: asyncio.Semaphore,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    async with semaphore:
        try:
            logger.info(
                "clappia_target_start trace_id=%s submission_id=%s target=%s url=%s",
                trace_id,
                submission_id,
                target_key,
                _log_url_summary(image_url),
            )
            image_bytes, content_type = await asyncio.to_thread(
                fetch_image_bytes_from_url,
                image_url,
                trace_id=trace_id,
                target_key=target_key,
            )
            analysis = await asyncio.to_thread(
                run_scale_reader_pipeline,
                image_bytes,
                content_type=content_type,
                trace_id=trace_id,
                source="clappia",
                target_key=target_key,
            )

            final = analysis["final"]
            numeric_value = final["value_number"] if final["status"] == "ok" else None
            single_result = ClappiaSingleResult(
                value=numeric_value,
                value_text=final.get("value_text"),
                status=final.get("status"),
                confidence=final.get("confidence"),
                reason=final.get("reason"),
                ignored_text_present=bool(final.get("ignored_text_present", False)),
            ).model_dump()

            logger.info(
                "clappia_target_complete trace_id=%s submission_id=%s target=%s status=%s value_text=%s confidence=%s",
                trace_id,
                submission_id,
                target_key,
                final.get("status"),
                final.get("value_text"),
                final.get("confidence"),
            )
            return target_key, single_result, analysis

        except Exception as e:
            logger.exception(
                "clappia_target_error trace_id=%s submission_id=%s target=%s error=%s",
                trace_id,
                submission_id,
                target_key,
                e,
            )
            single_result = ClappiaSingleResult(
                value=None,
                value_text=None,
                status="needs_review",
                confidence=0.0,
                reason=str(e),
                ignored_text_present=False,
            ).model_dump()
            return target_key, single_result, _make_clappia_target_error_analysis(e)

def validate_upload(file: Any, data: bytes) -> None:
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

def fetch_image_bytes_from_url(
    url: str,
    *,
    trace_id: str | None = None,
    target_key: str | None = None,
) -> tuple[bytes, str]:
    if not url or not isinstance(url, str):
        raise HTTPException(status_code=400, detail="Image URL is missing.")

    request_context = _pipeline_log_context(
        trace_id=trace_id,
        source="clappia_fetch",
        target_key=target_key,
    )
    safe_url = _log_url_summary(url)
    fetch_started = time.monotonic()
    logger.info("remote_fetch_start %s url=%s", request_context, safe_url)

    try:
        resp = requests.get(url, timeout=30)
    except requests.RequestException as e:
        logger.warning(
            "remote_fetch_error %s url=%s error=%s",
            request_context,
            safe_url,
            e,
        )
        raise HTTPException(status_code=400, detail=f"Could not fetch image URL: {e}")

    if resp.status_code != 200:
        logger.warning(
            "remote_fetch_bad_status %s url=%s status_code=%s",
            request_context,
            safe_url,
            resp.status_code,
        )
        raise HTTPException(
            status_code=400,
            detail=f"Image URL returned HTTP {resp.status_code}.",
        )

    content_type = resp.headers.get("content-type", "image/jpeg")
    data = resp.content or b""

    if not content_type.startswith("image/"):
        logger.warning(
            "remote_fetch_bad_content_type %s url=%s content_type=%s",
            request_context,
            safe_url,
            content_type,
        )
        raise HTTPException(
            status_code=400,
            detail=f"Remote URL is not an image. content-type={content_type}",
        )

    if len(data) > MAX_IMAGE_BYTES:
        logger.warning(
            "remote_fetch_too_large %s url=%s bytes=%s",
            request_context,
            safe_url,
            len(data),
        )
        raise HTTPException(
            status_code=400,
            detail=f"Remote image too large. Max allowed size is {MAX_IMAGE_MB} MB.",
        )

    logger.info(
        "remote_fetch_complete %s url=%s bytes=%s content_type=%s elapsed_seconds=%.2f",
        request_context,
        safe_url,
        len(data),
        content_type,
        time.monotonic() - fetch_started,
    )
    return data, content_type
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
    localizer_contents = [
        instructions,
        load_prompt_text("localizer_primary_image_label.txt"),
        localizer_original_img,
        load_prompt_text("localizer_context_image_label.txt"),
        localizer_enhanced_img,
    ]
    try:
        response = client.models.generate_content(
            model=selected_model,
            contents=localizer_contents,
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
                contents=localizer_contents,
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


def _display_kind_label(display_kind: str) -> str:
    return display_kind.upper() if display_kind in {"led", "lcd"} else "UNKNOWN"


def _primary_localizer_instructions() -> str:
    return load_prompt_text("localizer_primary.txt")


def _refine_localizer_instructions(previous: LocalizationResult) -> str:
    return render_prompt_template(
        "localizer_refine.txt",
        {
            "X1": previous.x1,
            "Y1": previous.y1,
            "X2": previous.x2,
            "Y2": previous.y2,
            "DISPLAY_KIND": previous.display_kind,
            "CONFIDENCE": previous.confidence,
        },
    )


def _region_refine_instructions(display_kind_hint: str, primary_cy_frac: float = 0.5) -> str:
    if display_kind_hint == "led":
        snippet_name = (
            "localizer_region_led_lower_bias.txt"
            if primary_cy_frac < 0.55
            else "localizer_region_led.txt"
        )
    elif display_kind_hint == "lcd":
        snippet_name = "localizer_region_lcd.txt"
    else:
        snippet_name = "localizer_region_unknown.txt"
    return render_prompt_template(
        "localizer_region_base.txt",
        {
            "WHAT_TO_FIND": load_prompt_text(snippet_name),
        },
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

        result.value_text = normalize_numeric_token_text(result.value_text)

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


def normalize_numeric_token_text(value_text: str) -> str:
    text = value_text.strip().replace(" ", "")
    if not re.fullmatch(r"\d+(\.\d+)?", text):
        return text
    if "." in text:
        left, right = text.split(".", 1)
        return f"{int(left)}.{right}"
    return str(int(text))


def count_numeric_digits(value_text: str) -> int:
    return len(value_text.replace(".", "")) if value_text else 0


def count_integer_digits(value_text: str) -> int:
    if not value_text:
        return 0
    return len(value_text.split(".", 1)[0])


def digits_only(value_text: str | None) -> str:
    if not value_text:
        return ""
    return value_text.replace(".", "")


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


def _is_led_single_digit_risk(
    result: ReadingResult,
    crop_diagnostics: CropDiagnostics | None,
    *,
    display_kind: str,
) -> bool:
    if display_kind != "led" or result.status != "ok" or not result.value_text:
        return False
    if "." not in result.value_text or count_integer_digits(result.value_text) != 1:
        return False
    left_digit = result.value_text.split(".", 1)[0]
    if result.ignored_text_present:
        return True
    if left_digit in {"0", "7", "8"}:
        return True
    if crop_diagnostics is None or crop_diagnostics.mode != "led":
        return False
    return (
        crop_diagnostics.leading_blank_ratio <= 0.30
        and crop_diagnostics.active_span_ratio >= 0.60
    )


def _support_extends_led_primary(primary_text: str, support_text: str) -> bool:
    if not primary_text or not support_text:
        return False
    if not re.fullmatch(r"\d+(\.\d+)?", primary_text):
        return False
    if not re.fullmatch(r"\d+(\.\d+)?", support_text):
        return False
    if "." in primary_text and "." in support_text:
        if len(primary_text.split(".", 1)[1]) != len(support_text.split(".", 1)[1]):
            return False
    if count_integer_digits(primary_text) != 1:
        return False
    if count_integer_digits(support_text) < 2:
        return False
    return support_text.startswith("1")


def _is_leading_one_shortening(longer_text: str, shorter_text: str) -> bool:
    if not longer_text or not shorter_text:
        return False
    if not re.fullmatch(r"\d+(\.\d+)?", longer_text):
        return False
    if not re.fullmatch(r"\d+(\.\d+)?", shorter_text):
        return False
    if not longer_text.startswith("1"):
        return False
    if "." in longer_text or "." in shorter_text:
        longer_right = longer_text.split(".", 1)[1] if "." in longer_text else ""
        shorter_right = shorter_text.split(".", 1)[1] if "." in shorter_text else ""
        if len(longer_right) != len(shorter_right):
            return False
    return longer_text[1:] == shorter_text


def _support_should_win_led_conflict(
    primary_result: ReadingResult,
    verifier_result: ReadingResult | None,
    support_candidate: ReadingResult | None,
    crop_diagnostics: CropDiagnostics | None,
    *,
    display_kind: str,
) -> bool:
    if display_kind != "led":
        return False
    if support_candidate is None or support_candidate.status != "ok" or not support_candidate.value_text:
        return False
    if is_suspicious_read(support_candidate):
        return False
    if primary_result.status != "ok" or not primary_result.value_text:
        return False

    primary_text = primary_result.value_text
    support_text = support_candidate.value_text
    verifier_text = verifier_result.value_text if verifier_result is not None else None

    if (
        _is_led_single_digit_risk(primary_result, crop_diagnostics, display_kind=display_kind)
        and _support_extends_led_primary(primary_text, support_text)
        and support_text != primary_text
    ):
        return True

    if (
        verifier_text
        and primary_text.startswith("1")
        and primary_text[1:] == verifier_text
        and support_text.startswith("1")
        and count_numeric_digits(support_text) >= count_numeric_digits(verifier_text) + 1
    ):
        return True

    return False


def _same_digits_except_decimal(first_text: str | None, second_text: str | None) -> bool:
    if not first_text or not second_text or first_text == second_text:
        return False
    if not re.fullmatch(r"\d+(\.\d+)?", first_text):
        return False
    if not re.fullmatch(r"\d+(\.\d+)?", second_text):
        return False
    return digits_only(first_text) == digits_only(second_text)


def _collect_decimal_conflict_texts(*value_texts: str | None) -> list[str]:
    seen: set[str] = set()
    texts: list[str] = []
    for value_text in value_texts:
        if not value_text or value_text in seen:
            continue
        if not re.fullmatch(r"\d+(\.\d+)?", value_text):
            continue
        seen.add(value_text)
        texts.append(value_text)
    for index, left_text in enumerate(texts):
        for right_text in texts[index + 1 :]:
            if _same_digits_except_decimal(left_text, right_text):
                signature = digits_only(left_text)
                return [text for text in texts if digits_only(text) == signature]
    return []


def _is_decimal_risk_text(value_text: str | None, *, display_kind: str) -> bool:
    if not value_text or "." in value_text:
        return False
    if not re.fullmatch(r"\d+", value_text):
        return False
    if count_numeric_digits(value_text) < 4:
        return False
    return display_kind in {"led", "lcd", "unknown"}


def _decimal_signature_for_verification(
    primary_text: str | None,
    verifier_text: str | None,
    support_text: str | None,
    *,
    display_kind: str,
) -> tuple[list[str], str | None]:
    conflict_texts = _collect_decimal_conflict_texts(primary_text, verifier_text, support_text)
    if conflict_texts:
        return conflict_texts, digits_only(conflict_texts[0])
    if _is_decimal_risk_text(primary_text, display_kind=display_kind):
        return [primary_text], digits_only(primary_text)
    return [], None


def _rebuild_decimal_text(digits_text: str, decimal_after_digit: int | None) -> str | None:
    if not digits_text or decimal_after_digit is None:
        return None
    if decimal_after_digit <= 0 or decimal_after_digit >= len(digits_text):
        return None
    return normalize_numeric_token_text(
        f"{digits_text[:decimal_after_digit]}.{digits_text[decimal_after_digit:]}"
    )


def _expected_decimals_for_target(target_key: str | None) -> int | None:
    if EXPECTED_DECIMALS:
        try:
            return int(EXPECTED_DECIMALS)
        except ValueError:
            return None

    key = (target_key or "").strip().lower()
    if not key:
        return None

    three_decimal_prefixes = (
        "ai_pre_weight_",
        "ai_post_weight_",
        "ai_side_wall_",
        "ai_centre_wall_",
    )
    three_decimal_fragments = (
        "pre_weight_sample_",
        "post_weight_sample_",
        "side_wall_thickness_sample_",
        "centre_wall_thickness_sample_",
    )
    if key.startswith(three_decimal_prefixes) or any(fragment in key for fragment in three_decimal_fragments):
        return 3
    return None


def _apply_target_decimal_contract(
    result: ReadingResult,
    *,
    target_key: str | None,
) -> ReadingResult:
    if result.status != "ok" or not result.value_text:
        return result

    expected = _expected_decimals_for_target(target_key)
    if expected is None or expected < 0:
        return result

    text = result.value_text.strip().replace(" ", "")
    if not re.fullmatch(r"\d+(\.\d+)?", text):
        return result

    if "." in text and len(text.split(".", 1)[1]) == expected:
        return result

    digits = digits_only(text)
    if not digits:
        return result

    if len(digits) <= expected:
        rebuilt = f"0.{digits.zfill(expected)}"
    else:
        rebuilt = f"{digits[:-expected]}.{digits[-expected:]}"
    rebuilt = normalize_numeric_token_text(rebuilt)
    if rebuilt == text:
        return result

    result.value_text = rebuilt
    result.value_number = float(rebuilt)
    result.confidence = min(result.confidence, 0.95)
    result.reason = (
        f"{result.reason} Applied the fixed {expected}-decimal contract for this target."
    )
    return result


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


def _reading_from_candidate_text(
    value_text: Optional[str],
    *,
    confidence: float,
    reason: str,
    ignored_text_present: bool = False,
) -> ReadingResult | None:
    if not value_text or not isinstance(value_text, str):
        return None
    candidate = ReadingResult(
        status="ok",
        value_text=value_text.strip().replace(" ", ""),
        value_number=None,
        confidence=max(0.0, min(confidence, 1.0)),
        reason=reason,
        ignored_text_present=ignored_text_present,
    )
    candidate = post_process_result(candidate)
    if candidate.status != "ok" or not candidate.value_text:
        return None
    return candidate


def _best_ok_result(*candidates: ReadingResult | None) -> ReadingResult | None:
    for candidate in candidates:
        if candidate is not None and candidate.status == "ok" and candidate.value_text:
            return candidate.model_copy(deep=True)
    return None


def _serialize_optional_reading_result(result: ReadingResult | None) -> dict[str, Any]:
    if result is None:
        return {"available": False}
    payload = result.model_dump()
    payload["available"] = True
    return payload


def _serialize_local_decode(local_decode: LocalDecodeResult | None) -> dict[str, Any]:
    if local_decode is None:
        return {"available": False}
    return {
        "available": True,
        "ok": local_decode.ok,
        "value_text": local_decode.value_text,
        "confidence": local_decode.confidence,
        "digit_count": local_decode.digit_count,
        "decimal_count": local_decode.decimal_count,
        "reason": local_decode.reason,
    }


def _serialize_read_verification(verification: ReadVerificationResult | None) -> dict[str, Any]:
    if verification is None:
        return {"available": False}
    return {
        "available": True,
        "verdict": verification.verdict,
        "suggested_value_text": verification.suggested_value_text,
        "confidence": verification.confidence,
        "agrees_with_primary": verification.agrees_with_primary,
        "decimal_visible": verification.decimal_visible,
        "leftmost_digit_fully_visible": verification.leftmost_digit_fully_visible,
        "crop_edge_clipped": verification.crop_edge_clipped,
        "reason": verification.reason,
    }


def _serialize_decimal_verification(
    verification: DecimalVerificationResult | None,
    *,
    candidate_texts: list[str] | None = None,
    resolved_value_text: str | None = None,
) -> dict[str, Any]:
    if verification is None:
        return {"available": False}
    return {
        "available": True,
        "same_digits_except_decimal": verification.same_digits_except_decimal,
        "decimal_visible": verification.decimal_visible,
        "decimal_after_digit": verification.decimal_after_digit,
        "dot_is_real_not_glare": verification.dot_is_real_not_glare,
        "confidence": verification.confidence,
        "reason": verification.reason,
        "candidate_texts": candidate_texts or [],
        "resolved_value_text": resolved_value_text,
    }


def _safe_analyze_crop_diagnostics(
    crop_img: Image.Image,
    *,
    fallback_mode: str,
) -> CropDiagnostics:
    try:
        return analyze_crop_diagnostics(crop_img)
    except Exception as e:
        logger.warning("crop_diagnostics_exception error=%s", e)
        return CropDiagnostics(
            is_reliable=False,
            mode=fallback_mode if fallback_mode in {"led", "lcd"} else "unknown",
            quality_score=0.0,
            lit_ratio=0.0,
            green_ratio=0.0,
            component_count=0,
            active_span_ratio=0.0,
            active_band_height_ratio=0.0,
            leading_blank_ratio=0.0,
            reason=f"Crop diagnostics failed: {e}",
        )


def _maybe_run_local_decoder(
    crop_img: Image.Image,
    crop_diagnostics: CropDiagnostics,
) -> LocalDecodeResult | None:
    if not crop_diagnostics.is_reliable or crop_diagnostics.mode != "led":
        return None
    try:
        return decode_display_crop(crop_img)
    except Exception as e:
        logger.warning("local_decoder_exception error=%s", e)
        return None


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
    prompt = load_prompt_text("scale_reader.txt")
    contents: list[object] = [prompt, instructions]

    if primary_source == "crop":
        contents.extend([load_prompt_text("reader_primary_crop_label.txt"), crop_img])
        if include_secondary:
            contents.extend([load_prompt_text("reader_context_original_label.txt"), original_img])
    else:
        contents.extend([load_prompt_text("reader_primary_original_label.txt"), original_img])
        if include_secondary:
            contents.extend([load_prompt_text("reader_context_crop_label.txt"), crop_img])

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


def _build_read_verifier_instructions(
    *,
    primary_candidate: Optional[str],
    display_kind: str,
    used_full_image_fallback: bool,
) -> str:
    candidate_text = primary_candidate or "no numeric candidate"
    prompt_name = (
        "read_verifier_full_image.txt"
        if used_full_image_fallback
        else "read_verifier_crop.txt"
    )
    return render_prompt_template(
        prompt_name,
        {
            "DISPLAY_KIND_LABEL": _display_kind_label(display_kind),
            "CANDIDATE_TEXT": candidate_text,
        },
    )


def _build_decimal_verifier_instructions(
    *,
    candidate_texts: list[str],
    display_kind: str,
) -> str:
    candidate_block = "\n".join(f"- {value_text}" for value_text in candidate_texts) or "- none"
    return render_prompt_template(
        "decimal_verifier.txt",
        {
            "DISPLAY_KIND_LABEL": _display_kind_label(display_kind),
            "CANDIDATE_LIST": candidate_block,
        },
    )


def call_gemini_decimal_verifier(
    primary_img: Image.Image,
    enhanced_primary_img: Image.Image,
    original_img: Image.Image,
    *,
    candidate_texts: list[str],
    display_kind: str,
) -> DecimalVerificationResult:
    instructions = _build_decimal_verifier_instructions(
        candidate_texts=candidate_texts,
        display_kind=display_kind,
    )
    contents: list[object] = [
        instructions,
        load_prompt_text("decimal_verifier_primary_label.txt"),
        primary_img,
        load_prompt_text("decimal_verifier_secondary_label.txt"),
        enhanced_primary_img,
        load_prompt_text("decimal_verifier_context_label.txt"),
        original_img,
    ]

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=DecimalVerificationResult,
            temperature=0.0,
        ),
    )

    if getattr(response, "parsed", None) is not None:
        parsed = response.parsed
        if isinstance(parsed, DecimalVerificationResult):
            return parsed
        if isinstance(parsed, dict):
            return DecimalVerificationResult(**parsed)

    raw_text = getattr(response, "text", None)
    if not raw_text:
        raise HTTPException(status_code=502, detail="Gemini decimal verifier returned empty response.")

    try:
        payload = json.loads(raw_text)
        return DecimalVerificationResult(**payload)
    except (json.JSONDecodeError, ValidationError) as e:
        raise HTTPException(status_code=502, detail=f"Invalid Gemini decimal verifier JSON response: {e}")


def call_gemini_read_verifier(
    primary_img: Image.Image,
    original_img: Image.Image,
    *,
    primary_candidate: Optional[str],
    display_kind: str,
    used_full_image_fallback: bool,
) -> ReadVerificationResult:
    instructions = _build_read_verifier_instructions(
        primary_candidate=primary_candidate,
        display_kind=display_kind,
        used_full_image_fallback=used_full_image_fallback,
    )
    contents: list[object] = [instructions]
    if used_full_image_fallback:
        contents.extend([load_prompt_text("verifier_primary_original_label.txt"), original_img])
    else:
        contents.extend(
            [
                load_prompt_text("verifier_primary_crop_label.txt"),
                primary_img,
                load_prompt_text("verifier_context_original_support_label.txt"),
                original_img,
            ]
        )

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ReadVerificationResult,
            temperature=0.0,
        ),
    )

    if getattr(response, "parsed", None) is not None:
        parsed = response.parsed
        if isinstance(parsed, ReadVerificationResult):
            return parsed
        if isinstance(parsed, dict):
            return ReadVerificationResult(**parsed)

    raw_text = getattr(response, "text", None)
    if not raw_text:
        raise HTTPException(status_code=502, detail="Gemini verifier returned empty response.")

    try:
        payload = json.loads(raw_text)
        return ReadVerificationResult(**payload)
    except (json.JSONDecodeError, ValidationError) as e:
        raise HTTPException(status_code=502, detail=f"Invalid Gemini verifier JSON response: {e}")


def _resolve_verified_result(
    primary_result: ReadingResult,
    *,
    crop_diagnostics: CropDiagnostics | None,
    local_decode: LocalDecodeResult | None,
    verification: ReadVerificationResult | None,
    support_result: ReadingResult | None,
    decimal_verification: DecimalVerificationResult | None,
    decimal_candidate_texts: list[str],
    decimal_resolved_text: str | None,
    display_kind: str,
    used_full_image_fallback: bool,
) -> ReadingResult:
    primary_suspicious = is_suspicious_read(
        primary_result,
        crop_diagnostics,
        expected_digit_count=None,
    )
    local_result = (
        _reading_from_candidate_text(
            local_decode.value_text,
            confidence=local_decode.confidence,
            reason=local_decode.reason,
        )
        if local_decode is not None and local_decode.ok and local_decode.value_text
        else None
    )
    verifier_result = (
        _reading_from_candidate_text(
            verification.suggested_value_text,
            confidence=verification.confidence,
            reason=f"Structured verifier: {verification.reason}",
        )
        if verification is not None and verification.suggested_value_text
        else None
    )
    support_candidate = (
        support_result.model_copy(deep=True)
        if support_result is not None and support_result.status == "ok" and support_result.value_text
        else None
    )

    primary_text = primary_result.value_text if primary_result.status == "ok" else None
    local_text = local_result.value_text if local_result is not None else None
    verifier_text = verifier_result.value_text if verifier_result is not None else None
    support_text = support_candidate.value_text if support_candidate is not None else None
    decimal_candidate = (
        _reading_from_candidate_text(
            decimal_resolved_text,
            confidence=decimal_verification.confidence if decimal_verification is not None else 0.0,
            reason=(
                f"Decimal-only verifier: {decimal_verification.reason}"
                if decimal_verification is not None
                else "Decimal-only verifier produced a decimal-resolved candidate."
            ),
        )
        if decimal_resolved_text
        and decimal_verification is not None
        and decimal_verification.decimal_visible == "yes"
        and decimal_verification.dot_is_real_not_glare
        else None
    )
    decimal_conflict_detected = len(decimal_candidate_texts) >= 2
    decimal_risk_unresolved = (
        decimal_verification is not None
        and decimal_candidate is None
        and (
            decimal_conflict_detected
            or decimal_verification.decimal_visible == "uncertain"
        )
    )
    support_suspicious = (
        is_suspicious_read(support_candidate)
        if support_candidate is not None
        else True
    )
    local_override_allowed = (
        local_decode is not None
        and local_decode.ok
        and local_decode.confidence >= LOCAL_DECODER_MIN_CONFIDENCE
        and local_decode.decimal_count <= 1
    )

    if verifier_result is not None and primary_result.status != "ok":
        resolved = verifier_result.model_copy(deep=True)
        resolved.reason = f"{resolved.reason} Structured verifier replaced an invalid primary read."
        return resolved

    if (
        decimal_candidate is not None
        and primary_result.status == "ok"
        and primary_text
        and _same_digits_except_decimal(decimal_candidate.value_text, primary_text)
    ):
        resolved = decimal_candidate.model_copy(deep=True)
        resolved.confidence = min(0.98, max(resolved.confidence, primary_result.confidence, 0.86))
        resolved.reason = (
            f"{resolved.reason} Decimal-only verifier resolved the decimal placement for the same visible digits."
        )
        return resolved

    if primary_result.status != "ok":
        fallback_result = _best_ok_result(support_candidate, local_result)
        if fallback_result is not None:
            fallback_result.reason = (
                f"{fallback_result.reason} Best available numeric candidate replaced an invalid primary read."
            )
            return fallback_result

    if (
        verifier_result is not None
        and local_result is not None
        and verifier_text
        and verifier_text == local_text
        and verifier_text != primary_text
        and local_override_allowed
    ):
        resolved = verifier_result.model_copy(deep=True)
        resolved.confidence = min(
            0.97,
            max(
                resolved.confidence,
                local_decode.confidence if local_decode is not None else resolved.confidence,
            ),
        )
        resolved.reason = (
            f"{resolved.reason} Consensus override: structured verifier and local decoder agreed against the primary read."
        )
        return resolved

    if (
        primary_result.status == "ok"
        and primary_text
        and primary_suspicious
        and support_candidate is not None
        and support_text
        and not support_suspicious
        and support_text != primary_text
    ):
        resolved = support_candidate.model_copy(deep=True)
        resolved.confidence = min(0.96, max(resolved.confidence, 0.84))
        resolved.reason = (
            f"{resolved.reason} Full-image support read replaced a suspicious crop read."
        )
        return resolved

    if _support_should_win_led_conflict(
        primary_result,
        verifier_result,
        support_candidate,
        crop_diagnostics,
        display_kind=display_kind,
    ):
        resolved = support_candidate.model_copy(deep=True)
        resolved.confidence = min(0.97, max(resolved.confidence, 0.86))
        resolved.reason = (
            f"{resolved.reason} Full-image support read won the LED leading-digit conflict."
        )
        return resolved

    if (
        verifier_result is not None
        and verification is not None
        and verification.verdict == "corrected"
        and primary_result.status == "ok"
        and primary_text
        and verifier_text
        and _is_leading_one_shortening(primary_text, verifier_text)
        and used_full_image_fallback
        and display_kind != "lcd"
        and (support_candidate is None or support_suspicious)
    ):
        resolved = primary_result.model_copy(deep=True)
        resolved.confidence = min(resolved.confidence, 0.82)
        resolved.reason = (
            f"{resolved.reason} Structured verifier proposed dropping the leading '1' during "
            "full-image fallback, but no stronger supporting read existed."
        )
        return resolved

    if (
        verifier_result is not None
        and verification is not None
        and verification.verdict == "corrected"
        and verifier_text
        and verifier_text != primary_text
        and (
            local_result is None
            or local_text == verifier_text
            or primary_suspicious
            or used_full_image_fallback
        )
    ):
        resolved = verifier_result.model_copy(deep=True)
        resolved.confidence = min(0.96, max(resolved.confidence, verification.confidence))
        resolved.reason = (
            f"{resolved.reason} Consensus override: structured verifier replaced the weaker primary read."
        )
        return resolved

    if (
        local_result is not None
        and local_text
        and local_text != primary_text
        and primary_suspicious
        and local_override_allowed
    ):
        resolved = local_result.model_copy(deep=True)
        resolved.reason = (
            f"{resolved.reason} Consensus override: local decoder replaced a suspicious primary read."
        )
        return resolved

    if (
        verification is not None
        and primary_result.status == "ok"
        and primary_text
        and verification.agrees_with_primary
        and (verifier_text is None or verifier_text == primary_text)
    ):
        resolved = primary_result.model_copy(deep=True)
        if decimal_conflict_detected or decimal_risk_unresolved:
            resolved.confidence = min(resolved.confidence, 0.78 if not used_full_image_fallback else 0.72)
            resolved.reason = (
                f"{resolved.reason} Structured verifier agreed on the digits, but decimal placement remained ambiguous."
            )
        else:
            resolved.confidence = min(0.99, max(resolved.confidence, verification.confidence))
            resolved.reason = f"{resolved.reason} Structured verifier confirmed the same reading."
        return resolved

    if (
        local_result is not None
        and primary_result.status == "ok"
        and primary_text
        and local_text == primary_text
    ):
        resolved = primary_result.model_copy(deep=True)
        resolved.confidence = min(
            0.97,
            max(
                resolved.confidence,
                local_decode.confidence if local_decode is not None else resolved.confidence,
            ),
        )
        resolved.reason = f"{resolved.reason} Local decoder matched the same reading."
        return resolved

    if support_candidate is not None and not support_suspicious and support_text == primary_text:
        resolved = primary_result.model_copy(deep=True)
        resolved.confidence = min(0.97, max(resolved.confidence, support_candidate.confidence))
        resolved.reason = f"{resolved.reason} Full-image support read matched the same reading."
        return resolved

    if primary_result.status == "ok":
        resolved = primary_result.model_copy(deep=True)
        if verification is not None and (
            verification.crop_edge_clipped
            or not verification.leftmost_digit_fully_visible
            or ("." in (resolved.value_text or "") and not verification.decimal_visible)
        ):
            resolved.confidence = min(
                resolved.confidence,
                0.78 if not used_full_image_fallback else 0.70,
            )
            resolved.reason = (
                f"{resolved.reason} Structured verifier found ambiguity but did not produce a stronger correction."
            )
        if decimal_conflict_detected or decimal_risk_unresolved:
            resolved.confidence = min(
                resolved.confidence,
                0.76 if not used_full_image_fallback else 0.70,
            )
            resolved.reason = (
                f"{resolved.reason} Decimal placement was checked separately and remained ambiguous."
            )
        return resolved

    if support_candidate is not None:
        return support_candidate
    if verifier_result is not None:
        return verifier_result
    if local_result is not None and local_override_allowed:
        return local_result
    return primary_result


def call_gemini_on_full_image(
    original_img: Image.Image,
    fallback_reason: str,
    *,
    display_kind: str = "unknown",
) -> ReadingResult:
    primary_result = call_gemini_with_instructions(
        render_prompt_template(
            "full_image_fallback.txt",
            {
                "DISPLAY_KIND_LABEL": _display_kind_label(display_kind),
                "FALLBACK_REASON": fallback_reason,
            },
        ),
        original_img,
        original_img,
        model_name=MODEL_NAME,
        primary_source="original",
        include_secondary=False,
    )
    if is_suspicious_read(primary_result):
        primary_result.confidence = min(primary_result.confidence, 0.55)
        primary_result.reason = (
            "Single-pass full-image fallback remained ambiguous and will require verification. "
            f"{primary_result.reason}"
        )
    return primary_result


def run_scale_reader_pipeline(
    data: bytes,
    *,
    content_type: str = "image/jpeg",
    trace_id: str | None = None,
    source: str = "upload",
    target_key: str | None = None,
) -> dict[str, Any]:
    t_start = time.monotonic()
    log_context = _pipeline_log_context(
        trace_id=trace_id,
        source=source,
        target_key=target_key,
    )
    logger.info(
        "pipeline_start %s bytes=%s content_type=%s",
        log_context,
        len(data),
        content_type,
    )

    class _TempFile:
        def __init__(self, content_type: str):
            self.content_type = content_type

    validate_upload(_TempFile(content_type), data)
    original_img = open_image(data)
    logger.info(
        "pipeline_image_opened %s image_size=%sx%s",
        log_context,
        original_img.size[0],
        original_img.size[1],
    )

    localizer_img = resize_keep_aspect(original_img, max_dim=LOCALIZER_MAX_DIMENSION)
    localizer_enhanced_img = make_enhanced_display_image(
        localizer_img, max_dim=LOCALIZER_MAX_DIMENSION
    )
    enhanced_full_img = make_enhanced_display_image(original_img, max_dim=MAX_DIMENSION)
    logger.info(
        "pipeline_images_prepared %s localizer_size=%sx%s enhanced_full_size=%sx%s",
        log_context,
        localizer_img.size[0],
        localizer_img.size[1],
        enhanced_full_img.size[0],
        enhanced_full_img.size[1],
    )

    crop_img: Image.Image = make_placeholder_preview((720, 180), "No localized ROI")
    debug_img: Image.Image = original_img.copy()
    result: ReadingResult
    localization_payload: dict[str, object]
    skipped_refine = False
    quality_recrop_attempted = False
    quality_recrop_applied = False

    best_box_pixels: tuple[int, int, int, int] | None = None
    best_localization: LocalizationResult | None = None
    best_confidence: float = 0.0
    primary_localization: LocalizationResult

    try:
        logger.info(
            "localizer_primary_start %s model=%s",
            log_context,
            LOCALIZER_MODEL_NAME,
        )
        primary_localization = call_gemini_localizer(
            localizer_img,
            localizer_enhanced_img,
            previous=None,
        )
        logger.info(
            "localizer_primary_result %s found=%s confidence=%.3f display_kind=%s reason=%s",
            log_context,
            primary_localization.found,
            primary_localization.confidence,
            primary_localization.display_kind,
            primary_localization.reason,
        )

        if primary_localization.found and primary_localization.confidence >= LOCALIZER_MIN_CONFIDENCE:
            raw_box = localization_box_to_pixels(primary_localization, original_img.size)
            if raw_box and is_valid_localization_box(raw_box, original_img.size):
                if not _is_bad_localization_box(
                    raw_box,
                    original_img.size,
                    primary_localization.display_kind,
                ):
                    expanded = expand_localization_box(
                        raw_box,
                        original_img.size,
                        primary_localization.display_kind,
                    )
                    best_box_pixels = expanded
                    best_localization = primary_localization
                    best_confidence = primary_localization.confidence

        if best_box_pixels is not None:
            _bcy = ((best_box_pixels[1] + best_box_pixels[3]) / 2.0) / max(original_img.size[1], 1)

            if _is_high_quality_localization(
                best_box_pixels,
                original_img.size,
                best_confidence,
                primary_localization.display_kind,
            ):
                skipped_refine = True
                logger.info(
                    "localizer_refine_skipped %s confidence=%.3f",
                    log_context,
                    best_confidence,
                )
            else:
                search_box = make_search_region_box(best_box_pixels, original_img.size)
                search_original = crop_from_box(original_img, search_box)
                search_enhanced = crop_from_box(enhanced_full_img, search_box)
                logger.info(
                    "localizer_refine_start %s search_box=%s",
                    log_context,
                    search_box,
                )

                refined = call_gemini_region_localizer(
                    search_original,
                    search_enhanced,
                    display_kind_hint=primary_localization.display_kind,
                    primary_cy_frac=_bcy,
                )
                logger.info(
                    "localizer_refine_result %s found=%s confidence=%.3f display_kind=%s reason=%s",
                    log_context,
                    refined.found,
                    refined.confidence,
                    refined.display_kind,
                    refined.reason,
                )
                if refined.found and refined.confidence >= LOCALIZER_MIN_CONFIDENCE:
                    refined_local_box = localization_box_to_pixels(refined, search_original.size)
                    if refined_local_box and is_valid_localization_box(
                        refined_local_box,
                        search_original.size,
                    ):
                        mapped = map_child_box_to_parent(search_box, refined_local_box)
                        if not _is_bad_localization_box(
                            mapped,
                            original_img.size,
                            refined.display_kind,
                        ):
                            expanded_mapped = expand_localization_box(
                                mapped,
                                original_img.size,
                                refined.display_kind,
                            )
                            if refined.confidence >= best_confidence:
                                best_box_pixels = expanded_mapped
                                best_localization = refined
                                best_confidence = refined.confidence

        if best_box_pixels is not None:
            tentative_crop = crop_from_box(original_img, best_box_pixels)
            tentative_crop_diagnostics = _safe_analyze_crop_diagnostics(
                tentative_crop,
                fallback_mode=best_localization.display_kind if best_localization is not None else "unknown",
            )
            logger.info(
                "crop_quality_gate %s reliable=%s mode=%s quality_score=%.3f reason=%s",
                log_context,
                tentative_crop_diagnostics.is_reliable,
                tentative_crop_diagnostics.mode,
                tentative_crop_diagnostics.quality_score,
                tentative_crop_diagnostics.reason,
            )
            if not tentative_crop_diagnostics.is_reliable:
                quality_recrop_attempted = True
                search_box = make_search_region_box(best_box_pixels, original_img.size)
                search_original = crop_from_box(original_img, search_box)
                search_enhanced = crop_from_box(enhanced_full_img, search_box)
                logger.info(
                    "localizer_quality_recrop_start %s search_box=%s",
                    log_context,
                    search_box,
                )
                quality_refined = call_gemini_region_localizer(
                    search_original,
                    search_enhanced,
                    display_kind_hint=best_localization.display_kind if best_localization is not None else "unknown",
                    primary_cy_frac=((best_box_pixels[1] + best_box_pixels[3]) / 2.0) / max(original_img.size[1], 1),
                )
                logger.info(
                    "localizer_quality_recrop_result %s found=%s confidence=%.3f display_kind=%s reason=%s",
                    log_context,
                    quality_refined.found,
                    quality_refined.confidence,
                    quality_refined.display_kind,
                    quality_refined.reason,
                )
                if quality_refined.found and quality_refined.confidence >= LOCALIZER_MIN_CONFIDENCE:
                    quality_refined_local_box = localization_box_to_pixels(
                        quality_refined,
                        search_original.size,
                    )
                    if quality_refined_local_box and is_valid_localization_box(
                        quality_refined_local_box,
                        search_original.size,
                    ):
                        mapped = map_child_box_to_parent(search_box, quality_refined_local_box)
                        if not _is_bad_localization_box(
                            mapped,
                            original_img.size,
                            quality_refined.display_kind,
                        ):
                            best_box_pixels = expand_localization_box(
                                mapped,
                                original_img.size,
                                quality_refined.display_kind,
                            )
                            best_localization = quality_refined
                            best_confidence = max(best_confidence, quality_refined.confidence)
                            skipped_refine = False
                            quality_recrop_applied = True

    except Exception as e:
        logger.exception("localizer_exception %s error=%s", log_context, e)
        primary_localization = LocalizationResult(
            found=False,
            confidence=0.0,
            display_kind="unknown",
            reason=f"Localization failed: {e}",
        )
        best_localization = primary_localization

    display_kind = best_localization.display_kind if best_localization else "unknown"
    local_decode: LocalDecodeResult | None = None
    verification: ReadVerificationResult | None = None
    support_result: ReadingResult | None = None
    decimal_verification: DecimalVerificationResult | None = None
    decimal_candidate_texts: list[str] = []
    decimal_resolved_text: str | None = None

    if best_box_pixels is not None and best_localization is not None:
        crop_img = crop_from_box(original_img, best_box_pixels)
        crop_diagnostics = _safe_analyze_crop_diagnostics(
            crop_img,
            fallback_mode=display_kind,
        )
        debug_img = draw_localization_debug(
            original_img,
            best_box_pixels,
            display_kind,
            best_localization.reason[:40],
        )
        _box_cy = round(
            ((best_box_pixels[1] + best_box_pixels[3]) / 2.0) / max(original_img.size[1], 1),
            3,
        )
        localization_payload = {
            "source": "vlm_quality_recrop" if quality_recrop_applied else ("vlm_pass1" if skipped_refine else "vlm_pass2"),
            "model": LOCALIZER_MODEL_NAME,
            "found": True,
            "confidence": round(best_confidence, 3),
            "display_kind": display_kind,
            "reason": best_localization.reason,
            "skipped_refine": skipped_refine,
            "quality_recrop_attempted": quality_recrop_attempted,
            "quality_recrop_applied": quality_recrop_applied,
            "box_cy_frac": _box_cy,
            "box_norm_1000": pixels_to_norm1000(best_box_pixels, original_img.size),
            "box_pixels": {
                "x1": best_box_pixels[0],
                "y1": best_box_pixels[1],
                "x2": best_box_pixels[2],
                "y2": best_box_pixels[3],
            },
        }
        local_decode = _maybe_run_local_decoder(crop_img, crop_diagnostics)
        logger.info(
            "crop_diagnostics %s reliable=%s mode=%s quality_score=%.3f reason=%s",
            log_context,
            crop_diagnostics.is_reliable,
            crop_diagnostics.mode,
            crop_diagnostics.quality_score,
            crop_diagnostics.reason,
        )
        if local_decode is not None:
            logger.info(
                "local_decoder_result %s ok=%s value_text=%s confidence=%.3f reason=%s",
                log_context,
                local_decode.ok,
                local_decode.value_text,
                local_decode.confidence,
                local_decode.reason,
            )
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
            "quality_recrop_attempted": False,
            "quality_recrop_applied": False,
        }
        crop_diagnostics = CropDiagnostics(
            is_reliable=False,
            mode=display_kind if display_kind in {"led", "lcd"} else "unknown",
            quality_score=0.0,
            lit_ratio=0.0,
            green_ratio=0.0,
            component_count=0,
            active_span_ratio=0.0,
            active_band_height_ratio=0.0,
            leading_blank_ratio=0.0,
            reason="Crop diagnostics unavailable because localization fell back to the full image.",
        )
    logger.info(
        "localization_selected %s found=%s source=%s display_kind=%s confidence=%s",
        log_context,
        localization_payload["found"],
        localization_payload["source"],
        localization_payload["display_kind"],
        localization_payload["confidence"],
    )

    if best_box_pixels is not None:
        logger.info(
            "reader_start %s mode=crop display_kind=%s",
            log_context,
            display_kind,
        )
        primary_result = call_gemini_with_instructions(
            render_prompt_template(
                "crop_read_overlay.txt",
                {
                    "DISPLAY_KIND_LABEL": _display_kind_label(display_kind),
                },
            ),
            crop_img,
            original_img,
            model_name=MODEL_NAME,
            primary_source="crop",
            include_secondary=True,
        )
    else:
        logger.info(
            "reader_start %s mode=full_image display_kind=%s",
            log_context,
            display_kind,
        )
        primary_result = call_gemini_on_full_image(
            original_img,
            localization_payload.get("reason", "localization failed"),
            display_kind=display_kind,
        )

    primary_suspicious = is_suspicious_read(
        primary_result,
        crop_diagnostics,
        expected_digit_count=None,
    )
    if is_suspicious_read(
        primary_result,
        crop_diagnostics,
        expected_digit_count=None,
    ):
        logger.warning(
            "reader_suspicious %s value_text=%s reason=%s",
            log_context,
            primary_result.value_text,
            primary_result.reason,
        )

    verification_primary_img = crop_img if best_box_pixels is not None else original_img
    try:
        verification = call_gemini_read_verifier(
            verification_primary_img,
            original_img,
            primary_candidate=primary_result.value_text,
            display_kind=display_kind,
            used_full_image_fallback=best_box_pixels is None,
        )
        logger.info(
            "reader_verification %s verdict=%s suggested_value_text=%s confidence=%.3f agrees_with_primary=%s edge_clipped=%s decimal_visible=%s leftmost_digit_fully_visible=%s reason=%s",
            log_context,
            verification.verdict,
            verification.suggested_value_text,
            verification.confidence,
            verification.agrees_with_primary,
            verification.crop_edge_clipped,
            verification.decimal_visible,
            verification.leftmost_digit_fully_visible,
            verification.reason,
        )
    except Exception as e:
        logger.warning("reader_verification_exception %s error=%s", log_context, e)

    verifier_candidate_text = (
        verification.suggested_value_text
        if verification is not None and verification.suggested_value_text
        else None
    )
    decimal_candidate_texts, decimal_signature = _decimal_signature_for_verification(
        primary_result.value_text if primary_result.status == "ok" else None,
        verifier_candidate_text,
        None,
        display_kind=display_kind,
    )
    if decimal_signature is not None:
        try:
            decimal_primary_img = crop_img if best_box_pixels is not None else original_img
            decimal_enhanced_img = make_enhanced_display_image(
                decimal_primary_img,
                max_dim=MAX_DIMENSION,
            )
            decimal_verification = call_gemini_decimal_verifier(
                decimal_primary_img,
                decimal_enhanced_img,
                original_img,
                candidate_texts=decimal_candidate_texts,
                display_kind=display_kind,
            )
            decimal_resolved_text = (
                _rebuild_decimal_text(decimal_signature, decimal_verification.decimal_after_digit)
                if decimal_verification.decimal_visible == "yes"
                and decimal_verification.dot_is_real_not_glare
                else None
            )
            logger.info(
                "decimal_verification %s visible=%s after_digit=%s real_dot=%s confidence=%.3f resolved_value_text=%s reason=%s",
                log_context,
                decimal_verification.decimal_visible,
                decimal_verification.decimal_after_digit,
                decimal_verification.dot_is_real_not_glare,
                decimal_verification.confidence,
                decimal_resolved_text,
                decimal_verification.reason,
            )
        except Exception as e:
            logger.warning("decimal_verification_exception %s error=%s", log_context, e)

    support_read_needed = (
        best_box_pixels is not None
        and (
            primary_suspicious
            or _is_led_single_digit_risk(
                primary_result,
                crop_diagnostics,
                display_kind=display_kind,
            )
            or (
                verification is not None
                and (
                    verification.crop_edge_clipped
                    or not verification.agrees_with_primary
                    or verification.verdict != "confirmed"
                )
            )
            or (
                decimal_verification is not None
                and (
                    decimal_verification.decimal_visible == "uncertain"
                    or decimal_resolved_text is None and len(decimal_candidate_texts) >= 2
                )
            )
        )
    )
    if support_read_needed:
        try:
            logger.info(
                "support_read_start %s display_kind=%s",
                log_context,
                display_kind,
            )
            support_result = call_gemini_on_full_image(
                original_img,
                "Crop read required full-image support during final consensus resolution.",
                display_kind=display_kind,
            )
            logger.info(
                "support_read_result %s status=%s value_text=%s confidence=%.3f reason=%s",
                log_context,
                support_result.status,
                support_result.value_text,
                support_result.confidence,
                support_result.reason,
            )
        except Exception as e:
            logger.warning("support_read_exception %s error=%s", log_context, e)

    post_support_candidate_texts, post_support_signature = _decimal_signature_for_verification(
        primary_result.value_text if primary_result.status == "ok" else None,
        verifier_candidate_text,
        support_result.value_text if support_result is not None and support_result.status == "ok" else None,
        display_kind=display_kind,
    )
    if len(post_support_candidate_texts) > len(decimal_candidate_texts):
        decimal_candidate_texts = post_support_candidate_texts
    if (
        post_support_signature is not None
        and (
            decimal_verification is None
            or (
                decimal_resolved_text is None
                and len(post_support_candidate_texts) >= len(decimal_candidate_texts)
            )
        )
    ):
        try:
            decimal_primary_img = crop_img if best_box_pixels is not None else original_img
            decimal_enhanced_img = make_enhanced_display_image(
                decimal_primary_img,
                max_dim=MAX_DIMENSION,
            )
            decimal_verification = call_gemini_decimal_verifier(
                decimal_primary_img,
                decimal_enhanced_img,
                original_img,
                candidate_texts=decimal_candidate_texts,
                display_kind=display_kind,
            )
            decimal_resolved_text = (
                _rebuild_decimal_text(post_support_signature, decimal_verification.decimal_after_digit)
                if decimal_verification.decimal_visible == "yes"
                and decimal_verification.dot_is_real_not_glare
                else None
            )
            logger.info(
                "decimal_verification_post_support %s visible=%s after_digit=%s real_dot=%s confidence=%.3f resolved_value_text=%s reason=%s",
                log_context,
                decimal_verification.decimal_visible,
                decimal_verification.decimal_after_digit,
                decimal_verification.dot_is_real_not_glare,
                decimal_verification.confidence,
                decimal_resolved_text,
                decimal_verification.reason,
            )
        except Exception as e:
            logger.warning("decimal_verification_post_support_exception %s error=%s", log_context, e)

    if not decimal_candidate_texts:
        decimal_candidate_texts, decimal_signature = _decimal_signature_for_verification(
            primary_result.value_text if primary_result.status == "ok" else None,
            verifier_candidate_text,
            support_result.value_text if support_result is not None and support_result.status == "ok" else None,
            display_kind=display_kind,
        )

    result = _resolve_verified_result(
        primary_result,
        crop_diagnostics=crop_diagnostics,
        local_decode=local_decode,
        verification=verification,
        support_result=support_result,
        decimal_verification=decimal_verification,
        decimal_candidate_texts=decimal_candidate_texts,
        decimal_resolved_text=decimal_resolved_text,
        display_kind=display_kind,
        used_full_image_fallback=best_box_pixels is None,
    )
    result = _apply_target_decimal_contract(
        result,
        target_key=target_key,
    )

    if is_suspicious_read(
        result,
        crop_diagnostics,
        expected_digit_count=None,
    ):
        if result.status == "ok" and result.value_text:
            result.confidence = min(
                result.confidence,
                0.62 if verification is not None or local_decode is not None or support_result is not None else 0.52,
            )
            result.reason = (
                f"{result.reason} Final read remained partially ambiguous after verification; "
                "returning the best-supported value."
            )
        else:
            fallback_result = _best_ok_result(support_result, primary_result)
            if fallback_result is not None:
                result = fallback_result
                result.confidence = min(result.confidence, 0.62)
                result.reason = (
                    f"{result.reason} Suspicious resolution fell back to the best available numeric candidate."
                )
            else:
                result = mark_suspicious_for_review(
                    result,
                    f"Read flagged as suspicious after verification: {result.reason}",
                )

    t_elapsed = round(time.monotonic() - t_start, 2)

    _LAST_PREVIEWS["original"] = pil_to_jpeg_bytes(original_img)
    _LAST_PREVIEWS["enhanced"] = pil_to_jpeg_bytes(enhanced_full_img)
    _LAST_PREVIEWS["crop"] = pil_to_jpeg_bytes(crop_img)
    _LAST_PREVIEWS["debug"] = pil_to_jpeg_bytes(debug_img)
    logger.info(
        "pipeline_complete %s status=%s value_text=%s confidence=%.3f elapsed_seconds=%.2f",
        log_context,
        result.status,
        result.value_text,
        result.confidence,
        t_elapsed,
    )

    return {
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
        "local_decoder": _serialize_local_decode(local_decode),
        "verification": _serialize_read_verification(verification),
        "decimal_verification": _serialize_decimal_verification(
            decimal_verification,
            candidate_texts=decimal_candidate_texts,
            resolved_value_text=decimal_resolved_text,
        ),
        "support_read": _serialize_optional_reading_result(support_result),
        "elapsed_seconds": t_elapsed,
        "preview_urls": {
            "original": "/api/preview/original",
            "enhanced": "/api/preview/enhanced",
            "crop": "/api/preview/crop",
            "debug": "/api/preview/debug",
        },
    }


def _log_writeback_target_decisions(
    *,
    trace_id: str,
    submission_id: Optional[str],
    final_results: dict[str, dict[str, Any]],
    target_field_map: dict[str, str],
    skipped_targets: dict[str, Any],
) -> None:
    for target_key in sorted(final_results.keys()):
        final = final_results.get(target_key) or {}
        skipped = skipped_targets.get(target_key, {})
        resolved_destination_field = target_field_map.get(target_key) or skipped.get("destination_field") or _field_config_for_target(target_key).get("value")
        numeric_value = final.get("value", final.get("value_number"))
        logger.info(
            "clappia_writeback_target trace_id=%s submission_id=%s target=%s destination_field=%s final_status=%s numeric_value=%s included=%s skipped_reason=%s",
            trace_id,
            submission_id,
            target_key,
            resolved_destination_field,
            final.get("status"),
            numeric_value,
            target_key in target_field_map,
            skipped.get("skipped_reason"),
        )


def _build_clappia_job_payload(
    *,
    trace_id: str,
    payload: ClappiaAnalyzeRequest,
    targets: dict[str, str],
    clappia_input_payload: dict[str, Any],
) -> ClappiaJobPayload:
    return ClappiaJobPayload(
        job_id=_trace_id("mfcjob"),
        trace_id=trace_id,
        submitted_at=time.time(),
        submission_id=payload.submission_id,
        workplace_id=payload.workplace_id,
        requesting_user_email_address=payload.requesting_user_email_address,
        targets=targets,
        clappia_input_payload=clappia_input_payload,
    )


def _build_async_clappia_response(
    *,
    job_payload: ClappiaJobPayload,
    enqueue_result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "ok": True,
        "trace_id": job_payload.trace_id,
        "mode": "async",
        "submission_id": job_payload.submission_id,
        "request_targets_received": sorted(job_payload.targets.keys()),
        "processed_targets": [],
        "results": {},
        "writeback_attempted": False,
        "writeback_status": "queued",
        "writeback_summary": _sanitize_for_log(enqueue_result),
        "clappia_input_payload": job_payload.clappia_input_payload,
        "clappia_writeback_payload": {},
        "elapsed_seconds": 0.0,
    }


async def _process_clappia_job(job_payload: ClappiaJobPayload) -> dict[str, Any]:
    t_start = time.monotonic()
    trace_id = job_payload.trace_id
    submission_id = job_payload.submission_id
    targets = {
        str(key).strip(): value
        for key, value in job_payload.targets.items()
        if str(key).strip()
    }
    if not targets:
        raise HTTPException(status_code=400, detail="No analyzable targets available for processing.")

    logger.info(
        "clappia_parallel_analysis_start trace_id=%s submission_id=%s target_count=%s concurrency=%s",
        trace_id,
        submission_id,
        len(targets),
        CLAPPIA_ANALYZE_CONCURRENCY,
    )
    semaphore = asyncio.Semaphore(CLAPPIA_ANALYZE_CONCURRENCY)
    target_tasks = [
        asyncio.create_task(
            _analyze_clappia_target(
                trace_id=trace_id,
                submission_id=submission_id,
                target_key=target_key,
                image_url=image_url,
                semaphore=semaphore,
            )
        )
        for target_key, image_url in targets.items()
    ]
    target_results = await asyncio.gather(*target_tasks)
    logger.info(
        "clappia_parallel_analysis_complete trace_id=%s submission_id=%s target_count=%s",
        trace_id,
        submission_id,
        len(target_results),
    )

    final_results: dict[str, dict[str, Any]] = {}
    for target_key, single_result, _ in target_results:
        final_results[target_key] = single_result

    writeback_data, target_field_map, skipped_targets = build_clappia_writeback_data(final_results)
    _log_writeback_target_decisions(
        trace_id=trace_id,
        submission_id=submission_id,
        final_results=final_results,
        target_field_map=target_field_map,
        skipped_targets=skipped_targets,
    )
    clappia_writeback = update_clappia_submission(
        trace_id=trace_id,
        submission_id=submission_id,
        data=writeback_data,
        target_field_map=target_field_map,
        skipped_targets=skipped_targets,
        workplace_id=job_payload.workplace_id,
        requesting_user_email_address=job_payload.requesting_user_email_address,
    )

    processed_targets = sorted(final_results.keys())
    partial_failures = sorted(
        target_key
        for target_key, result in final_results.items()
        if result.get("status") != "ok"
    )
    if partial_failures or skipped_targets:
        logger.info(
            "clappia_partial_failures trace_id=%s submission_id=%s failed_targets=%s skipped_writeback_targets=%s",
            trace_id,
            submission_id,
            partial_failures,
            sorted(skipped_targets.keys()),
        )

    response_payload = {
        "ok": True,
        "trace_id": trace_id,
        "mode": "sync",
        "submission_id": submission_id,
        "request_targets_received": sorted(targets.keys()),
        "processed_targets": processed_targets,
        "results": _build_per_target_response(
            final_results,
            target_field_map=target_field_map,
            skipped_targets=skipped_targets,
        ),
        "writeback_attempted": clappia_writeback.attempted,
        "writeback_status": _writeback_status_from_result(clappia_writeback),
        "writeback_summary": _build_clappia_writeback_summary(clappia_writeback),
        "clappia_input_payload": job_payload.clappia_input_payload,
        "clappia_writeback_payload": _sanitize_for_log(clappia_writeback.payload),
        "elapsed_seconds": round(time.monotonic() - t_start, 2),
    }
    logger.info(
        "clappia_response_payload trace_id=%s submission_id=%s response=%s",
        trace_id,
        submission_id,
        json.dumps(_sanitize_for_log(response_payload), sort_keys=True),
    )
    logger.info(
        "clappia_request_complete trace_id=%s submission_id=%s processed_targets=%s writeback_status=%s writeback_attempted=%s elapsed_seconds=%.2f",
        trace_id,
        submission_id,
        processed_targets,
        response_payload["writeback_status"],
        response_payload["writeback_attempted"],
        response_payload["elapsed_seconds"],
    )
    return response_payload


def _enqueue_clappia_job(job_payload: ClappiaJobPayload) -> dict[str, Any]:
    if not MFC_QUEUE.is_configured():
        raise HTTPException(
            status_code=503,
            detail="Async mode is enabled but REDIS_URL is not configured.",
        )

    logger.info(
        "mfc_queue_enqueue_start trace_id=%s submission_id=%s queue_name=%s jobs_key=%s failed_key=%s",
        job_payload.trace_id,
        job_payload.submission_id,
        MFC_QUEUE_NAME,
        MFC_QUEUE.config.jobs_key,
        MFC_QUEUE.config.failed_key,
    )
    try:
        enqueue_result = MFC_QUEUE.enqueue(job_payload.model_dump())
        logger.info(
            "mfc_queue_enqueue_complete trace_id=%s submission_id=%s result=%s",
            job_payload.trace_id,
            job_payload.submission_id,
            json.dumps(_sanitize_for_log(enqueue_result), sort_keys=True),
        )
        return enqueue_result
    except Exception as e:
        logger.exception(
            "mfc_queue_enqueue_exception trace_id=%s submission_id=%s error=%s",
            job_payload.trace_id,
            job_payload.submission_id,
            e,
        )
        raise HTTPException(
            status_code=503,
            detail=f"Could not enqueue Clappia job: {e}",
        )


def run_clappia_worker() -> None:
    if not MFC_QUEUE.is_configured():
        raise RuntimeError("REDIS_URL is required to run the MFC worker.")

    _run_clappia_worker_loop(runtime_mode="worker")


def _run_clappia_worker_loop(
    *,
    runtime_mode: str,
    stop_event: threading.Event | None = None,
) -> None:
    logger.info(
        "mfc_worker_start runtime_mode=%s queue_name=%s failed_queue_name=%s namespace=%s jobs_key=%s processing_key=%s failed_key=%s writeback_mode=%s",
        runtime_mode,
        MFC_QUEUE_NAME,
        MFC_FAILED_QUEUE_NAME,
        MFC_REDIS_NAMESPACE,
        MFC_QUEUE.config.jobs_key,
        MFC_QUEUE.config.processing_key,
        MFC_QUEUE.config.failed_key,
        MFC_WRITEBACK_MODE,
    )
    try:
        MFC_QUEUE.ping()
    except Exception as e:
        logger.exception(
            "mfc_worker_start_failed runtime_mode=%s queue_name=%s error=%s",
            runtime_mode,
            MFC_QUEUE_NAME,
            e,
        )
        return
    while stop_event is None or not stop_event.is_set():
        try:
            dequeued = MFC_QUEUE.dequeue(timeout_seconds=5)
        except Exception as e:
            if stop_event is not None and stop_event.is_set():
                break
            logger.exception("mfc_worker_dequeue_exception runtime_mode=%s error=%s", runtime_mode, e)
            time.sleep(1.0)
            continue

        if dequeued is None:
            continue

        raw_job = dequeued["raw_job"]
        logger.info(
            "mfc_queue_dequeue trace_id=%s raw_job_bytes=%s queue_name=%s processing_key=%s",
            dequeued["job"].get("trace_id"),
            len(raw_job.encode("utf-8")),
            MFC_QUEUE_NAME,
            MFC_QUEUE.config.processing_key,
        )
        job_payload = ClappiaJobPayload.model_validate(dequeued["job"])
        t_start = time.monotonic()
        logger.info(
            "mfc_worker_job_start trace_id=%s submission_id=%s job_id=%s queue_name=%s",
            job_payload.trace_id,
            job_payload.submission_id,
            job_payload.job_id,
            MFC_QUEUE_NAME,
        )

        try:
            response_payload = asyncio.run(_process_clappia_job(job_payload))
            MFC_QUEUE.complete(raw_job=raw_job, job_payload=job_payload.model_dump())
            logger.info(
                "mfc_worker_job_complete trace_id=%s submission_id=%s job_id=%s elapsed_seconds=%.2f writeback_status=%s",
                job_payload.trace_id,
                job_payload.submission_id,
                job_payload.job_id,
                time.monotonic() - t_start,
                response_payload["writeback_status"],
            )
        except Exception as e:
            failure_record = MFC_QUEUE.fail(
                raw_job=raw_job,
                job_payload=job_payload.model_dump(),
                failure_payload={
                    "trace_id": job_payload.trace_id,
                    "submission_id": job_payload.submission_id,
                    "job_id": job_payload.job_id,
                    "error": str(e),
                },
            )
            logger.exception(
                "mfc_worker_job_failed trace_id=%s submission_id=%s job_id=%s elapsed_seconds=%.2f failure=%s",
                job_payload.trace_id,
                job_payload.submission_id,
                job_payload.job_id,
                time.monotonic() - t_start,
                json.dumps(_sanitize_for_log(failure_record), sort_keys=True),
            )
    logger.info(
        "mfc_worker_stop runtime_mode=%s queue_name=%s",
        runtime_mode,
        MFC_QUEUE_NAME,
    )


def _start_embedded_clappia_worker() -> None:
    global _EMBEDDED_WORKER_THREAD, _EMBEDDED_WORKER_STOP_EVENT

    if MFC_WRITEBACK_MODE != "async":
        return
    if not MFC_QUEUE.is_configured():
        logger.warning(
            "embedded_worker_not_started reason=redis_not_configured writeback_mode=%s",
            MFC_WRITEBACK_MODE,
        )
        return
    if _EMBEDDED_WORKER_THREAD is not None and _EMBEDDED_WORKER_THREAD.is_alive():
        logger.info(
            "embedded_worker_already_running queue_name=%s namespace=%s",
            MFC_QUEUE_NAME,
            MFC_REDIS_NAMESPACE,
        )
        return

    stop_event = threading.Event()
    worker_thread = threading.Thread(
        target=_run_clappia_worker_loop,
        kwargs={"runtime_mode": "embedded", "stop_event": stop_event},
        name="mfc-clappia-embedded-worker",
        daemon=True,
    )
    _EMBEDDED_WORKER_STOP_EVENT = stop_event
    _EMBEDDED_WORKER_THREAD = worker_thread
    worker_thread.start()
    logger.info(
        "embedded_worker_started queue_name=%s namespace=%s jobs_key=%s processing_key=%s failed_key=%s",
        MFC_QUEUE_NAME,
        MFC_REDIS_NAMESPACE,
        MFC_QUEUE.config.jobs_key,
        MFC_QUEUE.config.processing_key,
        MFC_QUEUE.config.failed_key,
    )


def _stop_embedded_clappia_worker() -> None:
    global _EMBEDDED_WORKER_THREAD, _EMBEDDED_WORKER_STOP_EVENT

    if _EMBEDDED_WORKER_STOP_EVENT is not None:
        _EMBEDDED_WORKER_STOP_EVENT.set()
    if _EMBEDDED_WORKER_THREAD is not None:
        _EMBEDDED_WORKER_THREAD.join(timeout=6.0)
    _EMBEDDED_WORKER_THREAD = None
    _EMBEDDED_WORKER_STOP_EVENT = None


@app.get("/")
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.on_event("startup")
def startup_log_runtime_config() -> None:
    logger.info(
        "startup_runtime_config runtime_mode=web log_level=%s writeback_mode=%s queue_name=%s failed_queue_name=%s namespace=%s redis_configured=%s redis_namespace=%s",
        LOG_LEVEL,
        MFC_WRITEBACK_MODE,
        MFC_QUEUE_NAME,
        MFC_FAILED_QUEUE_NAME,
        MFC_REDIS_NAMESPACE,
        MFC_QUEUE.is_configured(),
        json.dumps(MFC_QUEUE.describe(), sort_keys=True),
    )
    logger.info(
        "startup_clappia_field_config mapping=%s",
        json.dumps(_compact_clappia_field_mapping(), sort_keys=True),
    )
    if MFC_QUEUE.is_configured():
        try:
            MFC_QUEUE.ping()
            logger.info(
                "startup_redis_ready runtime_mode=web queue_name=%s namespace=%s jobs_key=%s processing_key=%s failed_key=%s",
                MFC_QUEUE_NAME,
                MFC_REDIS_NAMESPACE,
                MFC_QUEUE.config.jobs_key,
                MFC_QUEUE.config.processing_key,
                MFC_QUEUE.config.failed_key,
            )
        except Exception as e:
            logger.warning(
                "startup_redis_unavailable queue_name=%s error=%s",
                MFC_QUEUE_NAME,
                e,
            )
    _start_embedded_clappia_worker()


@app.on_event("shutdown")
def shutdown_embedded_worker() -> None:
    _stop_embedded_clappia_worker()


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "log_level": LOG_LEVEL,
        "mfc_writeback_mode": MFC_WRITEBACK_MODE,
        "redis_configured": MFC_QUEUE.is_configured(),
        "mfc_queue_name": MFC_QUEUE_NAME,
        "mfc_failed_queue_name": MFC_FAILED_QUEUE_NAME,
        "redis_namespace": MFC_QUEUE.describe(),
        "embedded_worker_running": bool(
            _EMBEDDED_WORKER_THREAD is not None and _EMBEDDED_WORKER_THREAD.is_alive()
        ),
        "clappia_analyze_concurrency": CLAPPIA_ANALYZE_CONCURRENCY,
        "clappia_app_id": CLAPPIA_APP_ID or None,
        "clappia_workplace_id_configured": bool(_normalize_clappia_token(CLAPPIA_WORKPLACE_ID)),
        "clappia_base_url": CLAPPIA_BASE_URL,
        "clappia_writeback_enabled": bool(
            CLAPPIA_API_KEY
            and CLAPPIA_APP_ID
            and _normalize_clappia_token(CLAPPIA_WORKPLACE_ID)
        ),
        "model": MODEL_NAME,
        "localizer_model": LOCALIZER_MODEL_NAME,
        "localizer_min_confidence": LOCALIZER_MIN_CONFIDENCE,
        "localizer_skip_refine_threshold": LOCALIZER_SKIP_REFINE_THRESHOLD,
        "localizer_max_dimension": LOCALIZER_MAX_DIMENSION,
        "local_decoder_min_confidence": LOCAL_DECODER_MIN_CONFIDENCE,
        "expected_decimals": EXPECTED_DECIMALS or None,
        "clappia_endpoint": "/api/clappia/analyze",
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
    trace_id = _trace_id("upload")
    logger.info(
        "upload_request_received trace_id=%s filename=%s content_type=%s",
        trace_id,
        file.filename,
        file.content_type,
    )
    data = await file.read()
    payload = run_scale_reader_pipeline(
        data,
        content_type=file.content_type or "image/jpeg",
        trace_id=trace_id,
        source="upload",
    )
    logger.info(
        "upload_request_complete trace_id=%s status=%s value_text=%s",
        trace_id,
        payload["final"]["status"],
        payload["final"]["value_text"],
    )
    return JSONResponse(content=payload)

@app.post("/api/clappia/analyze")
async def clappia_analyze(request: Request) -> JSONResponse:
    trace_id = _trace_id("clappia")
    try:
        raw_payload = await request.json()
    except Exception as e:
        logger.warning("clappia_request_invalid_json trace_id=%s error=%s", trace_id, e)
        raise HTTPException(status_code=400, detail=f"Invalid JSON body: {e}")

    if not isinstance(raw_payload, dict):
        logger.warning(
            "clappia_request_invalid_shape trace_id=%s payload_type=%s",
            trace_id,
            type(raw_payload).__name__,
        )
        raise HTTPException(status_code=400, detail="Request body must be a JSON object.")

    try:
        payload = ClappiaAnalyzeRequest.model_validate(raw_payload)
    except ValidationError as e:
        logger.warning("clappia_request_invalid_payload trace_id=%s error=%s", trace_id, e)
        raise HTTPException(status_code=400, detail=json.loads(e.json()))

    targets = extract_clappia_targets(payload)
    sanitized_raw_payload = _sanitize_for_log(raw_payload)
    sanitized_input_payload = _build_sanitized_clappia_input_payload(
        payload=payload,
        targets=targets,
    )
    logger.info(
        "clappia_request_received trace_id=%s submission_id=%s payload=%s",
        trace_id,
        payload.submission_id,
        json.dumps(sanitized_raw_payload, sort_keys=True),
    )
    logger.info(
        "clappia_targets_normalized trace_id=%s submission_id=%s targets=%s",
        trace_id,
        payload.submission_id,
        json.dumps(_sanitize_for_log(targets), sort_keys=True),
    )
    if not targets:
        logger.warning(
            "clappia_request_rejected trace_id=%s submission_id=%s reason=no_analyzable_targets",
            trace_id,
            payload.submission_id,
        )
        raise HTTPException(
            status_code=400,
            detail=(
                "No analyzable image URLs found. Send image URLs either under "
                "'targets' or as top-level fields. Example: "
                "{\"targets\":{\"gross_weight\":\"https://...\"}} or "
                "{\"gross_weight\":\"https://...\"}."
            ),
        )

    job_payload = _build_clappia_job_payload(
        trace_id=trace_id,
        payload=payload,
        targets=targets,
        clappia_input_payload=sanitized_input_payload,
    )

    if MFC_WRITEBACK_MODE == "async":
        enqueue_result = _enqueue_clappia_job(job_payload)
        response_payload = _build_async_clappia_response(
            job_payload=job_payload,
            enqueue_result=enqueue_result,
        )
        logger.info(
            "clappia_response_payload trace_id=%s submission_id=%s response=%s",
            trace_id,
            payload.submission_id,
            json.dumps(_sanitize_for_log(response_payload), sort_keys=True),
        )
        return JSONResponse(content=response_payload)

    response_payload = await _process_clappia_job(job_payload)
    return JSONResponse(content=response_payload)

if __name__ == "__main__":
    import uvicorn
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        run_clappia_worker()
    else:
        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("PORT", "8000"))
        uvicorn.run("app:app", host=host, port=port, reload=True)
