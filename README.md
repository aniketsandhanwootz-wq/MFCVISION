# MFCVISION

FastAPI service for reading numeric values from photos of weighing scales and similar LED/LCD instrument displays using Gemini.

The app supports two input paths:

- direct image upload through `POST /api/read-scale`
- remote image URL analysis through `POST /api/clappia/analyze`

## What The Service Does

For each image, the backend tries to:

1. locate the display window
2. read the numeric value from the crop
3. fall back to the full image when localization is weak or invalid
4. mark suspicious outputs as `needs_review`

The latest original, enhanced, crop, and debug images are cached in memory for the web UI and preview endpoints.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set at least:

```env
GEMINI_API_KEY=your_api_key_here
```

Run the web app:

```bash
python app.py
```

Equivalent explicit web command:

```bash
uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000}
```

Run the worker when `MFC_WRITEBACK_MODE=async`:

```bash
python app.py worker
```

Open:

```text
http://127.0.0.1:8000
```

## Docker

Build:

```bash
docker build -t mfcvision .
```

Run the web container:

```bash
docker run --rm -p 8000:10000 --env-file .env mfcvision
```

The container starts Uvicorn on `0.0.0.0` and uses `PORT` if provided, otherwise `10000`.

For a separate worker process on Render, reuse the same image and override the start command to:

```bash
python app.py worker
```

## Current Runtime Flow

Both `/api/read-scale` and `/api/clappia/analyze` use the same internal image-analysis pipeline.

1. Validate the image content type and size.
2. Open the image with Pillow.
3. Resize the original image for localization and build an enhanced helper image.
4. Ask Gemini for a display bounding box.
5. If the first box is usable but not strong enough, run a second region-refinement pass.
6. If a valid box exists, read from the crop with the full image as context.
7. If localization fails, read directly from the full image.
8. Post-process the result, reject suspicious numeric shapes, and return `ok` or `needs_review`.
9. In `sync` mode, map successful numeric outputs and write them back directly to the same Clappia submission before responding.
10. In `async` mode, enqueue the normalized Clappia request into Redis and let a worker run the same analysis and writeback flow later.

## Repository Layout

- `app.py`: FastAPI app, Gemini calls, Clappia route, worker entrypoint, preview endpoints
- `mfc_queue.py`: Redis queue helper with isolated `mfc_clappia:*` namespace
- `image_enhance.py`: conservative enhancement used for preview and localization context
- `vision.py`: image utilities plus crop-analysis and seven-segment helpers not currently active in the live read path
- `prompts/scale_reader.txt`: reader prompt used for numeric extraction
- `static/index.html`: upload UI and debug preview page
- `requirements.txt`: Python dependency pins
- `Dockerfile`: container image build for deployment

## Environment Variables

`app.py` is the source of truth for runtime defaults.

- `GEMINI_API_KEY`: required
- `MODEL_NAME`: reader model, default `gemini-2.5-flash`
- `LOCALIZER_MODEL_NAME`: localizer model, default `gemini-2.5-flash`
- `READ_TEMPERATURE`: default `0.0`
- `LOCALIZER_TEMPERATURE`: default `0.0`
- `LOG_LEVEL`: app log level for Render/stdout logs, default `INFO`
- `REDIS_URL`: Render Key Value Redis connection string, required only when `MFC_WRITEBACK_MODE=async`
- `MFC_QUEUE_NAME`: logical queue name for this service, default `mfc_clappia_jobs`
- `MFC_FAILED_QUEUE_NAME`: logical failed queue name for this service, default `mfc_clappia_failed`
- `MFC_REDIS_NAMESPACE`: Redis key prefix used for this service only, default `mfc_clappia`
- `MFC_WRITEBACK_MODE`: `sync` or `async`, default `sync`
- `CLAPPIA_API_KEY`: optional workplace API key used for backend-side Clappia submission writeback
- `CLAPPIA_APP_ID`: Clappia app ID for writeback, default `MFC182090`
- `CLAPPIA_WORKPLACE_ID`: Clappia workplace ID required by the public `submissions/edit` API
- `CLAPPIA_BASE_URL`: Clappia public API base URL, default `https://api-public-v3.clappia.com`
- `CLAPPIA_ANALYZE_CONCURRENCY`: bounded parallel target-analysis concurrency for `/api/clappia/analyze`, default `6`
- `HOST`: default `0.0.0.0`
- `PORT`: default `8000`
- `MAX_IMAGE_MB`: upload or remote image size limit, default `10`
- `MAX_DIMENSION`: max size for the full-image enhanced preview, default `1200`
- `LOCALIZER_MAX_DIMENSION`: max size sent to the localizer path, default `1024`
- `LOCALIZER_MIN_CONFIDENCE`: minimum accepted localization confidence, default `0.45`
- `LOCALIZER_SKIP_REFINE_THRESHOLD`: skip the second localization pass above this confidence, default `0.90`
- `LOCAL_DECODER_MIN_CONFIDENCE`: retained setting for local decoder helpers, default `0.90`
- `EXPECTED_DECIMALS`: optional fixed decimal count for known devices
- `FIXED_DECIMALS`: alias for `EXPECTED_DECIMALS`

Important notes:

- If `LOCALIZER_MODEL_NAME` contains `lite`, the app replaces it with `MODEL_NAME`.
- `.env.example` is only a starter file and may lag `app.py`. Use the code as the source of truth.
- Missing `CLAPPIA_API_KEY` does not stop startup; it only disables direct Clappia writeback and returns a structured writeback error in `/api/clappia/analyze`.
- Missing `CLAPPIA_WORKPLACE_ID` also disables direct Clappia writeback; unresolved placeholders like `{workplaceId}` are ignored.
- Missing `REDIS_URL` does not stop startup in `sync` mode. In `async` mode, enqueue requests fail loudly until Redis is configured.

## Clappia Backend Writeback

When Clappia calls `POST /api/clappia/analyze`, the backend now:

1. logs the sanitized incoming request
2. normalizes the target image URLs
3. analyzes each target with the existing image-reading pipeline
4. maps successful numeric values to configured Clappia field keys
5. calls Clappia Public API `submissions/edit` from backend code

Recommended workflow simplification:

- Keep only the REST API step that calls this backend.
- Remove or disable the Clappia workflow-side `Edit Submission` step after verification.
- Do not rely on Clappia response-field mapping for writeback anymore.

Example backend writeback payload sent to Clappia:

```json
{
  "appId": "MFC182090",
  "workplaceId": "PIP121027",
  "submissionId": "QEF16211579",
  "data": {
    "pre_weight": 18.54,
    "pre_weight_1": 18.61
  }
}
```

## Redis Queue Modes

The same API route supports two runtime modes:

- `MFC_WRITEBACK_MODE=sync`: the request is processed inline and the response includes final per-target results plus writeback summary.
- `MFC_WRITEBACK_MODE=async`: the request is normalized, pushed into Redis, and the API returns immediately with queue metadata. A separate worker process later runs the same analysis and writeback path.

Raw Redis namespace used by this service:

- `${MFC_REDIS_NAMESPACE}:jobs`
- `${MFC_REDIS_NAMESPACE}:processing`
- `${MFC_REDIS_NAMESPACE}:failed`
- `${MFC_REDIS_NAMESPACE}:dedupe:<hash>`

This isolates MFCVISION from any other workload sharing the same Render Key Value Redis instance. The service never reads any queue outside the configured `${MFC_REDIS_NAMESPACE}:*` namespace.

## API

### `GET /`

Returns the static web UI.

### `GET /health`

Returns the current runtime settings snapshot.

Example:

```json
{
  "ok": true,
  "log_level": "INFO",
  "mfc_writeback_mode": "sync",
  "redis_configured": true,
  "mfc_queue_name": "mfc_clappia_jobs",
  "mfc_failed_queue_name": "mfc_clappia_failed",
  "redis_namespace": {
    "configured": true,
    "queue_name": "mfc_clappia_jobs",
    "failed_queue_name": "mfc_clappia_failed",
    "jobs_key": "mfc_clappia:jobs",
    "processing_key": "mfc_clappia:processing",
    "failed_key": "mfc_clappia:failed",
    "dedupe_prefix": "mfc_clappia:dedupe:",
    "dedupe_ttl_seconds": 1800
  },
  "clappia_analyze_concurrency": 6,
  "clappia_app_id": "MFC182090",
  "clappia_workplace_id_configured": true,
  "clappia_base_url": "https://api-public-v3.clappia.com",
  "clappia_writeback_enabled": true,
  "model": "gemini-2.5-flash",
  "localizer_model": "gemini-2.5-flash"
}
```

### `GET /api/preview/{kind}`

Returns the latest preview image for:

- `original`
- `enhanced`
- `crop`
- `debug`

These previews are process-local and only represent the latest analyzed image.

### `POST /api/read-scale`

Multipart upload endpoint with a single field named `file`.

Example:

```bash
curl -X POST \
  -F "file=@sample.jpg" \
  http://127.0.0.1:8000/api/read-scale
```

Response shape:

```json
{
  "final": {
    "status": "ok",
    "value_text": "18.540",
    "value_number": 18.54,
    "confidence": 0.96,
    "reason": "Read from localized crop.",
    "ignored_text_present": true
  },
  "localization": {
    "source": "vlm_pass1",
    "model": "gemini-2.5-flash",
    "found": true,
    "confidence": 0.88,
    "display_kind": "led",
    "reason": "Located the display window.",
    "skipped_refine": true
  },
  "elapsed_seconds": 1.74
}
```

### `POST /api/clappia/analyze`

JSON endpoint for analyzing one or more remote image URLs and writing successful numeric values back into Clappia from backend code.

Supported request shapes:

Nested `targets` object:

```json
{
  "submission_id": "sub_123",
  "targets": {
    "ai_pre_weight_1": "https://example.com/images/gross.jpg",
    "ai_pre_weight_2": "https://example.com/images/net.jpg"
  }
}
```

Flat top-level fields:

```json
{
  "submissionId": "sub_123",
  "ai_pre_weight_1": "https://example.com/images/gross.jpg",
  "ai_pre_weight_2": "https://example.com/images/net.jpg"
}
```

Sync mode response shape:

```json
{
  "ok": true,
  "trace_id": "clappia-8c06ec77",
  "mode": "sync",
  "submission_id": "sub_123",
  "request_targets_received": [
    "ai_pre_weight_1",
    "ai_pre_weight_2"
  ],
  "processed_targets": [
    "ai_pre_weight_1",
    "ai_pre_weight_2"
  ],
  "results": {
    "ai_pre_weight_1": {
      "value": 18.54,
      "value_text": "18.540",
      "status": "ok",
      "confidence": 0.96,
      "reason": "Read from localized crop.",
      "ignored_text_present": true,
      "writeback_field": "pre_weight",
      "included_in_writeback": true
    },
    "ai_pre_weight_2": {
      "value": null,
      "value_text": null,
      "status": "needs_review",
      "confidence": 0.0,
      "reason": "Image URL returned HTTP 404.",
      "ignored_text_present": false,
      "writeback_field": "pre_weight_1",
      "included_in_writeback": false,
      "writeback_skip_reason": "result status was not ok"
    }
  },
  "writeback_attempted": true,
  "writeback_status": "success",
  "writeback_summary": {
    "enabled": true,
    "attempted": true,
    "success": true,
    "response_status": 200,
    "written_fields": [
      "pre_weight"
    ],
    "target_field_map": {
      "ai_pre_weight_1": "pre_weight"
    },
    "skipped_targets": {
      "ai_pre_weight_2": {
        "destination_field": "pre_weight_1",
        "status": "needs_review",
        "reason": "Image URL returned HTTP 404.",
        "skipped_reason": "result status was not ok"
      }
    },
    "error": null
  },
  "clappia_input_payload": {
    "submission_id": "sub_123",
    "workplace_id": "PIP121027",
    "requesting_user_email_address": "su***@example.com",
    "targets": {
      "ai_pre_weight_1": "https://example.com/images/gross.jpg",
      "ai_pre_weight_2": "https://example.com/images/net.jpg"
    }
  },
  "clappia_writeback_payload": {
    "appId": "MFC182090",
    "submissionId": "sub_123",
    "workplaceId": "PIP121027",
    "data": {
      "pre_weight": 18.54
    }
  },
  "elapsed_seconds": 9.84
}
```

Async mode response shape:

```json
{
  "ok": true,
  "trace_id": "clappia-8c06ec77",
  "mode": "async",
  "submission_id": "sub_123",
  "request_targets_received": [
    "ai_pre_weight_1",
    "ai_pre_weight_2"
  ],
  "processed_targets": [],
  "results": {},
  "writeback_attempted": false,
  "writeback_status": "queued",
  "writeback_summary": {
    "enqueued": true,
    "duplicate": false,
    "job_id": "mfcjob-fd6f6f0e",
    "dedupe_hash": "ab12...",
    "queue_name": "mfc_clappia_jobs",
    "jobs_key": "mfc_clappia:jobs"
  },
  "clappia_input_payload": {
    "submission_id": "sub_123",
    "targets": {
      "ai_pre_weight_1": "https://example.com/images/gross.jpg",
      "ai_pre_weight_2": "https://example.com/images/net.jpg"
    }
  },
  "clappia_writeback_payload": {},
  "elapsed_seconds": 0.0
}
```

## Field Mapping

Backend-side writeback uses an explicit mapping block in `app.py` so Clappia field writes stay intentional.

Current configured mappings:

- `ai_pre_weight_1 -> pre_weight`
- `ai_pre_weight_2 -> pre_weight_1`
- `ai_pre_weight_3 -> pre_weight_2`
- `ai_pre_weight_4 -> pre_weight_3`
- `ai_post_weight_1 -> pre_weight_4`
- `ai_post_weight_2 -> pre_weight_5`
- `ai_post_weight_3 -> pre_weight_6`
- `ai_post_weight_4 -> pre_weight_7`
- `ai_side_wall_1 -> pre_weight_8`
- `ai_side_wall_2 -> side_wall_1`
- `ai_side_wall_3 -> side_wall_2`
- `ai_side_wall_4 -> side_wall_3`
- `ai_centre_wall_1 -> centre_wal`
- `ai_centre_wall_2 -> pre_weight_9`
- `ai_centre_wall_3 -> centre_wal_1`
- `ai_centre_wall_4 -> centre_wal_2`

Status and reason companion writes are supported by the config structure, but they remain disabled until you add explicit destination fields.

## Sample Log Lines

```text
2026-03-20 12:01:11,482 INFO mfcvision startup_runtime_config runtime_mode=web log_level=INFO writeback_mode=async queue_name=mfc_clappia_jobs failed_queue_name=mfc_clappia_failed namespace=mfc_clappia redis_configured=true redis_namespace={"configured":true,"dedupe_prefix":"mfc_clappia:dedupe:","failed_key":"mfc_clappia:failed","failed_queue_name":"mfc_clappia_failed","jobs_key":"mfc_clappia:jobs","namespace":"mfc_clappia","processing_key":"mfc_clappia:processing","queue_name":"mfc_clappia_jobs"}
2026-03-20 12:01:11,483 INFO mfcvision clappia_request_received trace_id=clappia-8c06ec77 submission_id=QEF16211579 payload={"submission_id":"QEF16211579","targets":{"ai_pre_weight_1":"https://drive.google.com/thumbnail","ai_pre_weight_2":"https://drive.google.com/thumbnail"}}
2026-03-20 12:01:11,483 INFO mfcvision clappia_targets_normalized trace_id=clappia-8c06ec77 submission_id=QEF16211579 targets={"ai_pre_weight_1":"https://drive.google.com/thumbnail","ai_pre_weight_2":"https://drive.google.com/thumbnail"}
2026-03-20 12:01:11,484 INFO mfcvision mfc_queue_enqueue_complete trace_id=clappia-8c06ec77 submission_id=QEF16211579 result={"dedupe_hash":"ab12...","duplicate":false,"enqueued":true,"job_id":"mfcjob-fd6f6f0e","jobs_key":"mfc_clappia:jobs","queue_name":"mfc_clappia_jobs"}
2026-03-20 12:01:12,010 INFO mfcvision mfc_worker_job_start trace_id=clappia-8c06ec77 submission_id=QEF16211579 job_id=mfcjob-fd6f6f0e queue_name=mfc_clappia_jobs
2026-03-20 12:01:12,011 INFO mfcvision clappia_target_start trace_id=clappia-8c06ec77 submission_id=QEF16211579 target=ai_pre_weight_1 url=https://drive.google.com/thumbnail
2026-03-20 12:01:20,122 INFO mfcvision clappia_writeback_target trace_id=clappia-8c06ec77 submission_id=QEF16211579 target=ai_pre_weight_1 destination_field=pre_weight final_status=ok numeric_value=18.54 included=True skipped_reason=None
2026-03-20 12:01:20,123 INFO mfcvision clappia_writeback_start trace_id=clappia-8c06ec77 submission_id=QEF16211579 field_map={"ai_pre_weight_1":"pre_weight"} payload_keys=["pre_weight"] payload={"appId":"MFC182090","submissionId":"QEF16211579","workplaceId":"PIP121027","data":{"pre_weight":18.54}}
2026-03-20 12:01:21,010 INFO mfcvision clappia_writeback_success trace_id=clappia-8c06ec77 submission_id=QEF16211579 response_status=200 application_status=uncertain written_fields=["pre_weight"] response_body={}
2026-03-20 12:01:21,011 INFO mfcvision clappia_response_payload trace_id=clappia-8c06ec77 submission_id=QEF16211579 response={"ok":true,"mode":"sync","submission_id":"QEF16211579","writeback_status":"success"}
```

## Result Semantics

- For `POST /api/read-scale`, `final.status` is `ok` or `needs_review`.
- For `POST /api/clappia/analyze`, each `results.<target>.status` is `ok` or `needs_review`.
- `value_text` is expected to be a single numeric token.
- Numeric values are cleared when the result is suspicious.
- `confidence` is a bounded heuristic score, not a calibrated probability.
- `ignored_text_present` comes from the model response schema.

## Known Caveats

- Success still depends heavily on localization quality.
- Full-image fallback can miss very small displays.
- `crop_diagnostics` is currently placeholder metadata; the live read path does not populate it from `vision.py`.
- Preview URLs are process-local and only reflect the latest request handled by the running server.
- The Clappia endpoint fetches remote images over HTTP and applies the same max-size checks as uploads.
- The endpoint returns `400` if it cannot find at least one analyzable image URL in the request body.
- Clappia writeback only sends fields whose final read status is `ok` and whose numeric value is present.

## Troubleshooting

### Missing API Key

- Ensure `GEMINI_API_KEY` is set in `.env`
- Ensure the process is started from the environment where `.env` is available

### Remote URL Errors

- Ensure the URL returns an actual image with an `image/*` content type
- Ensure the URL is reachable from the server running the app
- Ensure the remote image is within `MAX_IMAGE_MB`
- Set `LOG_LEVEL=INFO` or higher and inspect request-stage logs for `remote_fetch_*`, `localizer_*`, `reader_*`, and `pipeline_complete`

### Clappia Writeback

- Set `CLAPPIA_API_KEY` before expecting backend-side Clappia updates.
- Set `CLAPPIA_WORKPLACE_ID`; Clappia public `submissions/edit` requires it.
- Tune `CLAPPIA_ANALYZE_CONCURRENCY` if you change models or hit Gemini or Render limits.
- Ensure the incoming request includes `submissionId`; writeback is skipped without it.
- Inspect `writeback_status`, `writeback_summary`, and `clappia_writeback_payload` in the API response.
- In logs, inspect `clappia_writeback_start`, `clappia_writeback_target`, `clappia_writeback_success`, `clappia_writeback_failed`, and `clappia_writeback_exception`.

### Async Queue

- Set `REDIS_URL` when `MFC_WRITEBACK_MODE=async`.
- Run `python app.py worker` in at least one process.
- Inspect `mfc_queue_enqueue_*`, `mfc_worker_job_*`, and the `mfc_clappia:failed` list for failures.

### Poor Localization

- Keep the display clearly visible in the image.
- Increase `LOCALIZER_MAX_DIMENSION` for small or distant displays.
- Lower `LOCALIZER_MIN_CONFIDENCE` only if you are willing to accept more aggressive crops.

## Development Notes

- If you change request or response behavior in `app.py`, update this README in the same change.
- If you change the runtime container behavior, update `Dockerfile` and this README together.
