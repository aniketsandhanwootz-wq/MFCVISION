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

Run the app:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:8000
```

Equivalent manual launch:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

## Docker

Build:

```bash
docker build -t mfcvision .
```

Run:

```bash
docker run --rm -p 8000:10000 --env-file .env mfcvision
```

The container starts Uvicorn on `0.0.0.0` and uses `PORT` if provided, otherwise `10000`.

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

## Repository Layout

- `app.py`: FastAPI app, Gemini calls, Clappia URL analysis endpoint, preview endpoints
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

## API

### `GET /`

Returns the static web UI.

### `GET /health`

Returns the current runtime settings snapshot.

Example:

```json
{
  "ok": true,
  "model": "gemini-2.5-flash",
  "localizer_model": "gemini-2.5-flash",
  "localizer_min_confidence": 0.45,
  "localizer_skip_refine_threshold": 0.9,
  "localizer_max_dimension": 1024,
  "local_decoder_min_confidence": 0.9,
  "expected_decimals": null
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
    "skipped_refine": true,
    "box_cy_frac": 0.54,
    "box_norm_1000": {
      "x1": 210,
      "y1": 480,
      "x2": 760,
      "y2": 610
    },
    "box_pixels": {
      "x1": 128,
      "y1": 302,
      "x2": 451,
      "y2": 387
    }
  },
  "crop_diagnostics": {
    "is_reliable": false,
    "mode": "led",
    "quality_score": 0.0,
    "lit_ratio": 0.0,
    "green_ratio": 0.0,
    "component_count": 0,
    "active_span_ratio": 0.0,
    "active_band_height_ratio": 0.0,
    "leading_blank_ratio": 0.0,
    "reason": "Placeholder metadata; current route does not populate diagnostics."
  },
  "elapsed_seconds": 1.74,
  "preview_urls": {
    "original": "/api/preview/original",
    "enhanced": "/api/preview/enhanced",
    "crop": "/api/preview/crop",
    "debug": "/api/preview/debug"
  }
}
```

### `POST /api/clappia/analyze`

JSON endpoint for analyzing one or more remote image URLs. Each image field becomes a keyed result in the top-level response plus a full diagnostic payload under `details`.

Supported request shapes:

Nested `targets` object:

```json
{
  "submission_id": "sub_123",
  "targets": {
    "gross_weight": "https://example.com/images/gross.jpg",
    "net_weight": "https://example.com/images/net.jpg"
  }
}
```

Flat top-level fields, which is often easier to configure in Clappia Workflows:

```json
{
  "submissionId": "sub_123",
  "gross_weight": "https://example.com/images/gross.jpg",
  "net_weight": "https://example.com/images/net.jpg"
}
```

Response shape:

```json
{
  "ok": true,
  "submission_id": "sub_123",
  "gross_weight": 18.54,
  "gross_weight_text": "18.540",
  "gross_weight_status": "ok",
  "gross_weight_confidence": 0.96,
  "gross_weight_reason": "Read from localized crop.",
  "net_weight": null,
  "net_weight_text": null,
  "net_weight_status": "needs_review",
  "net_weight_confidence": 0.0,
  "net_weight_reason": "Image URL returned HTTP 404.",
  "details": {
    "gross_weight": {
      "final": {},
      "localization": {},
      "crop_diagnostics": {},
      "elapsed_seconds": 1.74,
      "preview_urls": {}
    },
    "net_weight": {
      "final": {
        "status": "needs_review",
        "value_text": null,
        "value_number": null,
        "confidence": 0.0,
        "reason": "Image URL returned HTTP 404.",
        "ignored_text_present": false
      },
      "localization": {
        "source": "clappia_url_error",
        "found": false,
        "confidence": 0.0,
        "display_kind": "unknown",
        "reason": "Image URL returned HTTP 404."
      }
    }
  }
}
```

## Result Semantics

- `final.status` is `ok` or `needs_review`
- `value_text` is expected to be a single numeric token
- `value_number` is cleared when the result is suspicious
- `confidence` is a bounded heuristic score, not a calibrated probability
- `ignored_text_present` comes from the model response schema

## Known Caveats

- Success still depends heavily on localization quality.
- Full-image fallback can miss very small displays.
- `crop_diagnostics` is currently placeholder metadata; the live read path does not populate it from `vision.py`.
- Preview URLs are process-local and only reflect the latest request handled by the running server.
- The Clappia endpoint fetches remote images over HTTP and applies the same max-size checks as uploads.
- The endpoint now returns `400` if it cannot find at least one analyzable image URL in the request body.

## Troubleshooting

### Missing API Key

- Ensure `GEMINI_API_KEY` is set in `.env`
- Ensure the process is started from the environment where `.env` is available

### Remote URL Errors

- Ensure the URL returns an actual image with an `image/*` content type
- Ensure the URL is reachable from the server running the app
- Ensure the remote image is within `MAX_IMAGE_MB`
- Set `LOG_LEVEL=INFO` or higher in Render and inspect request-stage logs for `remote_fetch_*`, `localizer_*`, `reader_*`, and `pipeline_complete`

### Poor Localization

- Keep the display clearly visible in the image
- Increase `LOCALIZER_MAX_DIMENSION` for small or distant displays
- Lower `LOCALIZER_MIN_CONFIDENCE` only if you are willing to accept more aggressive crops

## Development Notes

- If you change request or response behavior in `app.py`, update this README in the same change.
- If you change the runtime container behavior, update `Dockerfile` and this README together.
