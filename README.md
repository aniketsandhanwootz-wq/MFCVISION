# 📊 MFCVISION

> A **FastAPI service** for reading numeric values from photos of weighing scales and similar LED/LCD instrument displays using Google's Gemini API.

### How It Works

The backend leverages Gemini for two critical tasks:

1. **Display Localization** — Find the display window in the full image
2. **Value Extraction** — Read numeric values from the localized crop or full image fallback

## Quick Start

```bash
# Clone and set up the environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure API key
cp .env.example .env
# Edit .env and set your GEMINI_API_KEY

# Run the service
python app.py
```

Visit `http://127.0.0.1:8000` in your browser to access the web UI.

## Current Request Flow

1. Validate the upload and open it with Pillow.
2. Resize the image for localization and build an enhanced helper image.
3. Ask the Gemini localizer for a display bounding box.
4. If the first box is usable but not strong enough, run a second region-refinement pass.
5. If a valid box exists, send the crop as the primary image and the full photo as context to the reader model.
6. If localization fails, read directly from the full image.
7. Post-process the result, enforce numeric-shape rules, and downgrade suspicious reads to `needs_review`.
8. Store the latest original, enhanced, crop, and debug previews in memory for the UI.

## Repo Layout

- `app.py`: FastAPI app, Gemini calls, routing, response shaping, preview endpoints
- `image_enhance.py`: conservative enhancement used for preview and localization context
- `vision.py`: image utilities plus crop-analysis and seven-segment helpers not currently wired into the live `/api/read-scale` path
- `prompts/scale_reader.txt`: structured prompt used for numeric extraction
- `static/index.html`: simple upload and debug UI
- `requirements.txt`: Python dependencies

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### Environment Configuration

Edit `.env` and set at least:

```env
GEMINI_API_KEY=your_api_key_here
```

For reference, `.env.example` contains all available configuration options with defaults. Keys not set in `.env` will use hardcoded defaults from `app.py`.

## Run

```bash
python app.py
```

The app starts Uvicorn with reload enabled. Default address:

```text
http://127.0.0.1:8000
```

Equivalent manual launch:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

## Environment Variables

The runtime defaults below come from `app.py` and are the source of truth.

- `GEMINI_API_KEY`: required
- `MODEL_NAME`: reader model, default `gemini-2.5-flash`
- `LOCALIZER_MODEL_NAME`: localizer model, default `gemini-2.5-flash`
- `READ_TEMPERATURE`: default `0.0`
- `LOCALIZER_TEMPERATURE`: default `0.0`
- `HOST`: default `0.0.0.0`
- `PORT`: default `8000`
- `MAX_IMAGE_MB`: upload size limit, default `10`
- `MAX_DIMENSION`: max size for the full-image enhanced preview, default `1200`
- `LOCALIZER_MAX_DIMENSION`: max size sent to the localizer path, default `1024`
- `LOCALIZER_MIN_CONFIDENCE`: minimum accepted localization confidence, default `0.45`
- `LOCALIZER_SKIP_REFINE_THRESHOLD`: skip the second localization pass above this confidence, default `0.90`
- `LOCAL_DECODER_MIN_CONFIDENCE`: retained setting for local decoder helpers, default `0.90`
- `EXPECTED_DECIMALS`: optional fixed decimal count for known devices
- `FIXED_DECIMALS`: alias for `EXPECTED_DECIMALS`

Important note:

- If `LOCALIZER_MODEL_NAME` contains `lite`, the app intentionally replaces it with `MODEL_NAME`. The current code does not allow a lite localizer in production flow.

## API

### `GET /`

Returns the static web UI.

### `GET /health`

Returns the current runtime configuration snapshot.

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
    "source": "vlm_full",
    "model": "gemini-2.5-flash",
    "found": true,
    "confidence": 0.88,
    "display_kind": "led",
    "reason": "Located the display window.",
    "skipped_refine": true,
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

### `GET /api/preview/{kind}`

Returns the latest preview image for:

- `original`
- `enhanced`
- `crop`
- `debug`

These previews are stored in process memory and only represent the most recent request handled by the running server.

## Result Semantics

- `final.status` is `ok` or `needs_review`
- `value_text` is expected to be a single numeric token
- `value_number` is cleared when the read is marked suspicious
- `confidence` is a bounded heuristic score, not a calibrated probability
- `ignored_text_present` comes from the model response schema and indicates unrelated visible text was present or intentionally ignored

## Known Caveats

- Success still depends heavily on localization quality.
- Full-image fallback is slower to reason about and can miss very small displays.
- `crop_diagnostics` is currently returned as placeholder metadata; the live request path does not populate it from `vision.py`.
- The seven-segment decoder and crop-analysis helpers exist in `vision.py`, but they are not active in the current `/api/read-scale` route.
- Preview endpoints are not persistent storage and are shared across requests handled by the same process.

## Troubleshooting

### Module Not Found Errors
Ensure you're running from the activated virtual environment:
```bash
source .venv/bin/activate
```

### API Key Issues
- Verify `GEMINI_API_KEY` is set in `.env`
- Check that your API key is valid and has Gemini access enabled
- The key should be from `https://aistudio.google.com/`

### Poor Localization Results
- Ensure the scale/display is clearly visible in the photo
- Try adjusting `LOCALIZER_MIN_CONFIDENCE` in `.env` (lower = more aggressive)
- Check that `LOCALIZER_MODEL_NAME` is not a lite model; the app will auto-correct it

### Performance Tuning
- Increase `LOCALIZER_MAX_DIMENSION` if localization is missing small displays
- Decrease it if latency is an issue
- Raise `LOCALIZER_SKIP_REFINE_THRESHOLD` to skip refinement more often (saves ~1s per request)

## Development Notes

- If you change request/response behavior in `app.py`, update this README in the same change.
- If your local `.env.example` differs from the defaults documented here, treat `app.py` as authoritative.
