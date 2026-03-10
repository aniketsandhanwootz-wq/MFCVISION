# Gemini Scale Reader

FastAPI service for reading values from weighing-scale and gauge display images.

Current architecture:

1. Accept one uploaded image
2. Run a low-cost Gemini localizer to find the display window
3. If localization succeeds, crop the original image around that window
4. Read the value with `gemini-2.5-flash`
5. If localization is weak or invalid, skip cropping and read directly from the full image
6. For LED crops only, optionally use a deterministic seven-segment decoder as a structural helper or direct answer when confidence is high

There is no legacy CV localizer in the live request path and no separate `pro` verification pass.

## What This Project Does

This service is designed for images where:

- the measuring device may appear anywhere in the frame
- the display can be LED or LCD
- the image may contain extra text like watermarks, timestamps, labels, or branding
- the display may have dim placeholder digits or faint left-side slots

The backend tries to separate two problems:

1. `Where is the display?`
2. `What number does it show?`

Localization is handled by a cheap Gemini model. Reading is handled by `gemini-2.5-flash`.

## Current Runtime Pipeline

### 1. Upload validation

The backend checks:

- content type must be an image
- file size must be within `MAX_IMAGE_MB`
- image must open cleanly via Pillow

Relevant code:

- [app.py](/Users/aniketsandhan/Desktop/MFCVISION/app.py:72)
- [app.py](/Users/aniketsandhan/Desktop/MFCVISION/app.py:82)
- [app.py](/Users/aniketsandhan/Desktop/MFCVISION/app.py:92)

### 2. Full-image enhancement preview

An enhanced full-frame preview is generated for debugging and UI display. This is not the localization engine.

Relevant code:

- [image_enhance.py](/Users/aniketsandhan/Desktop/MFCVISION/image_enhance.py)
- [app.py](/Users/aniketsandhan/Desktop/MFCVISION/app.py:888)

### 3. Gemini localization

The original image is resized to `LOCALIZER_MAX_DIMENSION`, then sent to the localizer model with a structured-output prompt.

The localizer returns:

- `found`
- normalized box coordinates `x1/y1/x2/y2` in `0..1000`
- `confidence`
- `display_kind` as `led`, `lcd`, or `unknown`
- `reason`

The prompt forces the model to return the display window only, not:

- bowl rim
- steel ring
- machine body
- blue base
- buttons
- branding
- watermark text
- reflections

Relevant code:

- [app.py](/Users/aniketsandhan/Desktop/MFCVISION/app.py:117)
- [app.py](/Users/aniketsandhan/Desktop/MFCVISION/app.py:175)
- [app.py](/Users/aniketsandhan/Desktop/MFCVISION/app.py:191)
- [app.py](/Users/aniketsandhan/Desktop/MFCVISION/app.py:771)

### 4. Box validation and expansion

Returned boxes are not accepted blindly.

The backend checks:

- box size is non-zero
- aspect ratio is reasonable for a display
- area is neither too tiny nor too large

Then it expands the valid box with margins depending on `display_kind`:

- LED gets more left margin to preserve faint leading slots/digits
- LCD gets more balanced padding

Relevant code:

- [app.py](/Users/aniketsandhan/Desktop/MFCVISION/app.py:175)
- [app.py](/Users/aniketsandhan/Desktop/MFCVISION/app.py:206)

### 5. Crop analysis

If a VLM crop is available, it is analyzed locally in `vision.py` to estimate:

- whether the crop looks reliable
- whether it behaves more like LED or LCD
- green ratio
- component count
- active row span
- active band height
- leading blank area

This is used for routing and sanity checks. It is not a full localizer anymore.

Relevant code:

- [vision.py](/Users/aniketsandhan/Desktop/MFCVISION/vision.py:433)
- [app.py](/Users/aniketsandhan/Desktop/MFCVISION/app.py:283)

### 6. Reading

There are only two reading modes now.

#### A. Localized crop read

Used when VLM localization is valid.

Path:

- crop created from original image
- crop diagnostics computed
- `gemini-2.5-flash` reads the crop in a single pass
- if LED crop local decoder is highly confident, it can directly return the reading
- if the single pass is suspicious, result is marked `needs_review`

Relevant code:

- [app.py](/Users/aniketsandhan/Desktop/MFCVISION/app.py:600)

#### B. Full-image fallback read

Used when the VLM localizer fails or returns a weak/invalid box.

Path:

- no crop is used for reading
- full original image becomes the only source
- `gemini-2.5-flash` performs a single-pass read
- if suspicious, result is marked `needs_review`

Relevant code:

- [app.py](/Users/aniketsandhan/Desktop/MFCVISION/app.py:709)

## Project Structure

```text
.
├── app.py
├── image_enhance.py
├── vision.py
├── prompts/
│   └── scale_reader.txt
├── static/
│   └── index.html
├── requirements.txt
├── .env.example
└── README.md
```

### File responsibilities

- [app.py](/Users/aniketsandhan/Desktop/MFCVISION/app.py)
  - FastAPI app
  - Gemini localization call
  - Gemini read call
  - fallback routing
  - preview endpoints

- [vision.py](/Users/aniketsandhan/Desktop/MFCVISION/vision.py)
  - crop diagnostics
  - LED/LCD structural analysis on an already-available crop
  - deterministic seven-segment decoding for LED crops

- [image_enhance.py](/Users/aniketsandhan/Desktop/MFCVISION/image_enhance.py)
  - full-frame enhancement preview generation

- [prompts/scale_reader.txt](/Users/aniketsandhan/Desktop/MFCVISION/prompts/scale_reader.txt)
  - read prompt used for Gemini numeric extraction

- [static/index.html](/Users/aniketsandhan/Desktop/MFCVISION/static/index.html)
  - simple upload UI
  - preview display
  - JSON result panel

## Setup

### 1. Create and activate virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
```

Set at minimum:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 4. Run the server

```bash
python app.py
```

Open:

```text
http://127.0.0.1:8000
```

## Environment Variables

Current supported variables:

```env
GEMINI_API_KEY=your_gemini_api_key_here

MODEL_NAME=gemini-2.5-flash
LOCALIZER_MODEL_NAME=gemini-2.5-flash-lite

HOST=0.0.0.0
PORT=8000

MAX_IMAGE_MB=10
MAX_DIMENSION=1200
LOCALIZER_MAX_DIMENSION=1024

LOCALIZER_MIN_CONFIDENCE=0.45
LOCAL_DECODER_MIN_CONFIDENCE=0.90

# Optional: only set for known fixed-precision machines.
# EXPECTED_DECIMALS=3
```

### Variable notes

- `MODEL_NAME`
  - Gemini model used for numeric reading

- `LOCALIZER_MODEL_NAME`
  - Gemini model used only for display-window localization
  - default is a cheaper model than the reader

- `MAX_IMAGE_MB`
  - upload size limit

- `MAX_DIMENSION`
  - full-image enhancement preview dimension

- `LOCALIZER_MAX_DIMENSION`
  - resize cap before localization request
  - lower values reduce latency/cost
  - overly low values can hurt small-display localization

- `LOCALIZER_MIN_CONFIDENCE`
  - minimum localization confidence required before using the returned ROI

- `LOCAL_DECODER_MIN_CONFIDENCE`
  - minimum confidence for the deterministic LED decoder to override the model read directly

- `EXPECTED_DECIMALS`
  - optional
  - only use this for known machines with fixed decimal precision
  - do not set this globally for mixed-device deployments

## API

### `GET /`

Returns the web UI.

### `GET /health`

Returns current runtime settings.

Example response:

```json
{
  "ok": true,
  "model": "gemini-2.5-flash",
  "localizer_model": "gemini-2.5-flash-lite",
  "localizer_min_confidence": 0.45,
  "localizer_max_dimension": 1024,
  "local_decoder_min_confidence": 0.9,
  "expected_decimals": null
}
```

Relevant code:

- [app.py](/Users/aniketsandhan/Desktop/MFCVISION/app.py:748)

### `POST /api/read-scale`

Reads one uploaded image.

Request:

- multipart form-data
- field name: `file`

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
    "source": "vlm",
    "model": "gemini-2.5-flash-lite",
    "found": true,
    "confidence": 0.88,
    "display_kind": "led",
    "reason": "Located the full display window.",
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
    "is_reliable": true,
    "mode": "led",
    "quality_score": 0.81,
    "lit_ratio": 0.09,
    "green_ratio": 0.07,
    "component_count": 6,
    "active_span_ratio": 0.72,
    "active_band_height_ratio": 0.24,
    "leading_blank_ratio": 0.11,
    "reason": "localized crop contains a coherent active display row"
  },
  "preview_urls": {
    "original": "/api/preview/original",
    "enhanced": "/api/preview/enhanced",
    "crop": "/api/preview/crop",
    "debug": "/api/preview/debug"
  }
}
```

Relevant code:

- [app.py](/Users/aniketsandhan/Desktop/MFCVISION/app.py:771)

### `GET /api/preview/{kind}`

Returns the latest preview image for one of:

- `original`
- `enhanced`
- `crop`
- `debug`

Relevant code:

- [app.py](/Users/aniketsandhan/Desktop/MFCVISION/app.py:761)

## Preview Images

The UI shows four images after each request.

### Original

The uploaded image as stored for the latest request.

### Enhanced Full

A full-frame enhanced image for debugging. This is useful for visually inspecting contrast/glare issues.

### Localized Crop

If VLM localization succeeds:

- shows the actual cropped display ROI

If VLM localization fails:

- shows a placeholder image saying no localized ROI was used

### Detection Debug

If VLM localization succeeds:

- shows the original image with the chosen ROI box drawn on top

If VLM localization fails:

- shows the original image with a `full-image fallback` marker

## Output Semantics

### `final.status`

- `ok`
  - backend believes the read is acceptable

- `needs_review`
  - backend believes the result is suspicious or structurally invalid

### `ignored_text_present`

This indicates that the system detected non-measurement text or expected to ignore unrelated text in the image context.

### `confidence`

This is a bounded score in `0..1`.

It is useful for sorting or triage, but should not be treated as a formal probability calibration.

## Suspicious Result Handling

The current backend marks a result as suspicious if, for example:

- output is not a single numeric token
- multiple decimal points appear
- output shape is invalid
- digit count disagrees with strong crop structure hints
- a leading digit is likely missing in LED cases
- `EXPECTED_DECIMALS` is configured and the output violates it

Suspicious outputs are converted to:

- `status = "needs_review"`

Relevant code:

- [app.py](/Users/aniketsandhan/Desktop/MFCVISION/app.py:284)
- [app.py](/Users/aniketsandhan/Desktop/MFCVISION/app.py:419)
- [app.py](/Users/aniketsandhan/Desktop/MFCVISION/app.py:610)

## LED Local Decoder

For LED crops only, `vision.py` includes a deterministic seven-segment decoder.

This is not the primary localization method and not a separate request-stage localizer.

It is used for:

- slot-count hints
- decimal-dot hints
- direct return when decoder confidence is high
- sanity support when the Gemini crop read looks suspicious

Relevant code:

- [vision.py](/Users/aniketsandhan/Desktop/MFCVISION/vision.py:678)
- [app.py](/Users/aniketsandhan/Desktop/MFCVISION/app.py:600)

## Frontend

The UI is intentionally simple:

- upload one image
- run one read request
- show previews
- show final status/value
- show raw JSON for debugging

Relevant code:

- [static/index.html](/Users/aniketsandhan/Desktop/MFCVISION/static/index.html)

## Dependencies

Current Python dependencies:

```text
fastapi==0.116.1
uvicorn[standard]==0.35.0
python-multipart==0.0.22
pydantic==2.11.7
pillow==12.1.1
opencv-python==4.12.0.88
numpy==2.2.6
google-genai==1.33.0
python-dotenv==1.1.1
```

## Operational Notes

### When to lower cost or latency

- reduce `LOCALIZER_MAX_DIMENSION`
- keep `MODEL_NAME` on `gemini-2.5-flash`
- avoid setting overly low `LOCALIZER_MIN_CONFIDENCE`, which can create bad crops and waste time downstream

### When to improve localization reliability

- improve the localizer prompt in [app.py](/Users/aniketsandhan/Desktop/MFCVISION/app.py:117)
- increase `LOCALIZER_MAX_DIMENSION` for small/far displays
- tighten `LOCALIZER_MIN_CONFIDENCE` if too many weak crops are being accepted

### When to improve read quality

- adjust [prompts/scale_reader.txt](/Users/aniketsandhan/Desktop/MFCVISION/prompts/scale_reader.txt)
- tune LED crop diagnostics in [vision.py](/Users/aniketsandhan/Desktop/MFCVISION/vision.py:433)
- tune deterministic LED decoding in [vision.py](/Users/aniketsandhan/Desktop/MFCVISION/vision.py:678)

## Limitations

Known limitations of the current design:

- localization depends on VLM box quality
- no second-pass verify model exists now
- full-image fallback can still fail on very small displays
- LED deterministic decoding is crop-dependent and not suitable for LCD
- confidence scores are heuristic, not calibrated

## Development Notes

If you change runtime behavior, update this README together with:

- [app.py](/Users/aniketsandhan/Desktop/MFCVISION/app.py)
- [.env.example](/Users/aniketsandhan/Desktop/MFCVISION/.env.example)
- [static/index.html](/Users/aniketsandhan/Desktop/MFCVISION/static/index.html)

The current README is intended to describe the actual live path, not historical experiments.
