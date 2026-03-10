# Adaptive Scale Reader

FastAPI service that reads weighing-scale values from images using:

1. Full-image preprocessing
2. Gemini VLM-first display localization
3. CV fallback localization when VLM localization is weak
4. Gemini `2.5-flash` reading with verification for uncertain cases
5. Deterministic 7-segment hints for LED crops

## Project structure

```text
.
├── app.py
├── vision.py
├── prompts/
│   └── scale_reader.txt
├── static/
│   └── index.html
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python app.py
```

Open `http://127.0.0.1:8000`.

## Runtime behavior

- A low-cost Gemini localizer (`LOCALIZER_MODEL_NAME`) first finds the display window.
- If that localization is weak or invalid, the existing CV localizer from `vision.py` is used as fallback.
- Gemini reads the resulting ROI and can verify against the original photo when the crop-first answer is suspicious.

## Environment variables

```env
HOST=0.0.0.0
PORT=8000
MAX_IMAGE_MB=10

# Gemini localization + reading
GEMINI_API_KEY=your_gemini_api_key_here
LOCALIZER_MODEL_NAME=gemini-2.5-flash-lite
MODEL_NAME=gemini-2.5-flash
VERIFY_MODEL_NAME=gemini-2.5-pro
LOCALIZER_MAX_DIMENSION=1024
LOCALIZER_MIN_CONFIDENCE=0.45
LOCAL_DECODER_MIN_CONFIDENCE=0.90

# Optional: only configure this for a known machine with fixed precision.
# EXPECTED_DECIMALS=3
```

## Health endpoint

`GET /health` returns the read model, verify model, localizer model, and relevant thresholds.
