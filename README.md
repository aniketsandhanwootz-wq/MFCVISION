# Adaptive Scale Reader

FastAPI service that reads weighing-scale values from images using:

1. Full-image preprocessing
2. Adaptive display localization (no fixed crop)
3. Deterministic 7-segment decoding
4. Optional Gemini fallback for uncertain cases

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

- Local adaptive pipeline from `vision.py` runs first on every request.
- If Gemini fallback is enabled/configured, Gemini reads the adaptively localized ROI (not full image) and result reconciliation is applied.

## Environment variables

```env
HOST=0.0.0.0
PORT=8000
MAX_IMAGE_MB=10

# Gemini fallback (optional)
ENABLE_GEMINI_FALLBACK=1
GEMINI_API_KEY=your_gemini_api_key_here
MODEL_NAME=gemini-2.5-flash
VERIFY_MODEL_NAME=gemini-2.5-pro
LOCAL_DECODER_MIN_CONFIDENCE=0.90

# Optional: only configure this for a known machine with fixed precision.
# EXPECTED_DECIMALS=3
```

## Health endpoint

`GET /health` returns whether adaptive localization is active and Gemini fallback state.
