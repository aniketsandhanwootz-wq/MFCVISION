#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.parse import quote, unquote, urlparse
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app import fetch_image_bytes_from_url, run_scale_reader_pipeline

CLAPPIA_FILE_DOWNLOAD_URL = (
    os.getenv("CLAPPIA_FILE_DOWNLOAD_URL") or "https://apiv2.clappia.com/file/generateFileDownloadUrl"
).strip()


def _extract_file_key_from_clappia_wrapper(url: str) -> str | None:
    parsed = urlparse((url or "").strip())
    if parsed.netloc != "pipl.clappia.com":
        return None

    parts = [unquote(part) for part in parsed.path.split("/") if part]
    try:
        submission_idx = parts.index("submission")
    except ValueError:
        return None

    # Expected wrapper format:
    # /app/<appId>/submission/<submissionId>/<file-storage-prefix>/<file-name>
    remainder = parts[submission_idx + 2 :]
    if len(remainder) < 2:
        return None
    return "/".join(remainder)


def _resolve_history_image_url(url: str) -> tuple[str, str | None, str]:
    file_key = _extract_file_key_from_clappia_wrapper(url)
    if not file_key:
        return url, None, "direct_image_url"

    request = Request(
        f"{CLAPPIA_FILE_DOWNLOAD_URL}?fileName={quote(file_key, safe='/')}",
        headers={"Accept": "text/plain"},
    )
    with urlopen(request, timeout=60) as response:
        resolved_url = response.read().decode().strip()

    if not resolved_url.startswith("http"):
        raise RuntimeError("Clappia file resolver did not return a valid signed URL.")

    return resolved_url, file_key, "clappia_file_service"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one history validation case through the current pipeline.")
    parser.add_argument("--image-url", required=True)
    parser.add_argument("--trace-id", required=True)
    parser.add_argument("--target-key", required=True)
    args = parser.parse_args()

    fetch_url, file_key, resolver_used = _resolve_history_image_url(args.image_url)
    data, content_type = fetch_image_bytes_from_url(
        fetch_url,
        trace_id=args.trace_id,
        target_key=args.target_key,
    )
    analysis = run_scale_reader_pipeline(
        data,
        content_type=content_type,
        trace_id=args.trace_id,
        source="history_batch",
        target_key=args.target_key,
    )
    analysis["_history_source"] = {
        "original_image_url": args.image_url,
        "fetch_image_url": fetch_url,
        "file_key": file_key,
        "resolver_used": resolver_used,
    }
    sys.stdout.write(json.dumps(analysis))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
