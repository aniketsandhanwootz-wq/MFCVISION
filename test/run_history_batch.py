#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from html import escape
from pathlib import Path
from typing import Any
from zipfile import ZipFile
import xml.etree.ElementTree as ET

from openpyxl import Workbook, load_workbook

REPO_ROOT = Path(__file__).resolve().parents[1]
HELPER_SCRIPT = REPO_ROOT / "test" / "run_single_history_case.py"
VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"


NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
CREATED_AT_RE = re.compile(r"^(\d{2}-\d{2}-\d{4})")
HYPERLINK_RE = re.compile(r'^HYPERLINK\("([^"]+)",\s*"([^"]*)"\)$', re.IGNORECASE)
RESULT_HEADERS = [
    "run_timestamp",
    "source_workbook",
    "window_start",
    "window_end",
    "batch_day",
    "case_id",
    "submission_id",
    "owner",
    "created_at",
    "time_slot",
    "section",
    "case_label",
    "expected_raw",
    "expected_value",
    "expected_display",
    "image_url",
    "file_key",
    "report_image_url",
    "system_status",
    "system_value",
    "system_confidence",
    "matched",
    "result_label",
    "error_stage",
    "reason",
    "display_kind",
    "localization_source",
    "support_read_available",
    "elapsed_seconds",
]


@dataclass(frozen=True)
class CaseRecord:
    batch_day: str
    submission_id: str
    owner: str
    created_at: str
    time_slot: str
    section: str
    case_label: str
    expected_raw: str
    expected_value: str
    expected_display: str
    image_url: str

    @property
    def case_id(self) -> str:
        return "|".join(
            [
                self.batch_day,
                self.submission_id,
                _slug(self.section),
                _slug(self.case_label),
            ]
        )


def _slug(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", (text or "").strip().lower())
    return cleaned.strip("_") or "unknown"


def _col_letters(col_idx: int) -> str:
    letters = []
    while col_idx > 0:
        col_idx, remainder = divmod(col_idx - 1, 26)
        letters.append(chr(65 + remainder))
    return "".join(reversed(letters))


def _parse_created_day(value: str) -> str | None:
    match = CREATED_AT_RE.match((value or "").strip())
    if not match:
        return None
    return datetime.strptime(match.group(1), "%d-%m-%Y").date().isoformat()


def _parse_hyperlink_formula(formula_text: str) -> str | None:
    if not formula_text:
        return None
    match = HYPERLINK_RE.match(formula_text.strip())
    if not match:
        return None
    return match.group(1).strip() or None


def _normalize_expected(raw_text: str) -> tuple[str | None, str | None]:
    text = (raw_text or "").strip()
    if not text:
        return None, None

    lowered = text.lower()
    if lowered in {"-infinity", "infinity", "nan"}:
        return None, None

    try:
        if re.fullmatch(r"\d+", text):
            digits = text.lstrip("0") or "0"
            if len(digits) <= 3:
                normalized = Decimal(f"0.{digits.zfill(3)}")
            else:
                normalized = Decimal(f"{digits[:-3]}.{digits[-3:]}")
        else:
            normalized = Decimal(text)
    except InvalidOperation:
        return None, None

    quantized = normalized.quantize(Decimal("0.001"))
    return format(quantized, "f"), f"{quantized:.3f}"


def _read_cell_payload(cell: ET.Element) -> dict[str, str]:
    formula = cell.find("a:f", NS)
    value = cell.find("a:v", NS)
    inline_string = cell.find("a:is", NS)
    text = ""
    if inline_string is not None:
        text = "".join(inline_string.itertext())
    elif value is not None and value.text is not None:
        text = value.text
    return {
        "formula": formula.text if formula is not None and formula.text else "",
        "value": text,
    }


def parse_cases_from_workbook(xlsx_path: Path) -> list[CaseRecord]:
    with ZipFile(xlsx_path) as zf:
        root = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))

    rows = root.findall(".//a:sheetData/a:row", NS)
    if len(rows) < 3:
        return []

    row1 = {}
    row2 = {}
    for cell in rows[0].findall("a:c", NS):
        ref = cell.attrib["r"]
        col = "".join(ch for ch in ref if ch.isalpha())
        row1[col] = _read_cell_payload(cell)["value"]
    for cell in rows[1].findall("a:c", NS):
        ref = cell.attrib["r"]
        col = "".join(ch for ch in ref if ch.isalpha())
        row2[col] = _read_cell_payload(cell)["value"]

    image_cols: list[str] = []
    for idx in range(1, max(len(row2), 123) + 1):
        col = _col_letters(idx)
        header = (row2.get(col) or "").strip()
        if header.startswith("Images of "):
            image_cols.append(col)

    cases: list[CaseRecord] = []
    for row in rows[2:]:
        payload_by_col: dict[str, dict[str, str]] = {}
        for cell in row.findall("a:c", NS):
            ref = cell.attrib["r"]
            col = "".join(ch for ch in ref if ch.isalpha())
            payload_by_col[col] = _read_cell_payload(cell)

        created_at = payload_by_col.get("C", {}).get("value", "")
        batch_day = _parse_created_day(created_at)
        if not batch_day:
            continue

        submission_id = payload_by_col.get("A", {}).get("value", "").strip()
        owner = payload_by_col.get("B", {}).get("value", "").strip()
        time_slot = payload_by_col.get("G", {}).get("value", "").strip()

        for image_col in image_cols:
            value_col = _col_letters(_letters_to_num(image_col) - 1)
            value_header = (row2.get(value_col) or "").strip()
            section = (row1.get(image_col) or row1.get(value_col) or "").strip()
            expected_raw = payload_by_col.get(value_col, {}).get("value", "").strip()
            expected_value, expected_display = _normalize_expected(expected_raw)
            image_formula = payload_by_col.get(image_col, {}).get("formula", "").strip()
            image_url = _parse_hyperlink_formula(image_formula)

            if not value_header or not expected_value or not image_url:
                continue

            cases.append(
                CaseRecord(
                    batch_day=batch_day,
                    submission_id=submission_id,
                    owner=owner,
                    created_at=created_at.strip(),
                    time_slot=time_slot,
                    section=section or "Unknown Section",
                    case_label=value_header,
                    expected_raw=expected_raw,
                    expected_value=expected_value,
                    expected_display=expected_display,
                    image_url=image_url,
                )
            )

    return cases


def _letters_to_num(letters: str) -> int:
    value = 0
    for char in letters:
        value = value * 26 + (ord(char.upper()) - 64)
    return value


def select_last_two_week_days(cases: list[CaseRecord]) -> list[str]:
    all_days = sorted({case.batch_day for case in cases})
    if not all_days:
        return []
    max_day = date.fromisoformat(all_days[-1])
    window_start = max_day - timedelta(days=13)
    return [day for day in all_days if window_start <= date.fromisoformat(day) <= max_day]


def _matched(expected_value: str | None, system_value: str | None, status: str) -> bool:
    if status != "ok" or not expected_value or not system_value:
        return False
    try:
        expected_decimal = Decimal(expected_value).quantize(Decimal("0.001"))
        system_decimal = Decimal(system_value).quantize(Decimal("0.001"))
    except InvalidOperation:
        return False
    return expected_decimal == system_decimal


def _result_label(status: str, matched: bool) -> str:
    if matched:
        return "Correct"
    if status in {"fetch_failed", "execution_failed"}:
        return "Blocked"
    if status == "needs_review":
        return "Needs review"
    if status != "ok":
        return "Failed"
    return "Mismatch"


def _run_single_case(
    case: CaseRecord,
    *,
    run_timestamp: str,
    source_workbook: str,
    window_start: str,
    window_end: str,
) -> dict[str, Any]:
    row = {
        "run_timestamp": run_timestamp,
        "source_workbook": source_workbook,
        "window_start": window_start,
        "window_end": window_end,
        "batch_day": case.batch_day,
        "case_id": case.case_id,
        "submission_id": case.submission_id,
        "owner": case.owner,
        "created_at": case.created_at,
        "time_slot": case.time_slot,
        "section": case.section,
        "case_label": case.case_label,
        "expected_raw": case.expected_raw,
        "expected_value": case.expected_value,
        "expected_display": case.expected_display,
        "image_url": case.image_url,
        "file_key": "",
        "report_image_url": "",
        "system_status": "",
        "system_value": "",
        "system_confidence": "",
        "matched": "no",
        "result_label": "",
        "error_stage": "",
        "reason": "",
        "display_kind": "",
        "localization_source": "",
        "support_read_available": "no",
        "elapsed_seconds": "",
    }
    try:
        analysis = _run_analysis_helper(
            image_url=case.image_url,
            trace_id=f"history-{_slug(case.case_id)}",
            target_key=_slug(case.case_label),
        )
        final = analysis["final"]
        localization = analysis.get("localization") or {}
        support_read = analysis.get("support_read") or {}
        history_source = analysis.get("_history_source") or {}

        system_value = final.get("value_text")
        system_status = final.get("status") or "needs_review"
        is_match = _matched(case.expected_value, system_value, system_status)
        row.update(
            {
                "file_key": history_source.get("file_key"),
                "report_image_url": history_source.get("fetch_image_url") or case.image_url,
                "system_status": system_status,
                "system_value": system_value,
                "system_confidence": final.get("confidence"),
                "matched": "yes" if is_match else "no",
                "result_label": _result_label(system_status, is_match),
                "reason": final.get("reason"),
                "display_kind": localization.get("display_kind"),
                "localization_source": localization.get("source"),
                "support_read_available": "yes" if support_read.get("available") else "no",
                "elapsed_seconds": analysis.get("elapsed_seconds"),
            }
        )
    except Exception as exc:
        message = str(exc)
        status = "fetch_failed" if "remote url is not an image" in message.lower() or "could not fetch image url" in message.lower() else "execution_failed"
        row.update(
            {
                "system_status": status,
                "result_label": _result_label(status, False),
                "error_stage": "fetch" if status == "fetch_failed" else "execution",
                "reason": message,
            }
        )
    return row


def run_batch(
    cases: list[CaseRecord],
    *,
    source_workbook: str,
    window_start: str,
    window_end: str,
    workers: int,
) -> list[dict[str, Any]]:
    run_timestamp = datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")
    max_workers = max(1, min(workers, len(cases)))
    if max_workers == 1:
        return [
            _run_single_case(
                case,
                run_timestamp=run_timestamp,
                source_workbook=source_workbook,
                window_start=window_start,
                window_end=window_end,
            )
            for case in cases
        ]

    results_by_case_id: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(
                _run_single_case,
                case,
                run_timestamp=run_timestamp,
                source_workbook=source_workbook,
                window_start=window_start,
                window_end=window_end,
            ): case.case_id
            for case in cases
        }
        for future in concurrent.futures.as_completed(future_map):
            case_id = future_map[future]
            results_by_case_id[case_id] = future.result()

    return [results_by_case_id[case.case_id] for case in cases]


def _run_analysis_helper(*, image_url: str, trace_id: str, target_key: str) -> dict[str, Any]:
    completed = subprocess.run(
        [
            str(VENV_PYTHON),
            str(HELPER_SCRIPT),
            "--image-url",
            image_url,
            "--trace-id",
            trace_id,
            "--target-key",
            target_key,
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "History case helper failed.\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return json.loads(completed.stdout)


def load_existing_case_rows(workbook_path: Path) -> list[dict[str, Any]]:
    if not workbook_path.exists():
        return []
    wb = load_workbook(workbook_path)
    if "cases" not in wb.sheetnames:
        return []
    ws = wb["cases"]
    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        return []
    headers = [str(cell) if cell is not None else "" for cell in rows[0]]
    existing: list[dict[str, Any]] = []
    for row in rows[1:]:
        if not any(cell not in (None, "") for cell in row):
            continue
        existing.append({header: row[idx] for idx, header in enumerate(headers)})
    return existing


def write_report_workbook(workbook_path: Path, case_rows: list[dict[str, Any]]) -> None:
    wb = Workbook()
    cases_ws = wb.active
    cases_ws.title = "cases"
    cases_ws.append(RESULT_HEADERS)
    for row in case_rows:
        cases_ws.append([row.get(header) for header in RESULT_HEADERS])

    summary_ws = wb.create_sheet("daily_summary")
    summary_headers = [
        "batch_day",
        "total_cases",
        "correct",
        "mismatch",
        "needs_review",
        "blocked",
        "accuracy_percent",
    ]
    summary_ws.append(summary_headers)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in case_rows:
        grouped.setdefault(str(row["batch_day"]), []).append(row)
    for batch_day in sorted(grouped):
        rows = grouped[batch_day]
        total = len(rows)
        correct = sum(1 for row in rows if row["matched"] == "yes")
        needs_review = sum(1 for row in rows if row["system_status"] == "needs_review")
        blocked = sum(1 for row in rows if row["system_status"] in {"fetch_failed", "execution_failed"})
        mismatch = total - correct - needs_review - blocked
        evaluated = total - blocked
        accuracy = round((correct / evaluated) * 100, 2) if evaluated else 0.0
        summary_ws.append([batch_day, total, correct, mismatch, needs_review, blocked, accuracy])

    wb.save(workbook_path)


def build_day_report_html(
    *,
    batch_day: str,
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    total = len(rows)
    correct = sum(1 for row in rows if row["matched"] == "yes")
    needs_review = sum(1 for row in rows if row["system_status"] == "needs_review")
    blocked = sum(1 for row in rows if row["system_status"] in {"fetch_failed", "execution_failed"})
    mismatch = total - correct - needs_review - blocked
    evaluated = total - blocked
    accuracy = round((correct / evaluated) * 100, 2) if evaluated else 0.0

    worked = [row for row in rows if row["matched"] == "yes"]
    failed = [row for row in rows if row["matched"] != "yes"]

    def render_card(row: dict[str, Any]) -> str:
        image_url = escape(str(row.get("report_image_url") or row["image_url"]))
        original_image_url = escape(str(row["image_url"]))
        submission_id = escape(str(row["submission_id"]))
        case_label = escape(str(row["case_label"]))
        section = escape(str(row["section"]))
        expected = escape(str(row["expected_display"] or row["expected_raw"] or ""))
        output = escape(str(row["system_value"] or row["system_status"]))
        reason = escape(str(row["reason"] or ""))
        result_label = escape(str(row["result_label"]))
        created_at = escape(str(row["created_at"]))
        time_slot = escape(str(row["time_slot"] or ""))
        return f"""
        <article class="card">
          <div class="card-head">
            <span class="pill {'ok' if row['matched'] == 'yes' else 'bad'}">{result_label}</span>
            <div class="meta">
              <h3>{case_label}</h3>
              <p>{section}</p>
              <p>Submission: <code>{submission_id}</code></p>
              <p>Created At: {created_at}</p>
              <p>Time Slot: {time_slot}</p>
            </div>
          </div>
          <a class="image-link" href="{image_url}" target="_blank" rel="noreferrer">Open image</a>
          <a class="image-link" href="{original_image_url}" target="_blank" rel="noreferrer">Open original Clappia link</a>
          <img class="photo" src="{image_url}" alt="{case_label}" loading="lazy" referrerpolicy="no-referrer" />
          <dl class="metrics">
            <div><dt>Expected</dt><dd>{expected}</dd></div>
            <div><dt>System Output</dt><dd>{output}</dd></div>
            <div><dt>Status</dt><dd>{escape(str(row['system_status']))}</dd></div>
            <div><dt>Confidence</dt><dd>{escape(str(row['system_confidence']))}</dd></div>
            <div><dt>Reason</dt><dd>{reason}</dd></div>
          </dl>
        </article>
        """

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>History Batch Report {batch_day}</title>
  <style>
    :root {{
      --bg: #f5f1e8;
      --panel: #fffdf8;
      --ink: #1f2a3a;
      --muted: #5d6b7a;
      --line: #d9d1c3;
      --ok: #d8f4df;
      --ok-ink: #1d6a35;
      --bad: #ffe1e1;
      --bad-ink: #8b2020;
      --shadow: 0 14px 40px rgba(32, 30, 24, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: linear-gradient(180deg, #f9f5ec 0%, var(--bg) 100%);
      color: var(--ink);
      font-family: Georgia, "Times New Roman", serif;
    }}
    main {{
      max-width: 1500px;
      margin: 0 auto;
      padding: 40px 32px 80px;
    }}
    h1 {{ font-size: 64px; line-height: 1; margin: 0 0 16px; }}
    p.lead {{ font-size: 18px; color: var(--muted); max-width: 980px; }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 20px;
      margin: 30px 0 50px;
    }}
    .stat {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 26px;
      padding: 28px 30px;
      box-shadow: var(--shadow);
    }}
    .stat h2 {{ margin: 0 0 10px; font-size: 18px; color: var(--muted); letter-spacing: 0.08em; text-transform: uppercase; }}
    .stat strong {{ font-size: 54px; }}
    details {{
      background: transparent;
      margin: 24px 0;
      border-top: 2px solid var(--line);
      padding-top: 18px;
    }}
    summary {{
      cursor: pointer;
      font-size: 42px;
      font-weight: 700;
      list-style: none;
    }}
    summary::-webkit-details-marker {{ display: none; }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 28px;
      margin-top: 24px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 28px;
      padding: 28px;
      box-shadow: var(--shadow);
    }}
    .card-head {{
      display: flex;
      gap: 18px;
      align-items: flex-start;
      margin-bottom: 20px;
    }}
    .meta h3 {{ margin: 0 0 8px; font-size: 30px; }}
    .meta p {{ margin: 6px 0; color: var(--muted); font-size: 17px; }}
    .pill {{
      display: inline-block;
      border-radius: 999px;
      padding: 10px 16px;
      font-size: 14px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      white-space: nowrap;
    }}
    .pill.ok {{ background: var(--ok); color: var(--ok-ink); }}
    .pill.bad {{ background: var(--bad); color: var(--bad-ink); }}
    .image-link {{
      display: inline-block;
      margin-bottom: 14px;
      color: #0b4f8a;
      text-decoration: none;
      font-weight: 700;
    }}
    .photo {{
      width: 100%;
      height: 520px;
      object-fit: contain;
      background: #ece8df;
      border-radius: 18px;
      border: 1px solid var(--line);
      display: block;
      margin-bottom: 20px;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px 28px;
      margin: 0;
    }}
    .metrics div {{
      display: grid;
      grid-template-columns: 160px 1fr;
      gap: 12px;
    }}
    .metrics dt {{ font-weight: 700; }}
    .metrics dd {{ margin: 0; color: var(--muted); word-break: break-word; }}
    @media (max-width: 1100px) {{
      .stats, .cards {{ grid-template-columns: 1fr; }}
      h1 {{ font-size: 44px; }}
      summary {{ font-size: 32px; }}
      .photo {{ height: 380px; }}
      .metrics div {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>Day 1 Batch Report</h1>
    <p class="lead">
      Batch day <code>{escape(batch_day)}</code>. This report uses fresh signed Clappia file URLs for image display and keeps the original Clappia wrapper link on each card for traceability.
    </p>
    <section class="stats">
      <article class="stat"><h2>Total Cases</h2><strong>{total}</strong></article>
      <article class="stat"><h2>Correct</h2><strong>{correct}</strong></article>
      <article class="stat"><h2>Did Not Match</h2><strong>{mismatch}</strong></article>
      <article class="stat"><h2>Accuracy</h2><strong>{accuracy}%</strong></article>
    </section>
    <section class="stats">
      <article class="stat"><h2>Needs Review</h2><strong>{needs_review}</strong></article>
      <article class="stat"><h2>Blocked</h2><strong>{blocked}</strong></article>
      <article class="stat"><h2>Evaluated</h2><strong>{evaluated}</strong></article>
      <article class="stat"><h2>Batch Day</h2><strong style="font-size:34px">{escape(batch_day)}</strong></article>
    </section>
    <details open>
      <summary>Worked ({len(worked)})</summary>
      <div class="cards">
        {''.join(render_card(row) for row in worked)}
      </div>
    </details>
    <details open>
      <summary>Did Not Work / Needs Review ({len(failed)})</summary>
      <div class="cards">
        {''.join(render_card(row) for row in failed)}
      </div>
    </details>
  </main>
</body>
</html>
"""
    output_path.write_text(html)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one day of history validation from the exported Clappia workbook.")
    parser.add_argument("--source-xlsx", required=True, type=Path)
    parser.add_argument("--report-xlsx", required=True, type=Path)
    parser.add_argument("--results-json", required=True, type=Path)
    parser.add_argument("--report-html", required=True, type=Path)
    parser.add_argument("--manifest-json", required=True, type=Path)
    parser.add_argument("--day", default=None, help="ISO day to run, for example 2026-03-10. Defaults to Day 1 of the last 2-week window.")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel worker threads for case execution.")
    args = parser.parse_args()

    source_xlsx = args.source_xlsx.resolve()
    all_cases = parse_cases_from_workbook(source_xlsx)
    if not all_cases:
        raise SystemExit("No testable cases found in workbook.")

    window_days = select_last_two_week_days(all_cases)
    if not window_days:
        raise SystemExit("Could not derive the last 2-week window from Created At.")

    target_day = args.day or window_days[0]
    if target_day not in window_days:
        raise SystemExit(
            f"Requested day {target_day} is not in the last 2-week window: {window_days[0]} to {window_days[-1]}"
        )

    day_cases = [case for case in all_cases if case.batch_day == target_day]
    if not day_cases:
        raise SystemExit(f"No cases found for {target_day}.")

    write_json(
        args.manifest_json,
        {
            "source_workbook": str(source_xlsx),
            "window_start": window_days[0],
            "window_end": window_days[-1],
            "batch_day": target_day,
            "case_count": len(day_cases),
            "cases": [asdict(case) | {"case_id": case.case_id} for case in day_cases],
        },
    )

    new_results = run_batch(
        day_cases,
        source_workbook=source_xlsx.name,
        window_start=window_days[0],
        window_end=window_days[-1],
        workers=args.workers,
    )
    write_json(args.results_json, new_results)

    existing_rows = load_existing_case_rows(args.report_xlsx)
    kept_rows = [row for row in existing_rows if str(row.get("batch_day")) != target_day]
    combined_rows = kept_rows + new_results
    combined_rows.sort(key=lambda row: (str(row["batch_day"]), str(row["created_at"]), str(row["submission_id"]), str(row["case_label"])))
    write_report_workbook(args.report_xlsx, combined_rows)
    build_day_report_html(batch_day=target_day, rows=new_results, output_path=args.report_html)

    correct = sum(1 for row in new_results if row["matched"] == "yes")
    needs_review = sum(1 for row in new_results if row["system_status"] == "needs_review")
    blocked = sum(1 for row in new_results if row["system_status"] in {"fetch_failed", "execution_failed"})
    mismatch = len(new_results) - correct - needs_review - blocked
    evaluated = len(new_results) - blocked
    accuracy = round((correct / evaluated) * 100, 2) if evaluated else 0.0
    print(
        json.dumps(
            {
                "batch_day": target_day,
                "window_start": window_days[0],
                "window_end": window_days[-1],
                "case_count": len(new_results),
                "correct": correct,
                "mismatch": mismatch,
                "needs_review": needs_review,
                "blocked": blocked,
                "evaluated": evaluated,
                "accuracy_percent": accuracy,
                "report_xlsx": str(args.report_xlsx),
                "results_json": str(args.results_json),
                "report_html": str(args.report_html),
                "manifest_json": str(args.manifest_json),
                "workers": args.workers,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
