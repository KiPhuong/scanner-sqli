"""extract_payloads.py

Utility script to build a clean SQLi payload CSV from multiple datasets.

What it does
- Scans a directory (default: ./data) for *.csv
- Reads each CSV with common encodings (utf-8/utf-16/cp1252/...)
- Auto-detects common column names for payload text and label
- Keeps only label == 1
- Deduplicates payload strings
- Writes output CSV in the project-expected format: query,label

Example
  python extract_payloads.py --input-dir data --output data/payloads.csv

Notes
- Payloads are treated as atomic strings (no tokenization).
- This script is intentionally dependency-free (no pandas).
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


ENCODINGS_TO_TRY = [
    "utf-8",
    "utf-8-sig",
    "utf-16",
    "utf-16-le",
    "utf-16-be",
    "cp1252",
    "latin-1",
]

# Common header names seen in public SQLi datasets
PAYLOAD_COL_CANDIDATES = [
    "query",
    "payload",
    "sentence",
    "text",
    "input",
    "data",
    "sqli",
]

LABEL_COL_CANDIDATES = [
    "label",
    "class",
    "target",
    "y",
    "is_sqli",
    "issqli",
]


def _norm(s: str) -> str:
    return s.strip().lower()


def _iter_csv_files(input_dir: str) -> List[Path]:
    p = Path(input_dir)
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")
    return sorted([x for x in p.glob("*.csv") if x.is_file()])


def _open_csv_with_encodings(path: Path):
    last_err: Optional[Exception] = None
    for enc in ENCODINGS_TO_TRY:
        try:
            f = open(path, "r", encoding=enc, newline="")
            # Force a read of header via DictReader
            reader = csv.DictReader(f)
            _ = reader.fieldnames
            if reader.fieldnames is None:
                raise ValueError("CSV has no header")
            return f, reader
        except UnicodeDecodeError as e:
            last_err = e
            continue
        except Exception as e:
            last_err = e
            continue
    raise ValueError(
        f"Failed to read '{path}'. Tried encodings {ENCODINGS_TO_TRY}. Last error: {last_err}"
    )


def _detect_columns(fieldnames: Sequence[str]) -> Tuple[str, str]:
    mapping = {_norm(n): n for n in fieldnames}

    q_col = None
    for c in PAYLOAD_COL_CANDIDATES:
        if c in mapping:
            q_col = mapping[c]
            break

    l_col = None
    for c in LABEL_COL_CANDIDATES:
        if c in mapping:
            l_col = mapping[c]
            break

    if q_col is None or l_col is None:
        raise ValueError(f"Cannot detect payload/label columns from header: {list(fieldnames)}")

    return q_col, l_col


def _label_is_positive(v: str) -> bool:
    s = (v or "").strip().lower()
    return s in {"1", "true", "yes", "y", "pos", "positive"}


def extract_payloads(
    input_dir: str,
    output_csv: str,
    keep_duplicates: bool = False,
) -> Dict[str, object]:
    files = _iter_csv_files(input_dir)
    if not files:
        raise FileNotFoundError(f"No .csv files found in: {input_dir}")

    seen = set()
    out_rows: List[Tuple[str, str]] = []

    processed_files = 0
    total_rows = 0
    kept_rows = 0
    skipped_no_label = 0
    skipped_empty = 0
    skipped_dupe = 0

    for path in files:
        processed_files += 1
        f, reader = _open_csv_with_encodings(path)
        try:
            q_col, l_col = _detect_columns(reader.fieldnames or [])

            for row in reader:
                total_rows += 1
                if row is None:
                    continue

                if not _label_is_positive(row.get(l_col, "")):
                    skipped_no_label += 1
                    continue

                q = (row.get(q_col) or "").strip()
                if not q:
                    skipped_empty += 1
                    continue

                if (not keep_duplicates) and (q in seen):
                    skipped_dupe += 1
                    continue

                seen.add(q)
                out_rows.append((q, "1"))
                kept_rows += 1
        finally:
            f.close()

    # Write output
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query", "label"])
        w.writerows(out_rows)

    return {
        "input_dir": input_dir,
        "output_csv": str(out_path),
        "files": [str(p) for p in files],
        "processed_files": processed_files,
        "total_rows": total_rows,
        "kept_rows": kept_rows,
        "unique_payloads": len(seen),
        "skipped_no_label": skipped_no_label,
        "skipped_empty": skipped_empty,
        "skipped_dupe": skipped_dupe,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract/dedupe SQLi payloads from CSV datasets")
    ap.add_argument("--input-dir", default="data", help="Directory containing CSV files")
    ap.add_argument("--output", default="data/payloads.csv", help="Output CSV path")
    ap.add_argument("--keep-duplicates", action="store_true", help="Do not deduplicate payload strings")

    args = ap.parse_args()

    summary = extract_payloads(
        input_dir=args.input_dir,
        output_csv=args.output,
        keep_duplicates=bool(args.keep_duplicates),
    )

    print("Extraction complete")
    print(f"  Input dir     : {summary['input_dir']}")
    print(f"  Output        : {summary['output_csv']}")
    print(f"  Files         : {summary['processed_files']}")
    print(f"  Total rows    : {summary['total_rows']}")
    print(f"  Kept (label=1): {summary['kept_rows']}")
    print(f"  Unique payload: {summary['unique_payloads']}")
    print(f"  Skipped label : {summary['skipped_no_label']}")
    print(f"  Skipped empty : {summary['skipped_empty']}")
    print(f"  Skipped dupes : {summary['skipped_dupe']}")


if __name__ == "__main__":
    main()

