"""payload.payload_pool

Payload pool loader/manager.

Loads payloads from a CSV with columns:
- query
- label

Keeps only label == 1 rows, deduplicates payload strings, and exposes simple
sampling and lookup helpers.

Practical notes
- Many public datasets are NOT UTF-8. They may be UTF-16 (BOM 0xff 0xfe),
  Windows-1252, or have a UTF-8 BOM.
- To keep scanning ergonomic, we try a small set of common encodings.

Payloads are treated as atomic strings (no tokenization).
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import random
from typing import List, Optional


@dataclass(frozen=True)
class PayloadItem:
    id: int
    query: str


class PayloadPool:
    def __init__(self, csv_path: str, seed: Optional[int] = None):
        self.csv_path = csv_path
        self._rng = random.Random(seed)
        self._payloads: List[PayloadItem] = []
        self._load()

    def _load(self) -> None:
        seen = set()
        payloads: List[PayloadItem] = []

        # Try common encodings: UTF-8 (with/without BOM), UTF-16, CP1252/Latin-1
        encodings_to_try = ["utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "cp1252", "latin-1"]

        last_err: Optional[Exception] = None
        for enc in encodings_to_try:
            try:
                with open(self.csv_path, "r", encoding=enc, newline="") as f:
                    reader = csv.DictReader(f)
                    # Force header read
                    fieldnames = reader.fieldnames
                    if fieldnames is None:
                        raise ValueError(f"CSV has no header: {self.csv_path}")

                    # Normalize fieldnames
                    fields = {name.strip().lower(): name for name in fieldnames}
                    if "query" not in fields or "label" not in fields:
                        raise ValueError(
                            f"CSV must contain columns 'query' and 'label' (got {fieldnames})"
                        )

                    q_col = fields["query"]
                    l_col = fields["label"]

                    for row in reader:
                        if row is None:
                            continue
                        label_raw = (row.get(l_col) or "").strip()
                        if label_raw != "1":
                            continue

                        query = (row.get(q_col) or "").strip()
                        if not query:
                            continue

                        if query in seen:
                            continue
                        seen.add(query)

                        payloads.append(PayloadItem(id=len(payloads), query=query))

                # Success
                self._payloads = payloads
                return
            except UnicodeDecodeError as e:
                last_err = e
                continue
            except Exception as e:
                # Other errors should not be masked by trying other encodings,
                # but we do allow trying if it's clearly encoding-related.
                last_err = e
                continue

        # If we get here, we failed all attempts
        raise ValueError(
            f"Failed to read CSV '{self.csv_path}'. Tried encodings: {encodings_to_try}. "
            f"Last error: {last_err}"
        )

    def size(self) -> int:
        return len(self._payloads)

    def get_payload_by_id(self, id: int) -> str:
        if id < 0 or id >= len(self._payloads):
            raise IndexError(f"payload id out of range: {id}")
        return self._payloads[id].query

    def sample_payload(self) -> str:
        if not self._payloads:
            raise ValueError("payload pool is empty")
        return self._rng.choice(self._payloads).query

    def all_payloads(self) -> List[str]:
        """Return all payloads as a list of strings."""
        return [p.query for p in self._payloads]
