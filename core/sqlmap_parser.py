"""core.sqlmap_parser

Best-effort parsing for sqlmap stdout/stderr.

Goal:
- Determine whether sqlmap confirmed an injectable parameter
- Extract vulnerable parameter name (if present)
- Extract an exploited payload string (best-effort)
- Detect block/WAF-ish responses and timeouts

NOTE: sqlmap output varies by version/config. This parser intentionally uses
heuristics and regex; it should be improved with real samples from your runs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SqlmapObservation:
    injectable: bool
    vulnerable_param: Optional[str]
    exploited_payload: Optional[str]
    technique: Optional[str]
    dbms: Optional[str]
    blocked: bool
    timeout: bool
    raw_snippet: str


_RE_INJECTABLE = re.compile(r"(?i)parameter: .*? is vulnerable|is injectable")
_RE_PARAM = re.compile(r"(?i)parameter:\s*([\w\-\[\]\.]+)\s*\((GET|POST)\)")
_RE_TECHNIQUE = re.compile(r"(?i)Type:\s*(boolean-based blind|error-based|time-based blind|UNION query|stacked queries)")
_RE_DBMS = re.compile(r"(?i)back-end DBMS:\s*([^\n\r]+)")
_RE_PAYLOAD_LINE = re.compile(r"(?i)Payload:\s*(.+)")

_BLOCK_MARKERS = [
    "WAF",
    "web application firewall",
    "blocked",
    "forbidden",
    "not acceptable",
    "access denied",
    "request blocked",
    "security policy",
    "captcha",
]


def parse_sqlmap_output(stdout: str, stderr: str, *, timed_out: bool) -> SqlmapObservation:
    text = (stdout or "") + "\n" + (stderr or "")

    injectable = bool(_RE_INJECTABLE.search(text))

    m_param = _RE_PARAM.search(text)
    vuln_param = m_param.group(1) if m_param else None

    m_dbms = _RE_DBMS.search(text)
    dbms = m_dbms.group(1).strip() if m_dbms else None

    m_tech = _RE_TECHNIQUE.search(text)
    technique = m_tech.group(1).strip() if m_tech else None

    payload = None
    m_pay = _RE_PAYLOAD_LINE.search(text)
    if m_pay:
        payload = m_pay.group(1).strip()

    blocked = _is_blocked(text)

    timeout = bool(timed_out or re.search(r"(?i)tim(e|ed) out|connection timed out|read timed out", text))

    raw_snippet = _extract_relevant_snippet(text)

    return SqlmapObservation(
        injectable=bool(injectable),
        vulnerable_param=vuln_param,
        exploited_payload=payload,
        technique=technique,
        dbms=dbms,
        blocked=bool(blocked),
        timeout=bool(timeout),
        raw_snippet=raw_snippet,
    )


def _is_blocked(text: str) -> bool:
    t = text.lower()
    for m in _BLOCK_MARKERS:
        if m.lower() in t:
            return True
    if re.search(r"(?i)\b(403|406|429)\b", text):
        return True
    return False


def _extract_relevant_snippet(text: str, max_len: int = 2000) -> str:
    lines = []
    for line in text.splitlines():
        if re.search(r"(?i)parameter:|payload:|back-end dbms:|type:|vulnerable|injectable|waf|forbidden|captcha", line):
            lines.append(line.strip())
    snippet = "\n".join(lines).strip()
    if not snippet:
        snippet = text.strip()
    if len(snippet) > max_len:
        snippet = snippet[: max_len - 3] + "..."
    return snippet
