"""core.sqlmap_parser

Best-effort parsing for sqlmap stdout/stderr.

Goal:
- Determine whether sqlmap confirmed an injectable parameter
- Extract vulnerable parameter name (if present)
- Extract an exploited payload string (best-effort)
- Detect block/WAF-ish responses and timeouts

Blocked detection is intentionally conservative to avoid false positives.
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


# We want a *reliable* confirmation, not heuristic messages.
# Prefer sqlmap's final summary section ("sqlmap identified the following injection point(s)")
# and/or the detailed block containing Parameter/Type/Payload.
_RE_INJECTABLE_STRONG = re.compile(r"(?is)sqlmap\s+identified\s+the\s+following\s+injection\s+point\(s\)")
# This regex is now designed to find ANY block that has Type, Title, and Payload, which sqlmap prints for each successful finding.
_RE_INJECTION_POINT_BLOCK = re.compile(
    r"(?is)Type:\s*(?P<type>[^\n\r]+)\s*\n\s*Title:\s*(?P<title>[^\n\r]+)\s*\n\s*Payload:\s*(?P<payload>[^\n\r]+)"
)

# Fallback (less strict): lines like "GET parameter 'id' is vulnerable" or "... is ... injectable"
_RE_INJECTABLE_FALLBACK = re.compile(r"(?i)\b(GET|POST)\s+parameter\s+'[^']+'\s+(?:appears\s+to\s+be\s+|is\s+).*?\binjectable\b|\bis\s+vulnerable\b")
_RE_PARAM = re.compile(r"(?i)parameter:\s*([\w\-\[\]\.]+)\s*\((GET|POST)\)")
_RE_TECHNIQUE = re.compile(r"(?i)Type:\s*(boolean-based blind|error-based|time-based blind|UNION query|stacked queries)")
_RE_DBMS = re.compile(r"(?i)back-end DBMS:\s*([^\n\r]+)")
_RE_PAYLOAD_LINE = re.compile(r"(?i)Payload:\s*(.+)")

# Strong WAF/IPS indicators.
# IMPORTANT: Do not treat sqlmap's generic sentence "checking if the target is protected by some kind of WAF/IPS" as blocked.
_RE_WAF_STRONG = re.compile(
    r"(?is)(?:\bWAF\b\s*/\s*IPS\s+protection\s+identified|\bWAF\b\s+identified|"
    r"web\s+application\s+firewall\s+identified|"
    r"identified\s+waf|identified\s+as\s+waf|"
    r"mod_security|modsecurity|cloudflare|akamai|imperva|f5\s+big\s*ip|"
    r"access\s+denied|request\s+denied|captcha)")

_RE_HTTP_BLOCK = re.compile(r"(?i)\b(403|406|429)\b")


def parse_sqlmap_output(stdout: str, stderr: str, *, timed_out: bool) -> SqlmapObservation:
    text = (stdout or "") + "\n" + (stderr or "")

    # Strong confirmation: look for the final injection point summary block.
    # Primary confirmation: if sqlmap printed a concrete Payload, treat it as confirmed injectable.
    # This makes detection robust even if the run is cut short (timeout) before the final summary header.
    m_point = _RE_INJECTION_POINT_BLOCK.search(text)
    if m_point:
        injectable = True
        payload = m_point.group("payload").strip()

        # Best-effort: derive technique from the Type line when available
        technique = m_point.group("type").strip() if m_point.group("type") else None

        # Best-effort vulnerable parameter (might be missing depending on where we matched)
        m_param = _RE_PARAM.search(text)
        vuln_param = m_param.group(1) if m_param else None

        dbms = None
    else:
        injectable = bool(_RE_INJECTABLE_STRONG.search(text) or _RE_INJECTABLE_FALLBACK.search(text))

        m_param = _RE_PARAM.search(text)
        vuln_param = m_param.group(1) if m_param else None

        payload = None
        m_pay = _RE_PAYLOAD_LINE.search(text)
        if m_pay:
            payload = m_pay.group(1).strip()

    # If we didn't already extract dbms/technique from the strong block, try best-effort extraction
    if "dbms" not in locals() or dbms is None:
        m_dbms = _RE_DBMS.search(text)
        dbms = m_dbms.group(1).strip() if m_dbms else None

    if "technique" not in locals() or technique is None:
        m_tech = _RE_TECHNIQUE.search(text)
        technique = m_tech.group(1).strip() if m_tech else None

    blocked = bool(_RE_HTTP_BLOCK.search(text) or _RE_WAF_STRONG.search(text))

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


def _extract_relevant_snippet(text: str, max_len: int = 2000) -> str:
    lines = []
    for line in text.splitlines():
        if re.search(
            r"(?i)parameter:|payload:|back-end dbms:|type:|vulnerable|injectable|identified\s+waf|waf\b\s+identified|modsecurity|cloudflare|akamai|imperva|access denied|captcha|\b(403|406|429)\b",
            line,
        ):
            lines.append(line.strip())
    snippet = "\n".join(lines).strip()
    if not snippet:
        snippet = text.strip()
    if len(snippet) > max_len:
        snippet = snippet[: max_len - 3] + "..."
    return snippet
