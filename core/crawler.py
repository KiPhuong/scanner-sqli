"""core.crawler

A small HTML crawler focused on collecting injection targets.

Current behavior:
- Fetches a target URL (single page; no link-following by default).
- Extracts GET parameters from the URL query string.
- Extracts HTML forms (method, action, input names + default values).

Important for sqlmap integration:
- We need the *full default parameter set* for each form/URL, so that we can
  test one parameter at a time while keeping others fixed.

Therefore we expose a higher-level "RequestTemplate" structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

import requests
from bs4 import BeautifulSoup


@dataclass(frozen=True)
class RequestTemplate:
    """A concrete request template extracted from a URL or HTML form."""

    url: str
    method: str  # GET/POST
    params: Dict[str, str]  # default values (query params for GET, body for POST)


class Crawler:
    """Simple crawler/parser that turns a page into request templates."""

    def __init__(
        self,
        timeout: int = 15,
        user_agent: str = "scanner-sqli-crawler/1.0",
        verify_tls: bool = True,
        allow_redirects: bool = True,
        max_response_size: int = 2_000_000,
        session: Optional[requests.Session] = None,
    ):
        self.timeout = timeout
        self.verify_tls = verify_tls
        self.allow_redirects = allow_redirects
        self.max_response_size = max_response_size

        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", user_agent)
        self.session.headers.setdefault(
            "Accept",
            "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        )

    def crawl(self, target_url: str) -> List[RequestTemplate]:
        """Fetch target_url and return discovered request templates."""

        resp = self._fetch(target_url)
        content_type = (resp.headers.get("Content-Type") or "").lower()

        templates: List[RequestTemplate] = []

        # 1) GET template from the URL itself (if it has query params)
        get_tpl = self._extract_get_template(str(resp.url))
        if get_tpl is not None:
            templates.append(get_tpl)

        # 2) HTML forms
        if "html" in content_type or resp.text.lstrip().startswith("<"):
            templates.extend(self._extract_forms(str(resp.url), resp.text))

        return self._dedupe(templates)

    # -------------------------
    # Networking
    # -------------------------

    def _fetch(self, url: str) -> requests.Response:
        r = self.session.get(
            url,
            timeout=self.timeout,
            verify=self.verify_tls,
            allow_redirects=self.allow_redirects,
            stream=True,
        )
        r.raise_for_status()

        content = b""
        for chunk in r.iter_content(chunk_size=64 * 1024):
            if not chunk:
                continue
            content += chunk
            if len(content) > self.max_response_size:
                break

        r._content = content
        r.encoding = r.encoding or "utf-8"
        return r

    # -------------------------
    # Extraction
    # -------------------------

    def _extract_get_template(self, url: str) -> Optional[RequestTemplate]:
        parts = urlsplit(url)
        params = {k: v for k, v in parse_qsl(parts.query, keep_blank_values=True)}
        if not params:
            return None

        base_url = urlunsplit((parts.scheme, parts.netloc, parts.path, "", parts.fragment))
        return RequestTemplate(url=base_url, method="GET", params=params)

    def _extract_forms(self, base_url: str, html: str) -> List[RequestTemplate]:
        soup = BeautifulSoup(html, "html.parser")
        templates: List[RequestTemplate] = []

        for form in soup.find_all("form"):
            method = (form.get("method") or "GET").strip().upper()
            action = (form.get("action") or "").strip()
            action_url = urljoin(base_url, action) if action else base_url

            params: Dict[str, str] = {}
            for name, value in self._iter_form_fields(form):
                # keep the last value if duplicates (common in forms)
                params[name] = value

            # Even if params is empty, a form can still be submit-only; skip empty
            if not params:
                continue

            templates.append(RequestTemplate(url=action_url, method=method, params=params))

        return templates

    def _iter_form_fields(self, form_tag) -> Iterable[tuple[str, str]]:
        """Yield (name, value) for fuzzable fields in a form."""

        for inp in form_tag.find_all("input"):
            name = (inp.get("name") or "").strip()
            if not name:
                continue

            itype = (inp.get("type") or "text").strip().lower()
            if itype in {"submit", "button", "image", "reset", "file"}:
                continue

            if itype in {"checkbox", "radio"}:
                if inp.has_attr("checked"):
                    yield name, str(inp.get("value") or "on")
                continue

            yield name, str(inp.get("value") or "")

        for ta in form_tag.find_all("textarea"):
            name = (ta.get("name") or "").strip()
            if not name:
                continue
            yield name, (ta.text or "")

        for sel in form_tag.find_all("select"):
            name = (sel.get("name") or "").strip()
            if not name:
                continue

            options = sel.find_all("option")
            selected = None
            for opt in options:
                if opt.has_attr("selected"):
                    selected = opt
                    break
            chosen = selected or (options[0] if options else None)
            if chosen is None:
                yield name, ""
            else:
                yield name, str(chosen.get("value") or (chosen.text or ""))

    # -------------------------
    # Utilities
    # -------------------------

    def _dedupe(self, templates: List[RequestTemplate]) -> List[RequestTemplate]:
        seen = set()
        out: List[RequestTemplate] = []
        for t in templates:
            # Normalize params ordering for dedupe
            key = (t.url, t.method.upper(), tuple(sorted(t.params.items())))
            if key in seen:
                continue
            seen.add(key)
            out.append(t)
        return out


def encode_get_url(base_url: str, params: Dict[str, str]) -> str:
    """Build a GET URL with query parameters."""
    return base_url + ("?" + urlencode(params, doseq=True) if params else "")
