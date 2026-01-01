"""core.crawler

A small, reliable HTML crawler focused on collecting injection points for
pentesting.

Features
- Fetches a target URL (single page; no link-following by default).
- Extracts:
  - GET parameters from the URL query string
  - HTML forms (method, action, input names + default values)
- Produces a list of injection points, each containing:
  - url
  - method
  - param
  - value

Non-goals
- No JavaScript/AJAX execution.
- No complex crawling logic (depth/queue) unless you add it.

This module is intentionally dependency-light and prioritizes robustness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional
from urllib.parse import parse_qsl, urljoin, urlsplit, urlunsplit

import requests
from bs4 import BeautifulSoup


@dataclass(frozen=True)
class InjectionPoint:
    """A single parameter location that can be fuzzed/injected."""

    url: str
    method: str
    param: str
    value: str


class Crawler:
    """Simple crawler/parser that turns a page into injection points."""

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
        self.session.headers.setdefault("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")

    def crawl(self, target_url: str) -> List[InjectionPoint]:
        """Fetch target_url and return discovered injection points."""

        resp = self._fetch(target_url)
        content_type = (resp.headers.get("Content-Type") or "").lower()

        points: List[InjectionPoint] = []

        # 1) GET params from the URL itself
        points.extend(self._extract_get_params(str(resp.url)))

        # 2) HTML forms, only if the response looks like HTML
        if "html" in content_type or resp.text.lstrip().startswith("<"):
            points.extend(self._extract_forms(str(resp.url), resp.text))

        return self._dedupe(points)

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

        # Protect from huge bodies; read up to max_response_size
        content = b""
        for chunk in r.iter_content(chunk_size=64 * 1024):
            if not chunk:
                continue
            content += chunk
            if len(content) > self.max_response_size:
                break

        # Rebuild a normal Response-like object with limited body
        # requests.Response keeps the content in a private attribute
        r._content = content
        r.encoding = r.encoding or "utf-8"
        return r

    # -------------------------
    # Extraction
    # -------------------------

    def _extract_get_params(self, url: str) -> List[InjectionPoint]:
        parts = urlsplit(url)
        base_url = urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, parts.fragment))

        points: List[InjectionPoint] = []
        for k, v in parse_qsl(parts.query, keep_blank_values=True):
            points.append(
                InjectionPoint(
                    url=base_url,
                    method="GET",
                    param=k,
                    value=v,
                )
            )
        return points

    def _extract_forms(self, base_url: str, html: str) -> List[InjectionPoint]:
        soup = BeautifulSoup(html, "html.parser")
        points: List[InjectionPoint] = []

        for form in soup.find_all("form"):
            method = (form.get("method") or "GET").strip().upper()
            action = (form.get("action") or "").strip()
            action_url = urljoin(base_url, action) if action else base_url

            # Collect parameters from inputs/selects/textareas
            for name, value in self._iter_form_fields(form):
                points.append(
                    InjectionPoint(
                        url=action_url,
                        method=method,
                        param=name,
                        value=value,
                    )
                )

        return points

    def _iter_form_fields(self, form_tag) -> Iterable[tuple[str, str]]:
        """Yield (name, value) for fuzzable fields in a form.

        Notes:
        - Ignores fields without a name.
        - Ignores submit/button/image by default.
        - For checkboxes/radios, only includes if checked; falls back to "on".
        - For select, picks selected option, else first.
        """

        # input
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

        # textarea
        for ta in form_tag.find_all("textarea"):
            name = (ta.get("name") or "").strip()
            if not name:
                continue
            yield name, (ta.text or "")

        # select
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

    def _dedupe(self, points: List[InjectionPoint]) -> List[InjectionPoint]:
        seen = set()
        out: List[InjectionPoint] = []
        for p in points:
            key = (p.url, p.method.upper(), p.param, p.value)
            if key in seen:
                continue
            seen.add(key)
            out.append(p)
        return out


def crawl(target_url: str, **kwargs) -> List[InjectionPoint]:
    """Convenience functional API."""

    return Crawler(**kwargs).crawl(target_url)

