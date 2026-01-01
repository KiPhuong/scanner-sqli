"""core.requester

HTTP requester utilities for fuzzing.

Responsibilities
- Send GET/POST (or other common) requests.
- Inject a payload into a specific parameter.
- Return a structured response with timing/size/headers.

Design goals
- Reliability and simplicity for pentesting.
- Reasonable defaults; fully configurable timeout/retries.
- No JS/AJAX.

Notes on URL-encoding
- For GET: requests will handle query encoding when passing params dict.
- For POST (form-encoded): requests handles encoding when passing data dict.
- "optional URL encoding" here means you can pre-encode the payload before
  handing it to requests, which can be useful when you want double-encoding or
  consistent encoding behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
from urllib.parse import quote_plus
import time

import requests


@dataclass(frozen=True)
class FuzzResponse:
    status_code: int
    response_text: str
    response_length: int
    response_time: float  # seconds
    headers: Dict[str, str]
    final_url: str


class Requester:
    def __init__(
        self,
        timeout: int = 15,
        retries: int = 2,
        backoff: float = 0.5,
        verify_tls: bool = True,
        allow_redirects: bool = True,
        user_agent: str = "scanner-sqli-requester/1.0",
        session: Optional[requests.Session] = None,
        max_body_bytes: int = 2_000_000,
    ):
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self.verify_tls = verify_tls
        self.allow_redirects = allow_redirects
        self.max_body_bytes = max_body_bytes

        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", user_agent)
        self.session.headers.setdefault("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")

    def send(
        self,
        url: str,
        method: str,
        params: Dict[str, str],
        inject_param: str,
        payload: str,
        url_encode_payload: bool = False,
    ) -> FuzzResponse:
        """Send a request with payload injected into inject_param."""

        method_u = (method or "GET").upper()
        payload_value = quote_plus(payload) if url_encode_payload else payload

        final_params = dict(params)
        final_params[inject_param] = payload_value

        last_err: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            try:
                start = time.perf_counter()
                resp = self._request(method_u, url, final_params)
                elapsed = time.perf_counter() - start

                text = self._safe_text(resp)
                return FuzzResponse(
                    status_code=resp.status_code,
                    response_text=text,
                    response_length=len(resp.content or b""),
                    response_time=elapsed,
                    headers={k: v for k, v in resp.headers.items()},
                    final_url=str(resp.url),
                )
            except (requests.RequestException, ValueError) as e:
                last_err = e
                if attempt >= self.retries:
                    break
                time.sleep(self.backoff * (2**attempt))

        # If we get here, all retries failed
        raise last_err  # type: ignore[misc]

    def _request(self, method: str, url: str, params: Dict[str, str]) -> requests.Response:
        if method == "GET":
            r = self.session.get(
                url,
                params=params,
                timeout=self.timeout,
                verify=self.verify_tls,
                allow_redirects=self.allow_redirects,
                stream=True,
            )
        else:
            # Default to form-encoded body for non-GET
            r = self.session.request(
                method,
                url,
                data=params,
                timeout=self.timeout,
                verify=self.verify_tls,
                allow_redirects=self.allow_redirects,
                stream=True,
            )

        # Do not raise_for_status: during fuzzing, 4xx/5xx are meaningful signals
        self._limit_body(r)
        return r

    def _limit_body(self, resp: requests.Response) -> None:
        content = b""
        for chunk in resp.iter_content(chunk_size=64 * 1024):
            if not chunk:
                continue
            content += chunk
            if len(content) > self.max_body_bytes:
                break
        resp._content = content
        if resp.encoding is None:
            resp.encoding = "utf-8"

    def _safe_text(self, resp: requests.Response) -> str:
        enc = resp.encoding or "utf-8"
        try:
            return (resp.content or b"").decode(enc, errors="replace")
        except LookupError:
            return (resp.content or b"").decode("utf-8", errors="replace")


def send_injected(
    url: str,
    method: str,
    params: Dict[str, str],
    inject_param: str,
    payload: str,
    **kwargs,
) -> FuzzResponse:
    """Convenience functional API."""

    return Requester(**kwargs).send(
        url=url,
        method=method,
        params=params,
        inject_param=inject_param,
        payload=payload,
        url_encode_payload=kwargs.get("url_encode_payload", False),
    )

