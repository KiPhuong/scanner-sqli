"""core.baseline

Baseline response collection.

The baseline is a "clean" request (no SQLi payload) for a given injection
point. During fuzzing, later responses can be compared (delta time/length,
status changes, semantic similarity changes, etc.).

This module is intentionally self-contained and uses:
- requests for HTTP
- a lightweight, deterministic embedding to avoid heavy ML deps

If you later add a real embedding model, keep the BaselineResponse interface
stable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import time

import requests


def _simple_embedding(text: str, dim: int = 256) -> list[float]:
    """Create a lightweight semantic-ish embedding.

    This is NOT a true semantic embedding model.

    Rationale:
    - Reliability and simplicity for pentesting.
    - Avoid heavy dependencies / model downloads.
    - Still provides a stable vector useful for rough similarity checks.

    Implementation:
    - Tokenize on whitespace
    - Hash each token into a fixed-size bag-of-words vector
    - L2 normalize
    """

    vec = [0.0] * dim
    for tok in text.split():
        # Deterministic hashing (avoid Python's randomized hash)
        h = 2166136261
        for ch in tok:
            h ^= ord(ch)
            h = (h * 16777619) & 0xFFFFFFFF
        idx = h % dim
        vec[idx] += 1.0

    norm = sum(v * v for v in vec) ** 0.5
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


@dataclass(frozen=True)
class BaselineResponse:
    """Baseline metrics for a request."""

    url: str
    method: str
    params: Dict[str, str]

    status_code: int
    elapsed: float  # seconds
    length: int  # bytes
    embedding: list[float]

    @classmethod
    def from_injection_point(
        cls,
        injection_point,
        all_params: Optional[Dict[str, str]] = None,
        timeout: int = 15,
        verify_tls: bool = True,
        allow_redirects: bool = True,
        user_agent: str = "scanner-sqli-baseline/1.0",
        session: Optional[requests.Session] = None,
        max_body_bytes: int = 2_000_000,
    ) -> "BaselineResponse":
        """Create a baseline for an injection point.

        Parameters
        - injection_point: expected to have url, method, param, value attributes
          (e.g., core.crawler.InjectionPoint).
        - all_params: full parameter dictionary to send.
            If omitted, will send a dict containing just the injection point
            parameter/value.

        The "no SQLi payload" requirement is satisfied by sending the original
        parameter values as-is.
        """

        params = dict(all_params) if all_params is not None else {injection_point.param: injection_point.value}

        sess = session or requests.Session()
        sess.headers.setdefault("User-Agent", user_agent)
        sess.headers.setdefault("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")

        method = (injection_point.method or "GET").upper()
        url = injection_point.url

        start = time.perf_counter()
        resp = _send_request(
            sess,
            method=method,
            url=url,
            params=params,
            timeout=timeout,
            verify_tls=verify_tls,
            allow_redirects=allow_redirects,
            max_body_bytes=max_body_bytes,
        )
        elapsed = time.perf_counter() - start

        body = resp.content or b""
        emb = _simple_embedding(_safe_decode(body, resp.encoding))

        return cls(
            url=url,
            method=method,
            params=params,
            status_code=resp.status_code,
            elapsed=elapsed,
            length=len(body),
            embedding=emb,
        )


def _send_request(
    session: requests.Session,
    method: str,
    url: str,
    params: Dict[str, str],
    timeout: int,
    verify_tls: bool,
    allow_redirects: bool,
    max_body_bytes: int,
) -> requests.Response:
    method = method.upper()

    if method == "GET":
        r = session.get(
            url,
            params=params,
            timeout=timeout,
            verify=verify_tls,
            allow_redirects=allow_redirects,
            stream=True,
        )
    else:
        # Default to form-encoded POST for baseline
        r = session.request(
            method,
            url,
            data=params,
            timeout=timeout,
            verify=verify_tls,
            allow_redirects=allow_redirects,
            stream=True,
        )

    r.raise_for_status()

    # Limit response size
    content = b""
    for chunk in r.iter_content(chunk_size=64 * 1024):
        if not chunk:
            continue
        content += chunk
        if len(content) > max_body_bytes:
            break
    r._content = content
    return r


def _safe_decode(body: bytes, encoding: Optional[str]) -> str:
    enc = encoding or "utf-8"
    try:
        return body.decode(enc, errors="replace")
    except LookupError:
        return body.decode("utf-8", errors="replace")

