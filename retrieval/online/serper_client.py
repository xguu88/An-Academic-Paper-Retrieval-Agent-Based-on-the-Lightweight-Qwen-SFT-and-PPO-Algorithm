"""Minimal Serper Search API client with bounded error handling."""

from __future__ import annotations

import json
from time import perf_counter
from typing import Any

import httpx


SERPER_SEARCH_ENDPOINT = "https://google.serper.dev/search"


class SerperError(RuntimeError):
    """Safe user-facing Serper failure without credentials or response bodies."""


class SerperConfigurationError(SerperError):
    pass


class SerperAuthenticationError(SerperError):
    pass


class SerperRateLimitError(SerperError):
    pass


class SerperServiceError(SerperError):
    pass


class SerperResponseError(SerperError):
    pass


class SerperBudgetError(SerperError):
    pass


class SerperClient:
    def __init__(
        self,
        api_key: str,
        *,
        timeout_seconds: float = 10.0,
        client: httpx.Client | None = None,
        network_retries: int = 1,
        max_requests: int = 3,
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise SerperConfigurationError(
                "SERPER_API_KEY is required for online retrieval"
            )
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._api_key = api_key.strip()
        self._client = client or httpx.Client()
        self.timeout_seconds = timeout_seconds
        self.network_retries = max(0, min(int(network_retries), 1))
        self.max_requests = max(1, min(int(max_requests), 3))
        self.request_count = 0
        self.latency_ms = 0.0

    def search(self, query: str, *, num: int = 10) -> dict[str, Any]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Serper query must be non-empty")
        if num <= 0:
            raise ValueError("Serper num must be positive")
        attempts = self.network_retries + 1
        response: httpx.Response | None = None
        for attempt in range(attempts):
            if self.request_count >= self.max_requests:
                raise SerperBudgetError("Serper request budget exhausted")
            self.request_count += 1
            started = perf_counter()
            try:
                response = self._client.post(
                    SERPER_SEARCH_ENDPOINT,
                    headers={
                        "X-API-KEY": self._api_key,
                        "Content-Type": "application/json",
                    },
                    json={"q": query.strip(), "num": min(num, 100)},
                    timeout=self.timeout_seconds,
                )
                break
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                if attempt + 1 >= attempts:
                    raise SerperServiceError("Serper network request failed") from exc
            finally:
                self.latency_ms += (perf_counter() - started) * 1000
        assert response is not None
        if response.status_code in {401, 403}:
            raise SerperAuthenticationError("Serper authentication failed")
        if response.status_code == 429:
            raise SerperRateLimitError("Serper rate limit exceeded")
        if response.status_code >= 500:
            raise SerperServiceError("Serper service is temporarily unavailable")
        if response.is_error:
            raise SerperServiceError(f"Serper request failed with HTTP {response.status_code}")
        try:
            payload = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise SerperResponseError("Serper returned malformed JSON") from exc
        if not isinstance(payload, dict):
            raise SerperResponseError("Serper response must be a JSON object")
        organic = payload.get("organic", [])
        if not isinstance(organic, list):
            raise SerperResponseError("Serper organic results must be a list")
        return payload
