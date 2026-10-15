"""OpenAlex Works API client and candidate normalization."""

from __future__ import annotations

import json
from time import perf_counter
from typing import Any
from urllib.parse import urlsplit

import httpx

from src.retrieval.online.models import OnlinePaperCandidate, canonicalize_http_url, normalize_doi


OPENALEX_WORKS_ENDPOINT = "https://api.openalex.org/works"


class OpenAlexError(RuntimeError):
    pass


class OpenAlexRateLimitError(OpenAlexError):
    pass


class OpenAlexServiceError(OpenAlexError):
    pass


class OpenAlexResponseError(OpenAlexError):
    pass


def reconstruct_abstract(index: object) -> str | None:
    if not isinstance(index, dict):
        return None
    positioned: list[tuple[int, str]] = []
    for word, positions in index.items():
        if not isinstance(word, str) or not isinstance(positions, list):
            continue
        positioned.extend((position, word) for position in positions if isinstance(position, int))
    if not positioned:
        return None
    return " ".join(word for _, word in sorted(positioned))


def openalex_work_to_candidate(
    work: dict[str, Any], *, query: str, rank: int
) -> OnlinePaperCandidate | None:
    title = str(work.get("title") or work.get("display_name") or "").strip()
    openalex_url = str(work.get("id") or "").strip()
    primary = work.get("primary_location") if isinstance(work.get("primary_location"), dict) else {}
    landing = str(primary.get("landing_page_url") or openalex_url)
    url = canonicalize_http_url(landing) or canonicalize_http_url(openalex_url)
    if not title or url is None:
        return None
    source_info = primary.get("source") if isinstance(primary.get("source"), dict) else {}
    venue = str(source_info.get("display_name") or "").strip() or None
    authors: list[str] = []
    for authorship in work.get("authorships", []) if isinstance(work.get("authorships"), list) else []:
        if not isinstance(authorship, dict):
            continue
        author = authorship.get("author")
        if isinstance(author, dict) and author.get("display_name"):
            authors.append(str(author["display_name"]))
    ids = work.get("ids") if isinstance(work.get("ids"), dict) else {}
    doi = normalize_doi(str(work.get("doi") or ids.get("doi") or ""))
    external_ids = {
        "openalex_id": openalex_url.rsplit("/", 1)[-1],
        **({"doi": doi} if doi else {}),
    }
    if ids.get("pmid"):
        external_ids["pmid"] = str(ids["pmid"])
    abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
    domain = urlsplit(url).netloc.lower().removeprefix("www.")
    cited = work.get("cited_by_count")
    year = work.get("publication_year")
    references = [
        str(item).rsplit("/", 1)[-1]
        for item in work.get("referenced_works", [])
        if isinstance(item, str)
    ] if isinstance(work.get("referenced_works"), list) else []
    return OnlinePaperCandidate(
        id="openalex-" + external_ids["openalex_id"],
        title=title,
        url=url,
        snippet=(abstract or "")[:500],
        source="openalex",
        domain=domain,
        rank=rank,
        query_used=query,
        published_date=str(work.get("publication_date") or "") or None,
        queries_seen_in=[query],
        best_rank=rank,
        candidate_type="paper",
        search_rank=rank,
        external_ids=external_ids,
        abstract=abstract,
        authors=authors,
        publication_year=year if isinstance(year, int) else None,
        venue=venue,
        citation_count=cited if isinstance(cited, int) else None,
        references=references,
        cited_by_count=cited if isinstance(cited, int) else None,
    )


class OpenAlexClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        timeout_seconds: float = 10.0,
        client: httpx.Client | None = None,
        network_retries: int = 1,
        max_requests: int = 6,
    ) -> None:
        self.api_key = api_key.strip() if api_key else None
        self.timeout_seconds = timeout_seconds
        self.client = client or httpx.Client()
        self.network_retries = max(0, min(network_retries, 1))
        self.max_requests = max(1, max_requests)
        self.request_count = 0
        self.citation_request_count = 0
        self.latency_ms = 0.0

    def _get(self, params: dict[str, object], *, citation: bool = False) -> dict[str, Any]:
        if self.request_count >= self.max_requests:
            raise OpenAlexServiceError("OpenAlex request budget exhausted")
        request_params = dict(params)
        if self.api_key:
            request_params["api_key"] = self.api_key
        response: httpx.Response | None = None
        for attempt in range(self.network_retries + 1):
            if self.request_count >= self.max_requests:
                raise OpenAlexServiceError("OpenAlex request budget exhausted")
            self.request_count += 1
            if citation:
                self.citation_request_count += 1
            try:
                started = perf_counter()
                response = self.client.get(
                    OPENALEX_WORKS_ENDPOINT,
                    params=request_params,
                    timeout=self.timeout_seconds,
                )
                break
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                if attempt >= self.network_retries:
                    raise OpenAlexServiceError("OpenAlex network request failed") from exc
            finally:
                self.latency_ms += (perf_counter() - started) * 1000
        assert response is not None
        if response.status_code == 429:
            raise OpenAlexRateLimitError("OpenAlex rate limit exceeded")
        if response.status_code >= 500:
            raise OpenAlexServiceError("OpenAlex service is temporarily unavailable")
        if response.is_error:
            raise OpenAlexServiceError(f"OpenAlex request failed with HTTP {response.status_code}")
        try:
            payload = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise OpenAlexResponseError("OpenAlex returned malformed JSON") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("results", []), list):
            raise OpenAlexResponseError("OpenAlex response schema is invalid")
        return payload

    def search(self, query: str, *, per_page: int = 10) -> list[OnlinePaperCandidate]:
        #利用分隔符分开检索
        payload = self._get({"search": query.strip(), "per_page": min(max(per_page, 1), 100)})
        candidates: list[OnlinePaperCandidate] = []
        for rank, work in enumerate(payload.get("results", []), start=1):
            if isinstance(work, dict):
                candidate = openalex_work_to_candidate(work, query=query, rank=rank)
                if candidate is not None:
                    candidates.append(candidate)
        return candidates

    def expand(self, openalex_id: str, direction: str, *, per_page: int = 10) -> list[OnlinePaperCandidate]:
        normalized = openalex_id.rsplit("/", 1)[-1]
        if direction == "citations":
            params: dict[str, object] = {"filter": f"cites:{normalized}", "per_page": per_page}
        elif direction == "references":
            work = self._get({"filter": f"openalex:{normalized}", "per_page": 1}, citation=True)
            results = work.get("results", [])
            ids = results[0].get("referenced_works", []) if results and isinstance(results[0], dict) else []
            short_ids = [str(item).rsplit("/", 1)[-1] for item in ids[:per_page] if isinstance(item, str)]
            if not short_ids:
                return []
            params = {"filter": "openalex:" + "|".join(short_ids), "per_page": per_page}
        else:
            raise ValueError("direction must be references or citations")
        payload = self._get(params, citation=True)
        return [
            candidate
            for rank, work in enumerate(payload.get("results", []), start=1)
            if isinstance(work, dict)
            and (candidate := openalex_work_to_candidate(work, query=f"citation:{normalized}", rank=rank)) is not None
        ]
