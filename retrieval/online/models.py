"""Data types returned by online search backends."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
from typing import Literal
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
import re


_TRACKING_PARAMETERS = {"gclid", "fbclid", "ref", "source"}

CandidateType = Literal["paper", "related_resource"]


def normalize_doi(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.strip().lower()
    normalized = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", normalized)
    normalized = normalized.removeprefix("doi:").strip()
    return normalized or None


def normalize_title(value: str) -> str:
    return " ".join(re.findall(r"[\w]+", value.casefold(), flags=re.UNICODE))


def canonicalize_http_url(value: str) -> str | None:
    """Return a stable safe URL, rejecting non-HTTP browser schemes."""

    try:
        parsed = urlsplit(value.strip())
    except (TypeError, ValueError):
        return None
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return None
    query = [
        (key, item)
        for key, item in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.lower().startswith("utm_") and key.lower() not in _TRACKING_PARAMETERS
    ]
    return urlunsplit(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path.rstrip("/") or "/",
            urlencode(query),
            "",
        )
    )


@dataclass(slots=True)
class OnlinePaperCandidate:
    id: str
    title: str
    url: str
    snippet: str
    source: str
    domain: str
    rank: int
    query_used: str
    published_date: str | None = None
    first_seen_step: int = 0
    queries_seen_in: list[str] = field(default_factory=list)
    best_rank: int = 0
    academic_score: int = 0
    candidate_type: CandidateType = "related_resource"
    search_rank: int = 0
    recommended_rank: int = 0
    external_ids: dict[str, str] = field(default_factory=dict)
    abstract: str | None = None
    authors: list[str] = field(default_factory=list)
    publication_year: int | None = None
    venue: str | None = None
    citation_count: int | None = None
    references: list[str] = field(default_factory=list)
    cited_by_count: int | None = None
    relevance_score: float | None = None
    relevance_level: str | None = None
    authority_score: float | None = None
    freshness_score: float | None = None
    source_score: float | None = None
    final_score: float | None = None

    @classmethod
    def from_search_result(
        cls, item: dict[str, object], *, query: str, fallback_rank: int
    ) -> "OnlinePaperCandidate | None":
        title = str(item.get("title") or "").strip()
        canonical = canonicalize_http_url(str(item.get("link") or ""))#取出原始链接，URL标准化
        if not title or canonical is None:
            return None
        domain = urlsplit(canonical).netloc.lower().removeprefix("www.")#解析标准话后的URL，拿到域名netloc
        raw_rank = item.get("position", fallback_rank)
        rank = raw_rank if isinstance(raw_rank, int) and raw_rank > 0 else fallback_rank#取搜索结果里的`position`，搜索引擎返回的页面排序；取不到就先用`fallback_rank`
        source = str(item.get("source") or domain).strip() or domain#优先取搜索结果自带的`source`字段；如果没有 source、source 为空，就直接用域名当来源。
        return cls(
            id="web-" + sha256(canonical.encode("utf-8")).hexdigest()[:16],
            title=title,
            url=canonical,
            snippet=str(item.get("snippet") or "").strip(),
            source=source,
            domain=domain,
            rank=rank,
            query_used=query,
            published_date=(str(item["date"]).strip() if item.get("date") else None),
            queries_seen_in=[query],
            best_rank=rank,
            search_rank=rank,
        )

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        # ``rank`` remains the backwards-compatible Serper rank field.
        data["rank"] = self.search_rank or self.rank
        data["search_rank"] = self.search_rank or self.rank
        return data

    def dedup_key(self) -> str:
        doi = normalize_doi(self.external_ids.get("doi"))
        if doi:
            return f"doi:{doi}"
        for name in ("openalex_id", "semantic_scholar_id", "arxiv_id"):
            value = self.external_ids.get(name)
            if value:
                return f"{name}:{value.casefold()}"
        if self.id.startswith("web-"):
            return f"url:{self.url}"
        return f"title:{normalize_title(self.title)}"
