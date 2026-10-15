"""Map general web results into conservatively ranked academic candidates."""

from __future__ import annotations

from src.retrieval.online.models import OnlinePaperCandidate, normalize_title
from src.retrieval.online.openalex_client import OpenAlexClient, OpenAlexError
from src.retrieval.online.serper_client import SerperClient, SerperError


PAPER_DOMAINS = (
    "arxiv.org", "openreview.net", "semanticscholar.org", "aclanthology.org",
    "pubmed.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov", "nature.com", "science.org",
    "springer.com", "link.springer.com", "sciencedirect.com", "ieee.org",
    "ieeexplore.ieee.org", "dl.acm.org", "papers.neurips.cc",
    "proceedings.neurips.cc", "proceedings.mlr.press",
)
_LOW_QUALITY_DOMAINS = (
    "reddit.com", "zhihu.com", "medium.com", "substack.com", "youtube.com",
    "facebook.com", "x.com", "twitter.com", "github.com",
)
_PAPER_HINTS = ("paper", "proceedings", "journal", "conference", "doi", "pdf", "preprint")


def _domain_matches(domain: str, candidates: tuple[str, ...]) -> bool:
    return any(domain == item or domain.endswith("." + item) for item in candidates)


def classify_candidate(candidate: OnlinePaperCandidate) -> str:
    """Known publication hosts are papers; all other useful links stay resources."""

    return "paper" if _domain_matches(candidate.domain, PAPER_DOMAINS) else "related_resource"


def academic_score(candidate: OnlinePaperCandidate) -> int:
    domain = candidate.domain
    text = f"{candidate.title} {candidate.snippet} {candidate.url}".lower()
    score = 0
    if _domain_matches(domain, PAPER_DOMAINS):
        score += 100
    if domain.endswith(".edu") or domain.endswith(".ac.uk"):
        score += 60
    score += 8 * sum(hint in text for hint in _PAPER_HINTS)
    if _domain_matches(domain, _LOW_QUALITY_DOMAINS):
        score -= 80
    if any(word in text for word in ("blog", "marketing", "breaking news", "readme")):
        score -= 25
    return score


def _merge_candidate(
    existing: OnlinePaperCandidate, incoming: OnlinePaperCandidate
) -> OnlinePaperCandidate:
    """Merge richer academic metadata without inventing unavailable fields."""

    for field_name in (
        "abstract", "published_date", "publication_year", "venue", "citation_count",
        "cited_by_count",
    ):
        if getattr(existing, field_name) is None and getattr(incoming, field_name) is not None:
            setattr(existing, field_name, getattr(incoming, field_name))
    if not existing.authors and incoming.authors:
        existing.authors = list(incoming.authors)
    if not existing.references and incoming.references:
        existing.references = list(incoming.references)
    existing.external_ids.update(incoming.external_ids)
    existing.academic_score = max(existing.academic_score, incoming.academic_score)
    existing.best_rank = min(existing.best_rank, incoming.best_rank)
    existing.search_rank = min(existing.search_rank, incoming.search_rank)
    existing.rank = existing.search_rank
    for query in incoming.queries_seen_in:
        if query not in existing.queries_seen_in:
            existing.queries_seen_in.append(query)
    if existing.source != "openalex" and incoming.source == "openalex":
        existing.source = incoming.source
    return existing


class AcademicSearch:
    """OpenAlex-first academic retrieval with optional Serper complement."""

    def __init__(
        self,
        client: SerperClient | None = None,
        *,
        openalex_client: OpenAlexClient | None = None,
    ) -> None:
        if client is None and openalex_client is None:
            raise ValueError("at least one academic search backend is required")
        self.client = client
        self.openalex_client = openalex_client
        self.last_errors: list[str] = []

    @property
    def request_counts(self) -> dict[str, int]:
        return {
            "openalex": self.openalex_client.request_count if self.openalex_client else 0,
            "serper": self.client.request_count if self.client else 0,
            "citation": self.openalex_client.citation_request_count if self.openalex_client else 0,
        }

    @property
    def latency_breakdown(self) -> dict[str, float]:
        return {
            "openalex_latency_ms": self.openalex_client.latency_ms if self.openalex_client else 0.0,
            "serper_latency_ms": self.client.latency_ms if self.client else 0.0,
        }

    def _serper_candidates(self, query: str, top_k: int) -> list[OnlinePaperCandidate]:
        if self.client is None:
            return []
        payload = self.client.search(query, num=top_k)
        candidates: list[OnlinePaperCandidate] = []
        for index, raw in enumerate(payload.get("organic", []), start=1):
            if not isinstance(raw, dict):
                continue
            candidate = OnlinePaperCandidate.from_search_result(raw, query=query, fallback_rank=index)
            if candidate is None:
                continue
            candidate.academic_score = academic_score(candidate)
            candidate.candidate_type = classify_candidate(candidate)
            candidates.append(candidate)
        return candidates

    def search(self, query: str, *, top_k: int = 10) -> list[OnlinePaperCandidate]:#两种搜索并行尝试
        self.last_errors = []
        candidates: list[OnlinePaperCandidate] = []
        if self.openalex_client is not None:
            try:
                candidates.extend(self.openalex_client.search(query, per_page=top_k))
            except OpenAlexError as exc:
                self.last_errors.append(f"OpenAlex: {exc}")
        if self.client is not None:
            try:
                candidates.extend(self._serper_candidates(query, top_k))
            except SerperError as exc:
                self.last_errors.append(f"Serper: {exc}")
        if not candidates and self.last_errors:
            raise RuntimeError("; ".join(self.last_errors))
        deduplicated: dict[str, OnlinePaperCandidate] = {}
        title_keys: dict[str, str] = {}
        for candidate in candidates:
            key = candidate.dedup_key()
            title_key = normalize_title(candidate.title)
            key = title_keys.get(title_key, key)
            if key in deduplicated:
                _merge_candidate(deduplicated[key], candidate)
            else:
                deduplicated[key] = candidate
                title_keys[title_key] = key
        candidates = list(deduplicated.values())
        candidates.sort(
            key=lambda item: (-item.academic_score, item.search_rank, item.title.lower())
        )
        """
        1. `-item.academic_score`：**学术分数降序**（分数越高排越前面，负号实现降序）
        2. `item.search_rank`：检索原始排名升序；学术分数相同，原始搜索结果靠前的优先
        3. `item.title.lower()`：标题小写字典序；前面两项都一样，按标题字母排序，保证稳定排序
        """
        return candidates

    def expand_citations(
        self, openalex_id: str, direction: str, *, top_k: int = 10
    ) -> list[OnlinePaperCandidate]:
        if self.openalex_client is None:
            raise RuntimeError("OpenAlex is required for citation expansion")
        return self.openalex_client.expand(openalex_id, direction, per_page=top_k)
