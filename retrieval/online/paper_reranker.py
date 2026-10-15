"""Fine-grained online query-paper relevance reranking."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import exp, log1p
from collections.abc import Sequence

import numpy as np

from src.retrieval.online.models import OnlinePaperCandidate
from src.retrieval.pair_scorer import PairScorer


@dataclass(frozen=True, slots=True)
class OnlineRerankerConfig:
    highly_relevant_threshold: float = 0.75
    relevant_threshold: float = 0.45
    relevance_weight: float = 0.85
    authority_weight: float = 0.08
    freshness_weight: float = 0.04
    source_weight: float = 0.03
    cross_encoder_outputs_logits: bool = True

    def validate(self) -> None:
        if not 0 <= self.relevant_threshold <= self.highly_relevant_threshold <= 1:
            raise ValueError("invalid relevance thresholds")
        total = self.relevance_weight + self.authority_weight + self.freshness_weight + self.source_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError("online reranker weights must sum to 1")


class OnlinePaperReranker:
    """English MS MARCO CrossEncoder is the primary signal; metadata stays lightweight."""

    def __init__(self, scorer: PairScorer, config: OnlineRerankerConfig | None = None) -> None:
        self.scorer = scorer
        self.config = config or OnlineRerankerConfig()
        self.config.validate()

    def rerank(self, query: str, candidates: Sequence[OnlinePaperCandidate]) -> list[OnlinePaperCandidate]:
        if not candidates:
            return []
        papers = [item for item in candidates if item.candidate_type == "paper"]
        resources = [item for item in candidates if item.candidate_type != "paper"]
        pairs = [(query, "\n".join(filter(None, (item.title, item.abstract or item.snippet, item.venue or "")))) for item in papers]
        raw = np.asarray(self.scorer.score_pairs(pairs), dtype=np.float64)
        if raw.shape != (len(papers),) or not np.all(np.isfinite(raw)):
            raise ValueError("online PairScorer returned invalid scores")
        current_year = datetime.now(timezone.utc).year
        for candidate, value in zip(papers, raw, strict=True):
            relevance = 1.0 / (1.0 + exp(-float(value))) if self.config.cross_encoder_outputs_logits else min(max(float(value), 0.0), 1.0)
            authority = min(log1p(max(candidate.citation_count or 0, 0)) / log1p(1000), 1.0)
            age = max(0, current_year - (candidate.publication_year or current_year))
            freshness = max(0.0, 1.0 - age / 15.0)
            source = 1.0 if candidate.source == "openalex" else 0.7
            final = (
                self.config.relevance_weight * relevance
                + self.config.authority_weight * authority
                + self.config.freshness_weight * freshness
                + self.config.source_weight * source
            )
            candidate.relevance_score = relevance
            candidate.relevance_level = (
                "highly_relevant" if relevance >= self.config.highly_relevant_threshold
                else "relevant" if relevance >= self.config.relevant_threshold
                else "partially_relevant"
            )
            candidate.authority_score = authority
            candidate.freshness_score = freshness
            candidate.source_score = source
            candidate.final_score = final
        papers.sort(key=lambda item: (-(item.final_score or 0.0), item.search_rank, item.title.casefold()))
        resources.sort(key=lambda item: (-item.academic_score, item.search_rank, item.title.casefold()))
        ranked = papers + resources
        for rank, item in enumerate(ranked, start=1):
            item.recommended_rank = rank
        return ranked
