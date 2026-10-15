"""Deterministic result grouping using only retrieved metadata."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from src.retrieval.online.paper_reranker import OnlineRerankerConfig


@dataclass(frozen=True, slots=True)
class ResultOrganizerConfig:
    minimum_relevance_score: float = OnlineRerankerConfig().relevant_threshold
    recent_year_window: int = 3
    foundational_min_age: int = 10
    foundational_min_citations: int = 100

    def validate(self) -> None:
        if not 0 <= self.minimum_relevance_score <= 1:
            raise ValueError("minimum_relevance_score must be between 0 and 1")
        if min(self.recent_year_window, self.foundational_min_age, self.foundational_min_citations) < 0:
            raise ValueError("organizer thresholds must be non-negative")


def organize_results(
    candidates: list[dict[str, Any]], config: ResultOrganizerConfig | None = None
) -> list[dict[str, Any]]:
    selected = config or ResultOrganizerConfig()
    selected.validate()
    current_year = datetime.now(timezone.utc).year
    buckets: dict[str, list[str]] = {
        "core_recommendations": [],
        "recent_work": [],
        "foundational_work": [],
        "related_resources": [],
    }
    for item in candidates:
        identifier = str(item.get("id") or "")
        if item.get("candidate_type") != "paper":
            buckets["related_resources"].append(identifier)
            continue
        relevance = item.get("relevance_score")
        meets_relevance_gate = (
            isinstance(relevance, (int, float))
            and not isinstance(relevance, bool)
            and relevance >= selected.minimum_relevance_score
        )
        if meets_relevance_gate:
            buckets["core_recommendations"].append(identifier)
        year = item.get("publication_year")
        if meets_relevance_gate and isinstance(year, int) and year >= current_year - selected.recent_year_window:
            buckets["recent_work"].append(identifier)
        if (
            meets_relevance_gate and isinstance(year, int)
            and year <= current_year - selected.foundational_min_age
            and (item.get("citation_count") or 0) >= selected.foundational_min_citations
        ):
            buckets["foundational_work"].append(identifier)
    labels = {
        "core_recommendations": "Core recommendations",
        "recent_work": "Recent work",
        "foundational_work": "Foundational work",
        "related_resources": "Related resources",
    }
    return [
        {"id": key, "label": labels[key], "candidate_ids": values}
        for key, values in buckets.items() if values
    ]
