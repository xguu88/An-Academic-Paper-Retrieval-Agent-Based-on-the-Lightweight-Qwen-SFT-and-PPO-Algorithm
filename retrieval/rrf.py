"""实现支持任意多个检索排名的 Reciprocal Rank Fusion。"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from src.retrieval.types import RetrievalResult, RetrievalSource


class RRFValidationError(ValueError):
    """输入排名包含重复 ID、非法 rank 或非法融合参数。"""


@dataclass(slots=True)
class _FusedPaper:
    """内部累计单篇论文的 RRF 分数和来源信息。"""

    paper_id: str
    title: str
    score: float
    first_seen: int
    best_rank: int
    metadata: dict[str, float | int]


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[RetrievalResult]],
    *,
    k: int = 60,
    top_k: int | None = None,
) -> list[RetrievalResult]:
    """按 Σ 1/(k+rank) 融合，并拒绝同一 Retriever 内重复 paper_id。"""

    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise RRFValidationError("RRF k 必须是大于 0 的整数")
    if top_k is not None and (
        isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0
    ):
        raise RRFValidationError("top_k 必须是大于 0 的整数或 None")
    if not isinstance(rankings, Mapping) or not rankings:
        raise RRFValidationError("rankings 必须是非空映射")

    fused: dict[str, _FusedPaper] = {}
    seen_counter = 0
    for retriever_name, ranking in rankings.items():
        if not isinstance(retriever_name, str) or not retriever_name.strip():
            raise RRFValidationError("retriever name 必须是非空字符串")
        source_name = retriever_name.strip().lower()
        seen_in_ranking: set[str] = set()
        for result in ranking:
            if result.paper_id in seen_in_ranking:
                raise RRFValidationError(
                    f"{source_name} ranking 包含重复 paper_id: {result.paper_id}"
                )
            seen_in_ranking.add(result.paper_id)
            rank = result.rank
            if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
                raise RRFValidationError(f"{source_name} result rank 必须是正整数")
            contribution = 1.0 / (k + rank)
            if result.paper_id not in fused:
                fused[result.paper_id] = _FusedPaper(
                    paper_id=result.paper_id,
                    title=result.title,
                    score=0.0,
                    first_seen=seen_counter,
                    best_rank=rank,
                    metadata={},
                )
                seen_counter += 1
            item = fused[result.paper_id]
            item.score += contribution
            item.best_rank = min(item.best_rank, rank)
            item.metadata[f"{source_name}_rank"] = rank
            item.metadata[f"{source_name}_score"] = float(result.score)

    ordered = sorted(
        fused.values(),
        key=lambda item: (-item.score, item.best_rank, item.first_seen, item.paper_id),
    )
    if top_k is not None:
        ordered = ordered[:top_k]
    return [
        RetrievalResult(
            paper_id=item.paper_id,
            title=item.title,
            score=item.score,
            rank=rank,
            source=RetrievalSource.RRF,
            metadata={**item.metadata, "rrf_score": item.score},
        )
        for rank, item in enumerate(ordered, start=1)
    ]

