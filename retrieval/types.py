"""定义检索层对外返回的轻量、统一结果结构。"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RetrievalSource(str, Enum):
    """检索结果的产生阶段。"""

    BM25 = "bm25"
    DENSE = "dense"
    RRF = "rrf"
    RERANKER = "reranker"
    CITATION = "citation"


@dataclass(slots=True)
class RetrievalResult:
    """只携带排序所需的轻量信息，不直接保存论文正文。"""

    paper_id: str
    title: str
    score: float = 0.0
    rank: int | None = None
    source: RetrievalSource | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转换为可 JSON 序列化的字典。"""

        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "score": self.score,
            "rank": self.rank,
            "source": self.source.value if self.source is not None else None,
            "metadata": dict(self.metadata),
        }

