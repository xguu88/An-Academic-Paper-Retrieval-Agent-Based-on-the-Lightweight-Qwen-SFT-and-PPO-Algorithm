"""定义独立于召回器、Agent 与训练框架的重排接口。"""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from src.retrieval.types import RetrievalResult


class Reranker(ABC):
    """仅重新排列已有候选论文，不执行任何新召回。"""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """返回按新相关性分数排序的前 top_k 个候选结果。"""

