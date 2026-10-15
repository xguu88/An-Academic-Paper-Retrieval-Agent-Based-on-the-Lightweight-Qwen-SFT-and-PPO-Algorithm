"""定义与 Agent、Environment 和训练框架完全解耦的检索接口。"""

from abc import ABC, abstractmethod

from src.retrieval.types import RetrievalResult


class Retriever(ABC):
    """将自然语言 query 映射为有序 RetrievalResult 列表。"""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """返回按相关性从高到低排列的前 top_k 篇论文。"""

