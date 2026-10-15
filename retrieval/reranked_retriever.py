"""实现 base retrieval → candidate pool → reranker 的组合 Retriever。"""

from dataclasses import dataclass

from src.retrieval.base import Retriever
from src.retrieval.reranker import Reranker
from src.retrieval.types import RetrievalResult


@dataclass(slots=True)
class RerankedRetrieverConfig:
    """定义候选召回深度与最终最大返回深度。"""

    candidate_top_n: int = 50
    final_top_k: int = 20

    def validate(self) -> None:
        """检查重排深度为正整数且最终深度不超过候选深度。"""

        for name in ("candidate_top_n", "final_top_k"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} 必须是大于 0 的整数")
        if self.final_top_k > self.candidate_top_n:
            raise ValueError("final_top_k 不能大于 candidate_top_n")


class RerankedRetriever(Retriever):
    """可包装 Dense、Hybrid 或任意标准 Retriever，而不改变基础召回器。"""

    def __init__(
        self,
        base_retriever: Retriever,
        reranker: Reranker,
        config: RerankedRetrieverConfig | None = None,
    ) -> None:
        """注入基础 Retriever 和独立 Reranker。"""

        if not isinstance(base_retriever, Retriever):
            raise TypeError("base_retriever 必须实现 Retriever")
        if not isinstance(reranker, Reranker):
            raise TypeError("reranker 必须实现 Reranker")
        self.base_retriever = base_retriever
        self.reranker = reranker
        self.config = config or RerankedRetrieverConfig()
        self.config.validate()

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """先取固定候选池，再按调用深度和配置上限输出重排结果。"""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query 必须是非空字符串")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k 必须是大于 0 的整数")
        _, reranked = self.retrieve_with_candidates(query)
        return reranked[: min(top_k, self.config.final_top_k)]

    def retrieve_with_candidates(
        self,
        query: str,
    ) -> tuple[list[RetrievalResult], list[RetrievalResult]]:
        """返回完整重排前后候选池，供集合保持与排名移动诊断使用。"""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query 必须是非空字符串")
        candidates = self.base_retriever.retrieve(query, self.config.candidate_top_n)
        reranked = self.reranker.rerank(query, candidates, len(candidates))
        if {item.paper_id for item in candidates} != {item.paper_id for item in reranked}:
            raise RuntimeError("Reranker 改变了候选 paper set")
        return candidates, reranked
