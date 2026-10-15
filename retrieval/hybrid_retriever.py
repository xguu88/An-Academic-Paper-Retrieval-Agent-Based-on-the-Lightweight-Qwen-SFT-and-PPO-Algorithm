"""组合 BM25 与 Dense 排名并通过 RRF 返回统一 Hybrid 结果。"""

from dataclasses import dataclass

from src.retrieval.base import Retriever
from src.retrieval.rrf import reciprocal_rank_fusion
from src.retrieval.types import RetrievalResult


@dataclass(slots=True)
class HybridConfig:
    """Hybrid 两路召回深度和 RRF 平滑常数。"""

    bm25_top_n: int = 50
    dense_top_n: int = 50
    rrf_k: int = 60
    final_top_k: int = 20

    def validate(self) -> None:
        """检查全部融合参数为正整数。"""

        for name in ("bm25_top_n", "dense_top_n", "rrf_k", "final_top_k"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} 必须是大于 0 的整数")


class HybridRetriever(Retriever):
    """分别调用词法和向量 Retriever，再执行去重 RRF 融合。"""

    def __init__(
        self,
        bm25_retriever: Retriever,
        dense_retriever: Retriever,
        config: HybridConfig | None = None,
    ) -> None:
        """注入两路 Retriever，保持 Hybrid 与具体实现解耦。"""

        if not isinstance(bm25_retriever, Retriever):
            raise TypeError("bm25_retriever 必须实现 Retriever")
        if not isinstance(dense_retriever, Retriever):
            raise TypeError("dense_retriever 必须实现 Retriever")
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.config = config or HybridConfig()
        self.config.validate()

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """执行两路召回与 RRF，并补齐缺失一侧的 provenance 为 None。"""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query 必须是非空字符串")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k 必须是大于 0 的整数")
        bm25 = self.bm25_retriever.retrieve(query, top_k=self.config.bm25_top_n)
        dense = self.dense_retriever.retrieve(query, top_k=self.config.dense_top_n)
        fused = reciprocal_rank_fusion(
            {"bm25": bm25, "dense": dense},
            k=self.config.rrf_k,
            top_k=min(top_k, self.config.final_top_k),
        )
        for result in fused:
            result.metadata.setdefault("bm25_rank", None)
            result.metadata.setdefault("bm25_score", None)
            result.metadata.setdefault("dense_rank", None)
            result.metadata.setdefault("dense_score", None)
        return fused

