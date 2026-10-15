"""基于 rank_bm25 的纯 CPU 关键词检索实现。"""

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from src.retrieval.base import Retriever
from src.retrieval.corpus import PaperCorpus
from src.retrieval.document_text import retrieval_text_fields
from src.retrieval.paper import PaperDocument
from src.retrieval.tokenization import tokenize
from src.retrieval.types import RetrievalResult, RetrievalSource


class BM25DependencyError(RuntimeError):
    """当前 Python 环境缺少 rank_bm25 时抛出的明确依赖错误。"""


@dataclass(slots=True)
class BM25Config:
    """BM25 参数与字段权重近似配置。"""

    fields: dict[str, float] = field(
        default_factory=lambda: {"title": 2.0, "abstract": 1.0, "keywords": 1.5}
    )
    top_k: int = 20
    k1: float = 1.5
    b: float = 0.75
    weight_scale: int = 2

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "BM25Config":
        """从 YAML 解码后的映射构造配置。"""

        return cls(
            fields=dict(value.get("fields", {})) or cls().fields,
            top_k=value.get("top_k", 20),
            k1=value.get("k1", 1.5),
            b=value.get("b", 0.75),
            weight_scale=value.get("weight_scale", 2),
        )

    def validate(self) -> None:
        """拒绝无效 BM25 参数和未知索引字段。"""

        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k <= 0:
            raise ValueError("bm25.top_k 必须是大于 0 的整数")
        if not isinstance(self.weight_scale, int) or self.weight_scale <= 0:
            raise ValueError("bm25.weight_scale 必须是大于 0 的整数")
        if not isinstance(self.k1, (int, float)) or self.k1 <= 0:
            raise ValueError("bm25.k1 必须大于 0")
        if not isinstance(self.b, (int, float)) or not 0 <= self.b <= 1:
            raise ValueError("bm25.b 必须位于 [0, 1]")
        allowed_fields = {"title", "abstract", "keywords"}
        unknown = set(self.fields) - allowed_fields
        if unknown:
            raise ValueError(f"BM25 不支持字段: {', '.join(sorted(unknown))}")
        for name, weight in self.fields.items():
            if (
                isinstance(weight, bool)
                or not isinstance(weight, (int, float))
                or not math.isfinite(weight)
                or weight < 0
            ):
                raise ValueError(f"字段权重 {name} 必须是非负有限数")


class BM25Retriever(Retriever):
    """索引 title、abstract、keywords 并返回稳定排序的轻量结果。"""

    def __init__(
        self,
        corpus: PaperCorpus,
        config: BM25Config | Mapping[str, Any] | None = None,
    ) -> None:
        """构建 BM25Okapi 索引；缺依赖时不自动安装。"""

        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise BM25DependencyError(
                "缺少 rank_bm25；请在已有环境中提供该依赖，本项目不会自动安装"
            ) from exc
        if not isinstance(corpus, PaperCorpus):
            raise TypeError("corpus 必须是 PaperCorpus")
        if len(corpus) == 0:
            raise ValueError("不能为 BM25 建立空语料索引")
        if config is None:
            self.config = BM25Config()
        elif isinstance(config, BM25Config):
            self.config = config
        else:
            self.config = BM25Config.from_mapping(config)
        self.config.validate()
        self.corpus = corpus
        self._papers = list(corpus)
        tokenized = [self._document_tokens(paper) for paper in self._papers]
        if not any(tokenized):
            raise ValueError("语料不包含任何可索引 token")
        self._index = BM25Okapi(tokenized, k1=self.config.k1, b=self.config.b)

    def _repeat(self, tokens: list[str], weight: float) -> list[str]:
        """按固定粒度重复 token，以确定性方式近似字段权重。"""

        repetitions = max(0, round(weight * self.config.weight_scale))
        return tokens * repetitions

    def _document_tokens(self, paper: PaperDocument) -> list[str]:
        """按配置权重组合标题、摘要和关键词 token。"""

        values = retrieval_text_fields(paper)
        tokens: list[str] = []
        for field_name, weight in self.config.fields.items():
            tokens.extend(self._repeat(tokenize(values[field_name]), weight))
        return tokens

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """返回 BM25 分数降序结果，分数相同时保持语料原始顺序。"""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query 必须是非空字符串")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k 必须是大于 0 的整数")
        query_tokens = tokenize(query)
        if not query_tokens:
            raise ValueError("query 不包含可检索 token")
        scores = self._index.get_scores(query_tokens)
        ordered = sorted(range(len(self._papers)), key=lambda index: (-scores[index], index))
        results: list[RetrievalResult] = []
        for rank, index in enumerate(ordered[: min(top_k, len(ordered))], start=1):
            paper = self._papers[index]
            results.append(
                RetrievalResult(
                    paper_id=paper.paper_id,
                    title=paper.title,
                    score=float(scores[index]),
                    rank=rank,
                    source=RetrievalSource.BM25,
                    metadata={"query": query},
                )
            )
        return results
