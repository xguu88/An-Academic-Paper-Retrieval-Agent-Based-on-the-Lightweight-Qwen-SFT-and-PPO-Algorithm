"""使用可注入 PairScorer 实现 title+abstract Cross-Encoder 重排。"""

from collections.abc import Sequence

import numpy as np

from src.retrieval.corpus import PaperCorpus
from src.retrieval.pair_scorer import PairScorer
from src.retrieval.reranker import Reranker
from src.retrieval.types import RetrievalResult, RetrievalSource


def build_reranker_document_text(title: str, abstract: str | None) -> str:
    """只拼接标题与摘要，绝不加入引用关系、全文或 benchmark 标签。"""

    return "\n".join(part for part in (title, abstract or "") if part)


class CrossEncoderReranker(Reranker):
    """把语料正文解析与成对模型打分组合为稳定重排流程。"""

    def __init__(self, corpus: PaperCorpus, scorer: PairScorer) -> None:
        """注入论文语料和模型适配器，避免直接绑定 sentence-transformers。"""

        if not isinstance(corpus, PaperCorpus):
            raise TypeError("corpus 必须是 PaperCorpus")
        if not isinstance(scorer, PairScorer):
            raise TypeError("scorer 必须实现 PairScorer")
        self.corpus = corpus
        self.scorer = scorer

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """对候选逐对打分；同分时稳定保留候选原始顺序。"""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query 必须是非空字符串")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k 必须是大于 0 的整数")
        if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
            raise TypeError("candidates 必须是 RetrievalResult 序列")
        seen: set[str] = set()
        pairs: list[tuple[str, str]] = []
        for candidate in candidates:
            if not isinstance(candidate, RetrievalResult):
                raise TypeError("candidates 中每项必须是 RetrievalResult")
            if candidate.paper_id in seen:
                raise ValueError(f"候选包含重复 paper_id: {candidate.paper_id}")
            seen.add(candidate.paper_id)
            paper = self.corpus.require(candidate.paper_id)
            pairs.append((query, build_reranker_document_text(paper.title, paper.abstract)))

        scores = np.asarray(self.scorer.score_pairs(pairs), dtype=np.float64)
        if scores.shape != (len(candidates),):
            raise ValueError(
                f"PairScorer 必须返回 shape ({len(candidates)},)，实际为 {scores.shape}"
            )
        if not np.all(np.isfinite(scores)):
            raise ValueError("PairScorer 返回了非有限分数")
        order = sorted(range(len(candidates)), key=lambda index: (-scores[index], index))
        reranked: list[RetrievalResult] = []
        for new_rank, index in enumerate(order[:top_k], start=1):
            candidate = candidates[index]
            metadata = dict(candidate.metadata)
            metadata.update(
                {
                    "pre_rerank_rank": candidate.rank if candidate.rank is not None else index + 1,
                    "reranker_score": float(scores[index]),
                    "reranked_rank": new_rank,
                }
            )
            reranked.append(
                RetrievalResult(
                    paper_id=candidate.paper_id,
                    title=candidate.title,
                    score=float(scores[index]),
                    rank=new_rank,
                    source=RetrievalSource.RERANKER,
                    metadata=metadata,
                )
            )
        return reranked

