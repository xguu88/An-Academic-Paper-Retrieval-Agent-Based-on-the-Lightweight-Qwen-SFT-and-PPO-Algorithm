"""实现基于 numpy 文档向量和余弦相似度的 Dense Retriever。"""

from pathlib import Path

import numpy as np

from src.retrieval.base import Retriever
from src.retrieval.corpus import PaperCorpus
from src.retrieval.dense_index import (
    DenseIndexValidationError,
    DenseIndexMetadata,
    load_dense_index,
    save_dense_index,
)
from src.retrieval.document_text import build_retrieval_text
from src.retrieval.embedding import TextEmbedder
from src.retrieval.fingerprint import corpus_fingerprint
from src.retrieval.types import RetrievalResult, RetrievalSource


def cosine_similarities(
    query_vector: np.ndarray,
    document_vectors: np.ndarray,
    *,
    already_normalized: bool = False,
) -> np.ndarray:
    """计算 query 对所有文档的余弦相似度，零向量安全返回零分。"""

    query = np.asarray(query_vector, dtype=np.float32)
    documents = np.asarray(document_vectors, dtype=np.float32)
    if query.ndim != 1 or documents.ndim != 2 or documents.shape[1] != query.shape[0]:
        raise ValueError("query/document embedding shape 不兼容")
    if not np.all(np.isfinite(query)) or not np.all(np.isfinite(documents)):
        raise ValueError("embedding 包含 NaN 或 Infinity")
    dots = documents @ query
    if already_normalized:
        return np.asarray(dots, dtype=np.float32)
    denominators = np.linalg.norm(documents, axis=1) * np.linalg.norm(query)
    scores = np.zeros(documents.shape[0], dtype=np.float32)
    np.divide(dots, denominators, out=scores, where=denominators > 0)
    return scores


class DenseRetriever(Retriever):
    """只依赖 TextEmbedder，通过稳定余弦排序返回统一检索结果。"""

    def __init__(
        self,
        corpus: PaperCorpus,
        embedder: TextEmbedder,
        *,
        document_embeddings: np.ndarray | None = None,
        paper_ids: list[str] | None = None,
    ) -> None:
        """编码全部论文，或使用经过调用方加载校验的已有索引向量。"""

        if not isinstance(corpus, PaperCorpus):
            raise TypeError("corpus 必须是 PaperCorpus")
        if not isinstance(embedder, TextEmbedder):
            raise TypeError("embedder 必须实现 TextEmbedder")
        if len(corpus) == 0:
            raise ValueError("不能为 DenseRetriever 使用空语料")
        self.corpus = corpus
        self.embedder = embedder
        self._papers = list(corpus)
        corpus_ids = [paper.paper_id for paper in self._papers]
        if len(set(corpus_ids)) != len(corpus_ids):
            raise ValueError("corpus 包含重复 paper_id")

        if document_embeddings is None:
            texts = [build_retrieval_text(paper) for paper in self._papers]
            embeddings = embedder.encode(texts)
        else:
            embeddings = np.asarray(document_embeddings)
            if paper_ids is None or paper_ids != corpus_ids:
                raise ValueError("索引 paper_ids 与 corpus 顺序不一致")
        expected_shape = (len(self._papers), embedder.dimension)
        if embeddings.shape != expected_shape:
            raise ValueError(
                f"document embedding shape 非法: expected={expected_shape}, actual={embeddings.shape}"
            )
        if not np.all(np.isfinite(embeddings)):
            raise ValueError("document embeddings 包含 NaN 或 Infinity")
        self.document_embeddings = np.asarray(embeddings, dtype=np.float32)
        self.paper_ids = corpus_ids
        self.fingerprint = corpus_fingerprint(self._papers)

    @classmethod
    def from_index(
        cls,
        corpus: PaperCorpus,
        embedder: TextEmbedder,
        index_dir: str | Path,
    ) -> "DenseRetriever":
        """严格校验并加载与当前语料和编码器一致的离线索引。"""

        fingerprint = corpus_fingerprint(corpus)
        bundle = load_dense_index(
            index_dir,
            expected_corpus_fingerprint=fingerprint,
            expected_embedding_model=embedder.model_identifier,
            expected_dimension=embedder.dimension,
        )
        if bundle.metadata.normalize_embeddings != embedder.normalize_embeddings:
            raise DenseIndexValidationError("索引 normalize_embeddings 与 embedder 不匹配")
        return cls(
            corpus,
            embedder,
            document_embeddings=bundle.embeddings,
            paper_ids=bundle.paper_ids,
        )

    def save_index(self, index_dir: str | Path) -> None:
        """保存当前文档向量、论文顺序及一致性元数据。"""

        metadata = DenseIndexMetadata(
            embedding_model=self.embedder.model_identifier,
            dimension=self.embedder.dimension,
            normalize_embeddings=self.embedder.normalize_embeddings,
            num_documents=len(self.paper_ids),
            corpus_fingerprint=self.fingerprint,
        )
        save_dense_index(index_dir, self.document_embeddings, self.paper_ids, metadata)

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """编码 query、计算余弦相似度并稳定返回 Top-K。"""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query 必须是非空字符串")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k 必须是大于 0 的整数")
        query_embeddings = np.asarray(self.embedder.encode([query]), dtype=np.float32)
        expected = (1, self.embedder.dimension)
        if query_embeddings.shape != expected:
            raise ValueError(
                f"query embedding shape 非法: expected={expected}, actual={query_embeddings.shape}"
            )
        scores = cosine_similarities(
            query_embeddings[0],
            self.document_embeddings,
            already_normalized=self.embedder.normalize_embeddings,
        )
        ordered = sorted(range(len(self._papers)), key=lambda index: (-scores[index], index))
        return [
            RetrievalResult(
                paper_id=self._papers[index].paper_id,
                title=self._papers[index].title,
                score=float(scores[index]),
                rank=rank,
                source=RetrievalSource.DENSE,
                metadata={"query": query},
            )
            for rank, index in enumerate(ordered[: min(top_k, len(ordered))], start=1)
        ]
