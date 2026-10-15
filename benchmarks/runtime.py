"""为 benchmark CLI 构建现有 Retriever，并独立测量查询 latency。"""

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from src.evaluation.tasks import RetrievalTask
from src.retrieval.base import Retriever
from src.retrieval.bm25_retriever import BM25Config, BM25Retriever
from src.retrieval.corpus import PaperCorpus
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridConfig, HybridRetriever
from src.retrieval.sentence_transformer_embedder import SentenceTransformerEmbedder


@dataclass(slots=True)
class RetrieverBuildTimes:
    """分别记录模型加载与索引构建/加载耗时，不混入 query latency。"""

    bm25_index_build_ms: float | None = None
    embedding_model_load_ms: float | None = None
    dense_index_build_ms: float | None = None
    dense_index_load_ms: float | None = None
    dense_index_save_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """转换为 JSON 兼容字典。"""

        return asdict(self)


@dataclass(slots=True)
class LatencyReport:
    """记录只包围 retrieve(query) 的毫秒级延迟统计。"""

    mean_ms: float
    p50_ms: float
    p95_ms: float
    num_queries: int

    def to_dict(self) -> dict[str, Any]:
        """转换为 JSON 兼容字典。"""

        return asdict(self)


def build_retrievers(
    corpus: PaperCorpus,
    config: dict[str, Any],
    index_dir: Path,
    *,
    requested: set[str],
) -> tuple[dict[str, Retriever], RetrieverBuildTimes]:
    """按需构建 BM25/Dense/Hybrid，严格复用本地模型和 corpus fingerprint。"""

    allowed = {"bm25", "dense", "hybrid"}
    if not requested or not requested <= allowed:
        raise ValueError(f"requested 只能来自 {sorted(allowed)}")
    need_bm25 = bool(requested & {"bm25", "hybrid"})
    need_dense = bool(requested & {"dense", "hybrid"})
    retrievers: dict[str, Retriever] = {}
    times = RetrieverBuildTimes()

    bm25: BM25Retriever | None = None
    if need_bm25:
        start = perf_counter()
        bm25 = BM25Retriever(corpus, BM25Config.from_mapping(config["bm25"]))
        times.bm25_index_build_ms = (perf_counter() - start) * 1000
        if "bm25" in requested:
            retrievers["bm25"] = bm25

    dense: DenseRetriever | None = None
    if need_dense:
        dense_config = config["dense"]
        model_path = dense_config.get("model_name_or_path")
        if not model_path:
            raise ValueError("dense.model_name_or_path 未配置本地模型")
        start = perf_counter()
        embedder = SentenceTransformerEmbedder(
            model_path,
            device=dense_config.get("device", "cpu"),
            batch_size=dense_config.get("batch_size", 32),
            normalize_embeddings=dense_config.get("normalize_embeddings", True),
            allow_remote=False,
        )
        times.embedding_model_load_ms = (perf_counter() - start) * 1000
        index_files = [index_dir / name for name in ("embeddings.npy", "paper_ids.json", "metadata.json")]
        if all(path.is_file() for path in index_files):
            start = perf_counter()
            dense = DenseRetriever.from_index(corpus, embedder, index_dir)
            times.dense_index_load_ms = (perf_counter() - start) * 1000
        elif any(path.exists() for path in index_files):
            raise RuntimeError(f"Dense index 不完整: {index_dir}")
        else:
            start = perf_counter()
            dense = DenseRetriever(corpus, embedder)
            times.dense_index_build_ms = (perf_counter() - start) * 1000
            start = perf_counter()
            dense.save_index(index_dir)
            times.dense_index_save_ms = (perf_counter() - start) * 1000
        if "dense" in requested:
            retrievers["dense"] = dense

    if "hybrid" in requested:
        assert bm25 is not None and dense is not None
        rrf_config = config["rrf"]
        retrievers["hybrid"] = HybridRetriever(
            bm25,
            dense,
            HybridConfig(
                bm25_top_n=rrf_config.get("bm25_top_n", 50),
                dense_top_n=rrf_config.get("dense_top_n", 50),
                rrf_k=rrf_config.get("k", 60),
                final_top_k=rrf_config.get("final_top_k", 20),
            ),
        )
    return retrievers, times


def measure_query_latency(
    retriever: Retriever,
    tasks: list[RetrievalTask],
    top_k: int,
) -> tuple[LatencyReport, dict[str, list[str]]]:
    """只计时 retrieve 调用，并返回每个任务的排名 ID。"""

    durations: list[float] = []
    rankings: dict[str, list[str]] = {}
    for task in tasks:
        start = perf_counter()
        results = retriever.retrieve(task.question, top_k=top_k)
        durations.append((perf_counter() - start) * 1000)
        rankings[task.task_id] = [result.paper_id for result in results]
    values = np.asarray(durations, dtype=np.float64)
    return (
        LatencyReport(
            mean_ms=float(values.mean()),
            p50_ms=float(np.percentile(values, 50)),
            p95_ms=float(np.percentile(values, 95)),
            num_queries=len(durations),
        ),
        rankings,
    )

