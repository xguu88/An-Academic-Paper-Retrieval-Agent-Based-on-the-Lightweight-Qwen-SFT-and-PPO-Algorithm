"""用 HashingTextEmbedder 完全离线验证 Dense、RRF 与 Hybrid 工程链路。"""

from pathlib import Path
import sys

import numpy as np


# 直接运行脚本时使用项目 fixture，绝不访问模型服务或网络。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluator import evaluate_retriever  # noqa: E402
from src.evaluation.tasks import load_tasks_jsonl  # noqa: E402
from src.retrieval.bm25_retriever import BM25Retriever  # noqa: E402
from src.retrieval.corpus import PaperCorpus  # noqa: E402
from src.retrieval.dense_retriever import DenseRetriever, cosine_similarities  # noqa: E402
from src.retrieval.fingerprint import corpus_fingerprint  # noqa: E402
from src.retrieval.hybrid_retriever import HybridConfig, HybridRetriever  # noqa: E402
from src.retrieval.rrf import reciprocal_rank_fusion  # noqa: E402
from src.retrieval.testing import HashingTextEmbedder  # noqa: E402


FIXTURES = PROJECT_ROOT / "tests" / "fixtures"


def main() -> None:
    """逐项检查两路检索、融合、索引一致性和统一 Evaluator。"""

    print("HYBRID RETRIEVAL SMOKE TEST")
    print()
    corpus = PaperCorpus.from_jsonl(FIXTURES / "papers.jsonl")
    tasks = load_tasks_jsonl(FIXTURES / "tasks.jsonl")
    bm25 = BM25Retriever(corpus)
    assert bm25.retrieve("BM25 lexical ranking", 1)[0].paper_id == "p-bm25"
    print("[PASS] BM25")

    embedder = HashingTextEmbedder(dimension=256)
    first = embedder.encode(["same text"])
    assert np.array_equal(first, embedder.encode(["same text"]))
    print("[PASS] Hash embedding")

    dense = DenseRetriever(corpus, embedder)
    assert dense.document_embeddings.shape == (len(corpus), embedder.dimension)
    print("[PASS] Dense index")
    dense_results = dense.retrieve("BM25 lexical document ranking", 3)
    assert dense_results[0].paper_id == "p-bm25"
    print("[PASS] Dense retrieval")
    zero_scores = cosine_similarities(np.zeros(2), np.zeros((2, 2)))
    assert np.array_equal(zero_scores, np.zeros(2))
    print("[PASS] Cosine similarity")

    fused = reciprocal_rank_fusion(
        {"bm25": bm25.retrieve("retrieval", 5), "dense": dense.retrieve("retrieval", 5)},
        top_k=5,
    )
    assert fused and len({result.paper_id for result in fused}) == len(fused)
    print("[PASS] RRF")

    hybrid = HybridRetriever(
        bm25,
        dense,
        HybridConfig(bm25_top_n=8, dense_top_n=8, final_top_k=5),
    )
    hybrid_results = hybrid.retrieve("retrieval", 5)
    assert hybrid_results
    print("[PASS] Hybrid retrieval")
    assert all(
        {"bm25_rank", "bm25_score", "dense_rank", "dense_score", "rrf_score"}
        <= set(result.metadata)
        for result in hybrid_results
    )
    print("[PASS] Metadata provenance")

    index_dir = PROJECT_ROOT / "results" / "smoke_dense_index"
    index_files = [index_dir / name for name in ("embeddings.npy", "paper_ids.json", "metadata.json")]
    for path in index_files:
        path.unlink(missing_ok=True)
    try:
        dense.save_index(index_dir)
        loaded = DenseRetriever.from_index(corpus, embedder, index_dir)
        assert np.array_equal(loaded.document_embeddings, dense.document_embeddings)
    finally:
        for path in index_files:
            path.unlink(missing_ok=True)
    print("[PASS] Index save/load")
    assert corpus_fingerprint(corpus) == dense.fingerprint
    print("[PASS] Corpus fingerprint")

    assert evaluate_retriever(dense, tasks, 10).num_tasks == len(tasks)
    assert evaluate_retriever(hybrid, tasks, 10).num_tasks == len(tasks)
    print("[PASS] Evaluator compatibility")
    print()
    print("HYBRID RETRIEVAL SMOKE TEST PASS")


if __name__ == "__main__":
    main()
