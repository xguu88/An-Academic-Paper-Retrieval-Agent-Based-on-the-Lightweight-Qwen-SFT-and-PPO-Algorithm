"""使用人工 fixture 验证完整离线检索与评估链路。"""

from pathlib import Path
import sys


# 直接运行脚本时固定使用项目内 fixture，不访问网络。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluator import evaluate_retriever  # noqa: E402
from src.evaluation.metrics import hit_at_k, mrr, ndcg_at_k, recall_at_k  # noqa: E402
from src.evaluation.tasks import load_tasks_jsonl  # noqa: E402
from src.retrieval.bm25_retriever import (  # noqa: E402
    BM25DependencyError,
    BM25Retriever,
)
from src.retrieval.citation_graph import build_citation_graph  # noqa: E402
from src.retrieval.corpus import PaperCorpus  # noqa: E402
from src.retrieval.types import RetrievalResult, RetrievalSource  # noqa: E402


FIXTURES = PROJECT_ROOT / "tests" / "fixtures"


def main() -> int:
    """逐项验证语料、图、BM25、结果结构、指标和聚合评估。"""

    print("RETRIEVAL SMOKE TEST")
    print()
    corpus = PaperCorpus.from_jsonl(FIXTURES / "papers.jsonl")
    assert len(corpus) == 10
    print("[PASS] Corpus loaded")

    graph, stats = build_citation_graph(list(corpus))
    assert "p-attention" in graph.get_references("p-transformer")
    assert "p-transformer" in graph.get_citations("p-attention")
    assert stats.num_missing_targets == 1
    print("[PASS] Citation graph built")

    try:
        retriever = BM25Retriever(corpus)
    except BM25DependencyError as exc:
        print(f"[FAIL] {exc}")
        return 2
    print("[PASS] BM25 index built")

    results = retriever.retrieve("BM25 lexical keyword ranking", top_k=3)
    assert results and results[0].paper_id == "p-bm25"
    print("[PASS] Query retrieval")
    assert isinstance(results[0], RetrievalResult)
    assert results[0].source is RetrievalSource.BM25
    assert results[0].rank == 1
    print("[PASS] RetrievalResult schema")

    ids = [result.paper_id for result in results]
    relevant = {"p-bm25"}
    assert hit_at_k(ids, relevant, 1) == 1.0
    print("[PASS] Hit@K")
    assert recall_at_k(ids, relevant, 3) == 1.0
    print("[PASS] Recall@K")
    assert mrr(ids, relevant) == 1.0
    print("[PASS] MRR")
    assert ndcg_at_k(ids, relevant, 3) == 1.0
    print("[PASS] NDCG@K")

    tasks = load_tasks_jsonl(FIXTURES / "tasks.jsonl")
    evaluation = evaluate_retriever(retriever, tasks, top_k=10)
    assert evaluation.num_tasks == 6
    print("[PASS] Evaluation pipeline")
    print()
    print("RETRIEVAL SMOKE TEST PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

