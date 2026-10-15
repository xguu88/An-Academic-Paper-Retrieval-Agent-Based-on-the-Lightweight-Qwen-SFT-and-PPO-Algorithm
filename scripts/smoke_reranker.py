"""使用确定性人工 PairScorer 完全离线验证重排工程链路。"""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluator import evaluate_retriever  # noqa: E402
from src.evaluation.reranker_analysis import evaluate_candidate_pool  # noqa: E402
from src.evaluation.tasks import load_tasks_jsonl  # noqa: E402
from src.retrieval.bm25_retriever import BM25Retriever  # noqa: E402
from src.retrieval.corpus import PaperCorpus  # noqa: E402
from src.retrieval.cross_encoder_reranker import (  # noqa: E402
    CrossEncoderReranker,
    build_reranker_document_text,
)
from src.retrieval.reranked_retriever import (  # noqa: E402
    RerankedRetriever,
    RerankedRetrieverConfig,
)
from src.retrieval.testing import DeterministicPairScorer  # noqa: E402


FIXTURES = PROJECT_ROOT / "tests" / "fixtures"


def main() -> None:
    """逐项检查候选、打分、稳定排序、metadata、指标和 Evaluator。"""

    print("RERANKER SMOKE TEST")
    print()
    corpus = PaperCorpus.from_jsonl(FIXTURES / "papers.jsonl")
    tasks = load_tasks_jsonl(FIXTURES / "tasks.jsonl")
    base = BM25Retriever(corpus)
    query = "academic retrieval"
    candidates = base.retrieve(query, top_k=8)
    assert candidates
    print("[PASS] Candidate retrieval")

    best = corpus.require(candidates[-1].paper_id)
    pair = (query, build_reranker_document_text(best.title, best.abstract))
    scorer = DeterministicPairScorer({pair: 10.0})
    assert scorer.score_pairs([pair]).tolist() == [10.0]
    print("[PASS] Pair scoring")

    reranker = CrossEncoderReranker(corpus, scorer)
    reranked = reranker.rerank(query, candidates, top_k=5)
    assert reranked[0].paper_id == best.paper_id
    print("[PASS] Reranking")

    tied = CrossEncoderReranker(corpus, DeterministicPairScorer()).rerank(
        query, candidates, top_k=len(candidates)
    )
    assert [item.paper_id for item in tied] == [item.paper_id for item in candidates]
    print("[PASS] Stable tie")

    required = {"pre_rerank_rank", "reranker_score", "reranked_rank"}
    assert all(required <= set(item.metadata) for item in reranked)
    print("[PASS] Metadata")

    rankings = {
        task.task_id: [item.paper_id for item in base.retrieve(task.question, 10)]
        for task in tasks
    }
    assert evaluate_candidate_pool(tasks, rankings, 10).candidate_recall >= 0.0
    print("[PASS] Candidate recall")

    composed = RerankedRetriever(
        base,
        CrossEncoderReranker(corpus, DeterministicPairScorer()),
        RerankedRetrieverConfig(candidate_top_n=10, final_top_k=10),
    )
    assert evaluate_retriever(composed, tasks, 10).num_tasks == len(tasks)
    print("[PASS] Evaluator compatibility")
    print()
    print("RERANKER SMOKE TEST PASS")


if __name__ == "__main__":
    main()

