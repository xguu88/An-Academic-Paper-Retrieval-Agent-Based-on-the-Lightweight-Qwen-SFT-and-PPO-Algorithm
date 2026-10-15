"""使用本地 CrossEncoder 完全离线验证真实重排链路。"""

from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluator import evaluate_retriever  # noqa: E402
from src.evaluation.tasks import load_tasks_jsonl  # noqa: E402
from src.retrieval.bm25_retriever import BM25Retriever  # noqa: E402
from src.retrieval.corpus import PaperCorpus  # noqa: E402
from src.retrieval.cross_encoder_reranker import CrossEncoderReranker  # noqa: E402
from src.retrieval.reranked_retriever import (  # noqa: E402
    RerankedRetriever,
    RerankedRetrieverConfig,
)
from src.retrieval.sentence_transformer_cross_encoder import (  # noqa: E402
    SentenceTransformerCrossEncoderScorer,
)


FIXTURES = PROJECT_ROOT / "tests" / "fixtures"
LOCAL_MODEL = Path("C:/Users/swukw/Desktop/agent_project/models/ms-marco-MiniLM-L6-v2")


def main() -> None:
    """检查本地加载、成对打分、重排集成、metadata 与统一 Evaluator。"""

    print("REAL RERANKER SMOKE TEST")
    print()
    assert LOCAL_MODEL.is_dir()
    print("[PASS] Local model found")

    scorer = SentenceTransformerCrossEncoderScorer(
        LOCAL_MODEL,
        device="cpu",
        batch_size=16,
        allow_remote=False,
    )
    print("[PASS] CrossEncoder loaded")

    scores = scorer.score_pairs(
        [
            ("dense retrieval", "Dense retrieval uses vector representations."),
            ("dense retrieval", "Photosynthesis converts light energy."),
        ]
    )
    assert scores.shape == (2,)
    print("[PASS] Pair scoring")
    assert np.all(np.isfinite(scores))
    print("[PASS] Scores finite")

    corpus = PaperCorpus.from_jsonl(FIXTURES / "papers.jsonl")
    tasks = load_tasks_jsonl(FIXTURES / "tasks.jsonl")
    retriever = RerankedRetriever(
        BM25Retriever(corpus),
        CrossEncoderReranker(corpus, scorer),
        RerankedRetrieverConfig(candidate_top_n=10, final_top_k=10),
    )
    candidates, reranked = retriever.retrieve_with_candidates("academic retrieval")
    assert reranked and {item.paper_id for item in candidates} == {
        item.paper_id for item in reranked
    }
    print("[PASS] Reranker integration")

    required = {
        "pre_rerank_rank",
        "reranker_score",
        "reranked_rank",
    }
    assert all(required <= set(item.metadata) for item in reranked)
    print("[PASS] Metadata")

    assert evaluate_retriever(retriever, tasks, top_k=10).num_tasks == len(tasks)
    print("[PASS] Evaluator compatibility")
    print()
    print("REAL RERANKER SMOKE TEST PASS")


if __name__ == "__main__":
    main()

