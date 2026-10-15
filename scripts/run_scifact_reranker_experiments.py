"""运行 SciFact Hybrid+CrossEncoder Top-20/Top-50 两组受控实验。"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarks.runtime import build_retrievers  # noqa: E402
from src.evaluation.evaluator import evaluate_retriever  # noqa: E402
from src.evaluation.reranker_analysis import evaluate_candidate_pool  # noqa: E402
from src.evaluation.tasks import RetrievalTask, load_tasks_jsonl  # noqa: E402
from src.retrieval.base import Retriever  # noqa: E402
from src.retrieval.corpus import PaperCorpus  # noqa: E402
from src.retrieval.cross_encoder_reranker import CrossEncoderReranker  # noqa: E402
from src.retrieval.reranked_retriever import (  # noqa: E402
    RerankedRetriever,
    RerankedRetrieverConfig,
)
from src.retrieval.sentence_transformer_cross_encoder import (  # noqa: E402
    SentenceTransformerCrossEncoderScorer,
)
from src.retrieval.types import RetrievalResult  # noqa: E402


BENCHMARK = PROJECT_ROOT / "data" / "benchmarks" / "scifact" / "processed"
RESULTS = PROJECT_ROOT / "results" / "benchmarks" / "scifact"
CONFIG = PROJECT_ROOT / "configs" / "retrieval.yaml"


@dataclass(slots=True)
class MovementStats:
    """汇总候选池内相关与非相关论文的排名移动。"""

    relevant_occurrences: int
    mean_relevant_rank_before: float | None
    mean_relevant_rank_after: float | None
    promoted_relevant_count: int
    demoted_relevant_count: int
    unchanged_relevant_count: int
    promoted_nonrelevant_count: int
    demoted_nonrelevant_count: int
    unchanged_nonrelevant_count: int


class PrecomputedRetriever(Retriever):
    """把真实重排结果交给统一 Evaluator，避免重复运行 CrossEncoder。"""

    def __init__(self, rankings: dict[str, list[RetrievalResult]]) -> None:
        self.rankings = rankings

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """按 query 返回已由真实系统生成的排名并执行安全截断。"""

        return self.rankings[query][:top_k]


def _latency_report(durations: list[float]) -> dict[str, float | int]:
    """从真实端到端 retrieve 耗时计算 mean/p50/p95。"""

    values = np.asarray(durations, dtype=np.float64)
    return {
        "mean_ms": float(values.mean()),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "num_queries": len(durations),
    }


def _evaluate_baseline(
    retriever: Retriever,
    tasks: list[RetrievalTask],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """让统一 Evaluator 执行一次真实检索，并在包装层记录 query latency。"""

    durations: list[float] = []

    class TimedRetriever(Retriever):
        """仅为当前 baseline 记录实际 retrieve 调用时长。"""

        def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
            start = perf_counter()
            results = retriever.retrieve(query, top_k)
            durations.append((perf_counter() - start) * 1000)
            return results

    metrics = evaluate_retriever(TimedRetriever(), tasks, top_k=20).to_dict()
    return metrics, _latency_report(durations)


def _movement(
    tasks: list[RetrievalTask],
    before: dict[str, list[RetrievalResult]],
    after: dict[str, list[RetrievalResult]],
) -> MovementStats:
    """按 query-paper occurrence 统计完整候选池内排名移动。"""

    relevant_before: list[int] = []
    relevant_after: list[int] = []
    promoted_relevant = demoted_relevant = unchanged_relevant = 0
    promoted_nonrelevant = demoted_nonrelevant = unchanged_nonrelevant = 0
    for task in tasks:
        old = {item.paper_id: int(item.rank or 0) for item in before[task.task_id]}
        new = {item.paper_id: int(item.rank or 0) for item in after[task.task_id]}
        relevant = set(task.relevant_paper_ids)
        for paper_id, old_rank in old.items():
            new_rank = new[paper_id]
            delta = new_rank - old_rank
            if paper_id in relevant:
                relevant_before.append(old_rank)
                relevant_after.append(new_rank)
                if delta < 0:
                    promoted_relevant += 1
                elif delta > 0:
                    demoted_relevant += 1
                else:
                    unchanged_relevant += 1
            elif delta < 0:
                promoted_nonrelevant += 1
            elif delta > 0:
                demoted_nonrelevant += 1
            else:
                unchanged_nonrelevant += 1
    return MovementStats(
        relevant_occurrences=len(relevant_before),
        mean_relevant_rank_before=(float(np.mean(relevant_before)) if relevant_before else None),
        mean_relevant_rank_after=(float(np.mean(relevant_after)) if relevant_after else None),
        promoted_relevant_count=promoted_relevant,
        demoted_relevant_count=demoted_relevant,
        unchanged_relevant_count=unchanged_relevant,
        promoted_nonrelevant_count=promoted_nonrelevant,
        demoted_nonrelevant_count=demoted_nonrelevant,
        unchanged_nonrelevant_count=unchanged_nonrelevant,
    )


def _run_experiment(
    depth: int,
    hybrid: Retriever,
    reranker: CrossEncoderReranker,
    tasks: list[RetrievalTask],
) -> dict[str, Any]:
    """运行一个候选深度，断言完整候选集合保持并汇总真实结果。"""

    composed = RerankedRetriever(
        hybrid,
        reranker,
        RerankedRetrieverConfig(candidate_top_n=depth, final_top_k=20),
    )
    before: dict[str, list[RetrievalResult]] = {}
    after: dict[str, list[RetrievalResult]] = {}
    final_by_query: dict[str, list[RetrievalResult]] = {}
    durations: list[float] = []
    preserved = 0
    for index, task in enumerate(tasks, start=1):
        start = perf_counter()
        candidates, reranked = composed.retrieve_with_candidates(task.question)
        durations.append((perf_counter() - start) * 1000)
        before[task.task_id] = candidates
        after[task.task_id] = reranked
        final_by_query[task.question] = reranked[:20]
        required_metadata = {
            "bm25_rank",
            "bm25_score",
            "dense_rank",
            "dense_score",
            "rrf_score",
            "pre_rerank_rank",
            "reranker_score",
            "reranked_rank",
        }
        if any(required_metadata - set(item.metadata) for item in reranked):
            raise RuntimeError(f"reranker metadata 不完整: task_id={task.task_id}")
        if {item.paper_id for item in candidates} != {item.paper_id for item in reranked}:
            raise RuntimeError(f"candidate set mismatch: task_id={task.task_id}")
        preserved += 1
        if index % 25 == 0 or index == len(tasks):
            print(f"Top-{depth} progress: {index}/{len(tasks)}", flush=True)

    metrics = evaluate_retriever(PrecomputedRetriever(final_by_query), tasks, top_k=20).to_dict()
    candidate_rankings = {
        task_id: [item.paper_id for item in results] for task_id, results in before.items()
    }
    oracle = evaluate_candidate_pool(tasks, candidate_rankings, depth).to_dict()
    gaps = {
        "hit_at_1": {
            "actual": metrics["hit_at_1"],
            "oracle": oracle["oracle_hit_at_1"],
            "oracle_gap": oracle["oracle_hit_at_1"] - metrics["hit_at_1"],
        },
        "mrr": {
            "actual": metrics["mrr"],
            "oracle": oracle["oracle_mrr"],
            "oracle_gap": oracle["oracle_mrr"] - metrics["mrr"],
        },
        "ndcg_at_10": {
            "actual": metrics["ndcg_at_10"],
            "oracle": oracle["oracle_ndcg_at_10"],
            "oracle_gap": oracle["oracle_ndcg_at_10"] - metrics["ndcg_at_10"],
        },
    }
    return {
        "candidate_top_n": depth,
        "final_top_k": 20,
        "metrics": metrics,
        "query_latency": _latency_report(durations),
        "candidate_pool": oracle,
        "candidate_set_preservation": {
            "checked_queries": len(tasks),
            "preserved_queries": preserved,
            "mismatch_count": len(tasks) - preserved,
            "preserved": preserved == len(tasks),
        },
        "rank_movement": asdict(_movement(tasks, before, after)),
        "oracle_gap": gaps,
    }


def main() -> None:
    """固定所有参数，仅改变 candidate_top_n，保存三份可复现实验报告。"""

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    reranker_config = config["reranker"]
    model_path = Path(reranker_config["model_name_or_path"])
    if not model_path.is_dir():
        raise FileNotFoundError(f"本地 CrossEncoder 模型目录不存在: {model_path}")
    if reranker_config.get("allow_remote") is not False:
        raise ValueError("reranker.allow_remote 必须显式为 false")
    if reranker_config.get("device") != "cpu":
        raise ValueError("本阶段 reranker.device 必须为 cpu")

    corpus = PaperCorpus.from_jsonl(BENCHMARK / "papers.jsonl")
    tasks = load_tasks_jsonl(BENCHMARK / "test_tasks.jsonl")
    retrievers, build_times = build_retrievers(
        corpus,
        config,
        BENCHMARK / "indexes" / "dense",
        requested={"bm25", "dense", "hybrid"},
    )
    scorer = SentenceTransformerCrossEncoderScorer(
        model_path,
        device="cpu",
        batch_size=int(reranker_config["batch_size"]),
        allow_remote=False,
    )
    reranker = CrossEncoderReranker(corpus, scorer)

    baseline_metrics: dict[str, Any] = {}
    baseline_latency: dict[str, Any] = {}
    for name in ("bm25", "dense", "hybrid"):
        baseline_metrics[name], baseline_latency[name] = _evaluate_baseline(
            retrievers[name], tasks
        )
        print(f"Baseline complete: {name}", flush=True)

    experiments = {
        "top20": _run_experiment(20, retrievers["hybrid"], reranker, tasks),
        "top50": _run_experiment(50, retrievers["hybrid"], reranker, tasks),
    }
    common = {
        "dataset": "scifact",
        "split": "test",
        "documents": len(corpus),
        "queries": len(tasks),
        "model": {
            "name": "cross-encoder/ms-marco-MiniLM-L6-v2",
            "local_path": str(model_path.resolve()),
            "device": "cpu",
            "batch_size": int(reranker_config["batch_size"]),
            "local_files_only": True,
            "allow_remote": False,
        },
        "fixed_hybrid_rrf": dict(config["rrf"]),
        "build_times": build_times.to_dict(),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    for key, filename in (("top20", "reranker_top20.json"), ("top50", "reranker_top50.json")):
        payload = {**common, "experiment": experiments[key]}
        (RESULTS / filename).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    comparison = {
        **common,
        "baseline_metrics": baseline_metrics,
        "baseline_query_latency": baseline_latency,
        "experiments": experiments,
        "warning": "Oracle values are analytical upper bounds, not model metrics.",
    }
    output = RESULTS / "reranker_comparison.json"
    output.write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("SCIFACT REAL RERANKER EXPERIMENTS PASS")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
