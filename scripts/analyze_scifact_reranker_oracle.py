"""离线计算 SciFact Hybrid 候选召回与重排 Oracle 上界。"""

import argparse
import json
from pathlib import Path
import sys

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarks.runtime import build_retrievers  # noqa: E402
from src.evaluation.reranker_analysis import evaluate_candidate_pool  # noqa: E402
from src.evaluation.tasks import load_tasks_jsonl  # noqa: E402
from src.retrieval.corpus import PaperCorpus  # noqa: E402


def parse_args() -> argparse.Namespace:
    """解析已转换 SciFact、配置、旧错误报告与输出位置。"""

    parser = argparse.ArgumentParser(description="分析 SciFact 重排候选池与 Oracle 上界")
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("data/benchmarks/scifact/processed"),
    )
    parser.add_argument("--config", type=Path, default=Path("configs/retrieval.yaml"))
    parser.add_argument(
        "--error-analysis",
        type=Path,
        default=Path("results/benchmarks/scifact/error_analysis.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/benchmarks/scifact/reranker_oracle_analysis.json"),
    )
    return parser.parse_args()


def main() -> None:
    """复用现有 Hybrid 召回，并分别分析 Top-20 与 Top-50 候选池。"""

    args = parse_args()
    corpus = PaperCorpus.from_jsonl(args.benchmark / "papers.jsonl")
    tasks = load_tasks_jsonl(args.benchmark / "test_tasks.jsonl")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    # Oracle 需要一次取足 Top-50；不改变已有 comparison/error_analysis 结果。
    config["rrf"] = dict(config["rrf"])
    config["rrf"]["final_top_k"] = max(50, int(config["rrf"].get("final_top_k", 20)))
    retrievers, build_times = build_retrievers(
        corpus,
        config,
        args.benchmark / "indexes" / "dense",
        requested={"hybrid"},
    )
    hybrid = retrievers["hybrid"]
    rankings = {
        task.task_id: [
            result.paper_id for result in hybrid.retrieve(task.question, top_k=50)
        ]
        for task in tasks
    }
    pools = {
        str(depth): evaluate_candidate_pool(tasks, rankings, depth).to_dict()
        for depth in (20, 50)
    }
    existing_errors = json.loads(args.error_analysis.read_text(encoding="utf-8"))
    report = {
        "dataset": "scifact",
        "split": "test",
        "documents": len(corpus),
        "queries": len(tasks),
        "candidate_source": "hybrid_bm25_bge_rrf",
        "candidate_pools": pools,
        "existing_error_analysis": {
            **existing_errors["counts"],
            "mixed_outcome": existing_errors.get("mixed_outcome", 0),
        },
        "build_times": build_times.to_dict(),
        "notes": [
            "Candidate Recall 是各 query 相关论文召回率的宏平均。",
            "Candidate Hit Rate 是候选池至少含一篇相关论文的 query 比例。",
            "Oracle 只在 Hybrid 候选池内把相关论文提前，不是模型结果。",
            "未加载或评估任何真实 CrossEncoder reranker。",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("SCIFACT RERANKER ORACLE ANALYSIS")
    print()
    for depth in (20, 50):
        metrics = pools[str(depth)]
        print(
            f"Candidate Top-{depth}: hit_rate={metrics['candidate_hit_rate']:.6f}, "
            f"recall={metrics['candidate_recall']:.6f}"
        )
        print(
            f"Oracle Top-{depth}: Hit@1={metrics['oracle_hit_at_1']:.6f}, "
            f"MRR={metrics['oracle_mrr']:.6f}, "
            f"NDCG@10={metrics['oracle_ndcg_at_10']:.6f}"
        )
    print()
    print("REAL RERANKER MODEL TEST SKIPPED")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

