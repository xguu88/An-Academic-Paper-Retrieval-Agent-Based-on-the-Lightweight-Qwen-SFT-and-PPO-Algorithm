"""统一比较已转换 benchmark 的 BM25、真实 Dense 与 RRF Hybrid。"""

import argparse
import json
from pathlib import Path
import re
import sys

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarks.runtime import build_retrievers, measure_query_latency  # noqa: E402
from src.evaluation.error_analysis import analyze_retrieval_errors  # noqa: E402
from src.evaluation.evaluator import evaluate_retriever  # noqa: E402
from src.evaluation.tasks import load_tasks_jsonl  # noqa: E402
from src.retrieval.corpus import PaperCorpus  # noqa: E402


def parse_args() -> argparse.Namespace:
    """解析 processed benchmark、split、top-k 与输出位置。"""

    parser = argparse.ArgumentParser(description="比较 benchmark 上的三种 Retriever")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "dev", "test"), default="test")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--config", type=Path, default=Path("configs/retrieval.yaml"))
    parser.add_argument("--results-root", type=Path, default=Path("results/benchmarks"))
    return parser.parse_args()


def _slug(name: str) -> str:
    """生成安全数据集结果目录名。"""

    value = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-.")
    if not value:
        raise ValueError("dataset_name 非法")
    return value


def main() -> None:
    """构建三路 Retriever、复用 Evaluator、测 latency 并保存 error analysis。"""

    args = parse_args()
    manifest = json.loads((args.benchmark / "benchmark_manifest.json").read_text(encoding="utf-8"))
    corpus = PaperCorpus.from_jsonl(args.benchmark / "papers.jsonl")
    tasks = load_tasks_jsonl(args.benchmark / f"{args.split}_tasks.jsonl")
    if not tasks:
        raise ValueError(f"benchmark split 为空: {args.split}")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    retrievers, build_times = build_retrievers(
        corpus,
        config,
        args.benchmark / "indexes" / "dense",
        requested={"bm25", "dense", "hybrid"},
    )
    metrics = {}
    latencies = {}
    rankings = {}
    for name in ("bm25", "dense", "hybrid"):
        metrics[name] = evaluate_retriever(retrievers[name], tasks, args.top_k).to_dict()
        latency, method_rankings = measure_query_latency(retrievers[name], tasks, args.top_k)
        latencies[name] = latency.to_dict()
        rankings[name] = method_rankings
    error_analysis = analyze_retrieval_errors(tasks, rankings, top_k=min(10, args.top_k))
    report = {
        "dataset": manifest["dataset_name"],
        "split": args.split,
        "documents": len(corpus),
        "queries": len(tasks),
        "metrics": metrics,
        "query_latency": latencies,
        "build_times": build_times.to_dict(),
    }
    output_dir = args.results_root / _slug(manifest["dataset_name"])
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / "comparison.json"
    error_path = output_dir / "error_analysis.json"
    comparison_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    error_path.write_text(
        json.dumps(error_analysis, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("REAL ACADEMIC RETRIEVAL BENCHMARK COMPARISON")
    print()
    print(f"Dataset: {manifest['dataset_name']}  Split: {args.split}")
    print("Metric         BM25       Dense      Hybrid")
    rows = [
        ("Hit@1", "hit_at_1"),
        ("Hit@5", "hit_at_5"),
        ("Hit@10", "hit_at_10"),
        ("Recall@5", "recall_at_5"),
        ("Recall@10", "recall_at_10"),
        ("Recall@20", "recall_at_20"),
        ("MRR", "mrr"),
        ("NDCG@10", "ndcg_at_10"),
    ]
    for label, key in rows:
        values = [metrics[name][key] for name in ("bm25", "dense", "hybrid")]
        print(f"{label:<14}{values[0]:<11.6f}{values[1]:<11.6f}{values[2]:.6f}")
    print()
    for name in ("bm25", "dense", "hybrid"):
        latency = latencies[name]
        print(
            f"{name} latency ms: mean={latency['mean_ms']:.3f}, "
            f"p50={latency['p50_ms']:.3f}, p95={latency['p95_ms']:.3f}"
        )
    print(f"Comparison saved: {comparison_path}")
    print(f"Error analysis saved: {error_path}")


if __name__ == "__main__":
    main()

