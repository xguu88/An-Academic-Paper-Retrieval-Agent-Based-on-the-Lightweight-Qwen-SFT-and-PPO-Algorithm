"""在已转换 benchmark split 上运行单个现有 Retriever 与统一 Evaluator。"""

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
from src.evaluation.evaluator import evaluate_retriever  # noqa: E402
from src.evaluation.tasks import load_tasks_jsonl  # noqa: E402
from src.retrieval.corpus import PaperCorpus  # noqa: E402


def parse_args() -> argparse.Namespace:
    """解析 processed benchmark、split、Retriever 和输出根目录。"""

    parser = argparse.ArgumentParser(description="评估一个已转换真实学术检索 benchmark")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "dev", "test"), default="test")
    parser.add_argument("--retriever", choices=("bm25", "dense", "hybrid"), required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--config", type=Path, default=Path("configs/retrieval.yaml"))
    parser.add_argument("--results-root", type=Path, default=Path("results/benchmarks"))
    return parser.parse_args()


def _dataset_slug(name: str) -> str:
    """把 manifest 数据集名称转换成安全结果目录名。"""

    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-.")
    if not slug:
        raise ValueError("dataset_name 无法转换为安全目录名")
    return slug


def main() -> None:
    """构建 Retriever，分离测量 query latency，并保存指标与构建耗时。"""

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
        requested={args.retriever},
    )
    retriever = retrievers[args.retriever]
    metrics = evaluate_retriever(retriever, tasks, args.top_k)
    latency, rankings = measure_query_latency(retriever, tasks, args.top_k)
    report = {
        "dataset": manifest["dataset_name"],
        "split": args.split,
        "documents": len(corpus),
        "queries": len(tasks),
        "retriever": args.retriever,
        "metrics": metrics.to_dict(),
        "query_latency": latency.to_dict(),
        "build_times": build_times.to_dict(),
        "rankings": rankings,
    }
    output_dir = args.results_root / _dataset_slug(manifest["dataset_name"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.retriever}.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("REAL ACADEMIC RETRIEVAL BENCHMARK")
    print()
    print(f"Dataset: {manifest['dataset_name']}")
    print(f"Split: {args.split}")
    print(f"Documents: {len(corpus)}")
    print(f"Queries: {len(tasks)}")
    print(f"Retriever: {args.retriever}")
    print()
    labels = [
        ("Hit@1", metrics.hit_at_1),
        ("Hit@5", metrics.hit_at_5),
        ("Hit@10", metrics.hit_at_10),
        ("Recall@5", metrics.recall_at_5),
        ("Recall@10", metrics.recall_at_10),
        ("Recall@20", metrics.recall_at_20),
        ("MRR", metrics.mrr),
        ("NDCG@10", metrics.ndcg_at_10),
    ]
    for label, value in labels:
        print(f"{label}: {value:.6f}")
    print()
    print(
        f"Query latency ms: mean={latency.mean_ms:.3f}, "
        f"p50={latency.p50_ms:.3f}, p95={latency.p95_ms:.3f}"
    )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()

