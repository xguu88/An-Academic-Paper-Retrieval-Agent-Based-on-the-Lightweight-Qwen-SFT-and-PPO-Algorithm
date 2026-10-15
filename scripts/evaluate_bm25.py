"""运行完全离线的 BM25 fixture 或本地语料检索基线。"""

import argparse
import json
from pathlib import Path
import sys


# 允许从项目根目录直接执行脚本而无需安装本包。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluator import evaluate_retriever  # noqa: E402
from src.evaluation.tasks import load_tasks_jsonl  # noqa: E402
from src.retrieval.bm25_retriever import (  # noqa: E402
    BM25DependencyError,
    BM25Retriever,
)
from src.retrieval.corpus import PaperCorpus  # noqa: E402


def parse_args() -> argparse.Namespace:
    """解析语料、任务、检索深度和报告路径。"""

    parser = argparse.ArgumentParser(description="评估离线 BM25 检索基线")
    parser.add_argument("--corpus", type=Path, required=True, help="论文 JSONL 路径")
    parser.add_argument("--tasks", type=Path, required=True, help="任务 JSONL 路径")
    parser.add_argument("--top-k", type=int, default=20, help="每个任务返回结果数")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/bm25_baseline.json"),
        help="基线 JSON 报告输出路径",
    )
    return parser.parse_args()


def main() -> int:
    """运行评估、打印固定指标并保存 JSON 报告。"""

    args = parse_args()
    print("BM25 RETRIEVAL BASELINE")
    print()
    corpus = PaperCorpus.from_jsonl(args.corpus)
    tasks = load_tasks_jsonl(args.tasks)
    try:
        retriever = BM25Retriever(corpus)
    except BM25DependencyError as exc:
        print(f"[FAIL] {exc}")
        return 2
    result = evaluate_retriever(retriever, tasks, args.top_k)
    report = {
        "corpus_papers": len(corpus),
        "top_k": args.top_k,
        **result.to_dict(),
    }
    print(f"Corpus papers: {len(corpus)}")
    print(f"Tasks: {result.num_tasks}")
    print()
    print(f"Hit@1: {result.hit_at_1:.6f}")
    print(f"Hit@5: {result.hit_at_5:.6f}")
    print(f"Hit@10: {result.hit_at_10:.6f}")
    print(f"Recall@5: {result.recall_at_5:.6f}")
    print(f"Recall@10: {result.recall_at_10:.6f}")
    print(f"Recall@20: {result.recall_at_20:.6f}")
    print(f"MRR: {result.mrr:.6f}")
    print(f"NDCG@10: {result.ndcg_at_10:.6f}")
    print(f"Average results: {result.average_num_results:.6f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print()
    print("BM25 RETRIEVAL BASELINE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

