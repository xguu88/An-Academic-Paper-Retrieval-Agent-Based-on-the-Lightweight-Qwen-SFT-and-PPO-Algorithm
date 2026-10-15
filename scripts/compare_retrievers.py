"""仅在本地真实 embedding 模型可用时比较 BM25、Dense 与 Hybrid。"""

import argparse
import importlib.util
import json
from pathlib import Path
import sys


# 允许从项目根目录直接运行脚本，无需安装项目包。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.comparison import compare_retrievers  # noqa: E402
from src.evaluation.tasks import load_tasks_jsonl  # noqa: E402
from src.retrieval.bm25_retriever import BM25Retriever  # noqa: E402
from src.retrieval.corpus import PaperCorpus  # noqa: E402
from src.retrieval.dense_retriever import DenseRetriever  # noqa: E402
from src.retrieval.hybrid_retriever import HybridConfig, HybridRetriever  # noqa: E402
from src.retrieval.sentence_transformer_embedder import (  # noqa: E402
    EmbeddingDependencyError,
    EmbeddingModelUnavailableError,
    SentenceTransformerEmbedder,
)


FIXTURE_WARNING = (
    "This benchmark contains only 10 synthetic fixture papers and 6 synthetic "
    "retrieval tasks. The metrics validate the real Dense/Hybrid pipeline but do "
    "not represent real academic retrieval quality."
)


def parse_args() -> argparse.Namespace:
    """解析真实模型、本地索引、fixture/语料和输出路径。"""

    parser = argparse.ArgumentParser(description="比较 BM25、真实 Dense 与 RRF Hybrid")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--model-name-or-path", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--index-dir", type=Path, default=Path("data/indexes/dense"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/retrieval_comparison.json"),
    )
    return parser.parse_args()


def _print_table(report: dict[str, object]) -> None:
    """打印三路 Retriever 的固定指标对比表。"""

    metrics = report["metrics"]
    assert isinstance(metrics, dict)
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
    print("Metric         BM25       Dense      Hybrid")
    for label, key in rows:
        values = [metrics[name][key] for name in ("bm25", "dense", "hybrid")]
        print(f"{label:<14}{values[0]:<11.6f}{values[1]:<11.6f}{values[2]:.6f}")


def _print_dense_diagnostics(report: dict[str, object]) -> None:
    """逐任务打印真实 Dense Top-K、分数、排名和首个相关排名。"""

    per_task = report["per_task"]
    assert isinstance(per_task, list)
    print()
    print("DENSE PER-TASK DIAGNOSTICS")
    for item in per_task:
        print()
        print(f"Task: {item['task_id']}")
        print(f"Query: {item['question']}")
        print(f"Relevant: {item['relevant_paper_ids']}")
        print(f"First relevant rank: {item['dense']['first_relevant_rank']}")
        for result in item["dense"]["results"]:
            print(
                f"  rank={result['rank']:>2} score={result['score']:.6f} "
                f"paper_id={result['paper_id']}"
            )


def main() -> int:
    """检测真实本地模型边界，满足条件后运行并保存 comparison。"""

    print("RETRIEVAL COMPARISON")
    print()
    args = parse_args()
    if importlib.util.find_spec("sentence_transformers") is None:
        print("REAL DENSE MODEL TEST SKIPPED")
        print("Reason: 当前 pasa_rl 环境缺少 sentence_transformers，未自动安装。")
        return 0
    if args.model_name_or_path is None:
        print("REAL DENSE MODEL TEST SKIPPED")
        print("Reason: 未配置本地 model_name_or_path，未尝试下载模型。")
        return 0
    if not args.model_name_or_path.is_dir():
        print("REAL DENSE MODEL TEST SKIPPED")
        print(f"Reason: 本地 embedding 模型目录不存在: {args.model_name_or_path}")
        return 0

    corpus = PaperCorpus.from_jsonl(args.corpus)
    tasks = load_tasks_jsonl(args.tasks)
    try:
        embedder = SentenceTransformerEmbedder(
            args.model_name_or_path,
            device=args.device,
            batch_size=args.batch_size,
            normalize_embeddings=True,
        )
    except (EmbeddingDependencyError, EmbeddingModelUnavailableError) as exc:
        print("REAL DENSE MODEL TEST SKIPPED")
        print(f"Reason: {exc}")
        return 0

    bm25 = BM25Retriever(corpus)
    index_files = [args.index_dir / name for name in ("embeddings.npy", "paper_ids.json", "metadata.json")]
    if all(path.is_file() for path in index_files):
        dense = DenseRetriever.from_index(corpus, embedder, args.index_dir)
    elif any(path.exists() for path in index_files):
        raise RuntimeError("Dense index 不完整；请移走损坏索引后重新构建")
    else:
        dense = DenseRetriever(corpus, embedder)
        dense.save_index(args.index_dir)
    hybrid = HybridRetriever(
        bm25,
        dense,
        HybridConfig(
            bm25_top_n=max(50, args.top_k),
            dense_top_n=max(50, args.top_k),
            final_top_k=args.top_k,
        ),
    )
    report = compare_retrievers(
        {"bm25": bm25, "dense": dense, "hybrid": hybrid},
        tasks,
        args.top_k,
    )
    report["embedding_model"] = embedder.model_identifier
    report["warning"] = FIXTURE_WARNING
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _print_table(report)
    _print_dense_diagnostics(report)
    print()
    print("WARNING:")
    print("This benchmark contains only 10 synthetic fixture papers")
    print("and 6 synthetic retrieval tasks.")
    print("The metrics validate the real Dense/Hybrid pipeline")
    print("but do not represent real academic retrieval quality.")
    print()
    print(f"Comparison saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

