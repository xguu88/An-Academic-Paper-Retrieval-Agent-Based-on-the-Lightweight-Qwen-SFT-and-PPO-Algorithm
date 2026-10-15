"""使用用户提供的本地 CrossEncoder 比较三路 baseline 与 Hybrid+Reranker。"""

import argparse
import json
from pathlib import Path
import sys

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarks.runtime import build_retrievers, measure_query_latency  # noqa: E402
from src.evaluation.evaluator import evaluate_retriever  # noqa: E402
from src.evaluation.reranker_analysis import evaluate_candidate_pool  # noqa: E402
from src.evaluation.tasks import load_tasks_jsonl  # noqa: E402
from src.retrieval.corpus import PaperCorpus  # noqa: E402
from src.retrieval.cross_encoder_reranker import CrossEncoderReranker  # noqa: E402
from src.retrieval.reranked_retriever import (  # noqa: E402
    RerankedRetriever,
    RerankedRetrieverConfig,
)
from src.retrieval.sentence_transformer_cross_encoder import (  # noqa: E402
    SentenceTransformerCrossEncoderScorer,
)


def parse_args() -> argparse.Namespace:
    """解析 benchmark、split、配置和输出路径；不提供任何远程模型参数。"""

    parser = argparse.ArgumentParser(description="比较 Hybrid CrossEncoder 重排效果")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "dev", "test"), default="test")
    parser.add_argument("--config", type=Path, default=Path("configs/retrieval.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """严格加载本地模型，复用统一 Evaluator 与 query-only latency 计时。"""

    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    reranker_config = config["reranker"]
    model_path_value = reranker_config.get("model_name_or_path")
    if not model_path_value:
        raise RuntimeError("reranker.model_name_or_path 未配置；不会自动下载模型")
    model_path = Path(model_path_value)
    if not model_path.is_dir():
        raise FileNotFoundError(f"本地 CrossEncoder 模型目录不存在: {model_path}")

    corpus = PaperCorpus.from_jsonl(args.benchmark / "papers.jsonl")
    tasks = load_tasks_jsonl(args.benchmark / f"{args.split}_tasks.jsonl")
    if not tasks:
        raise ValueError(f"benchmark split 为空: {args.split}")
    max_candidate_depth = max(20, 50, int(reranker_config["candidate_top_n"]))
    config["rrf"] = dict(config["rrf"])
    config["rrf"]["final_top_k"] = max_candidate_depth
    retrievers, build_times = build_retrievers(
        corpus,
        config,
        args.benchmark / "indexes" / "dense",
        requested={"bm25", "dense", "hybrid"},
    )
    scorer = SentenceTransformerCrossEncoderScorer(
        model_path,
        device=reranker_config.get("device", "cpu"),
        batch_size=reranker_config.get("batch_size", 16),
        allow_remote=False,
    )
    retrievers["hybrid_reranker"] = RerankedRetriever(
        retrievers["hybrid"],
        CrossEncoderReranker(corpus, scorer),
        RerankedRetrieverConfig(
            candidate_top_n=int(reranker_config["candidate_top_n"]),
            final_top_k=int(reranker_config["final_top_k"]),
        ),
    )

    metrics = {}
    latencies = {}
    for name in ("bm25", "dense", "hybrid", "hybrid_reranker"):
        metrics[name] = evaluate_retriever(retrievers[name], tasks, top_k=20).to_dict()
        latency, _ = measure_query_latency(retrievers[name], tasks, top_k=20)
        latencies[name] = latency.to_dict()
    hybrid_rankings = {
        task.task_id: [
            item.paper_id for item in retrievers["hybrid"].retrieve(task.question, max_candidate_depth)
        ]
        for task in tasks
    }
    candidate_analysis = {
        str(depth): evaluate_candidate_pool(tasks, hybrid_rankings, depth).to_dict()
        for depth in (20, 50)
    }
    report = {
        "dataset_split": args.split,
        "documents": len(corpus),
        "queries": len(tasks),
        "reranker_model": str(model_path.resolve()),
        "metrics": metrics,
        "query_latency": latencies,
        "candidate_analysis": candidate_analysis,
        "build_times": build_times.to_dict(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("REAL CROSS-ENCODER RERANKER COMPARISON PASS")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

