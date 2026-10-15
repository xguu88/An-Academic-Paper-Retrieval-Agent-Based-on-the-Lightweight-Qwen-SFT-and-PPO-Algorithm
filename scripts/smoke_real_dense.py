"""使用配置中的本地 BGE 模型验证真实 Dense 检索、索引和 Evaluator。"""

import argparse
from pathlib import Path
import sys

import numpy as np
import yaml


# 直接运行脚本时导入项目源码；模型只允许读取本地目录。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluator import evaluate_retriever  # noqa: E402
from src.evaluation.tasks import load_tasks_jsonl  # noqa: E402
from src.retrieval.corpus import PaperCorpus  # noqa: E402
from src.retrieval.dense_retriever import DenseRetriever  # noqa: E402
from src.retrieval.sentence_transformer_embedder import (  # noqa: E402
    SentenceTransformerEmbedder,
)
from src.retrieval.types import RetrievalResult, RetrievalSource  # noqa: E402


def parse_args() -> argparse.Namespace:
    """解析配置、fixture 语料和任务路径。"""

    parser = argparse.ArgumentParser(description="真实本地 BGE Dense Retrieval smoke")
    parser.add_argument("--config", type=Path, default=Path("configs/retrieval.yaml"))
    parser.add_argument("--corpus", type=Path, default=Path("tests/fixtures/papers.jsonl"))
    parser.add_argument("--tasks", type=Path, default=Path("tests/fixtures/tasks.jsonl"))
    return parser.parse_args()


def main() -> None:
    """验证本地加载、384 维归一化、索引持久化和检索接口。"""

    args = parse_args()
    print("REAL DENSE RETRIEVAL SMOKE TEST")
    print()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    dense_config = config["dense"]
    model_path = Path(dense_config["model_name_or_path"])
    assert model_path.is_dir()
    print("[PASS] Local BGE model found")

    embedder = SentenceTransformerEmbedder(
        model_path,
        device=dense_config["device"],
        batch_size=dense_config["batch_size"],
        normalize_embeddings=dense_config["normalize_embeddings"],
        allow_remote=False,
    )
    print("[PASS] SentenceTransformerEmbedder loaded")
    vectors = embedder.encode(["transformer attention", "dense academic retrieval"])
    assert vectors.shape == (2, 384) and embedder.dimension == 384
    print("[PASS] Embedding shape = 384")
    assert np.all(np.isfinite(vectors))
    print("[PASS] Embedding finite")
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms[norms > 0], 1.0, atol=1e-5)
    assert embedder.normalize_embeddings is True
    print("[PASS] Embedding normalized")

    corpus = PaperCorpus.from_jsonl(args.corpus)
    dense = DenseRetriever(corpus, embedder)
    assert dense.document_embeddings.shape == (10, 384)
    print("[PASS] Dense index built")
    index_dir = Path(dense_config["index_dir"])
    dense.save_index(index_dir)
    assert all(
        (index_dir / name).is_file()
        for name in ("embeddings.npy", "paper_ids.json", "metadata.json")
    )
    print("[PASS] Dense index saved")
    reloaded = DenseRetriever.from_index(corpus, embedder, index_dir)
    assert np.array_equal(reloaded.document_embeddings, dense.document_embeddings)
    print("[PASS] Dense index reloaded")

    results = reloaded.retrieve("dense academic retrieval", top_k=10)
    assert results and len(results) == len(corpus)
    print("[PASS] Dense retrieval")
    assert all(
        isinstance(result, RetrievalResult)
        and result.source is RetrievalSource.DENSE
        and result.rank == index + 1
        for index, result in enumerate(results)
    )
    print("[PASS] RetrievalResult schema")
    tasks = load_tasks_jsonl(args.tasks)
    assert evaluate_retriever(reloaded, tasks, top_k=10).num_tasks == len(tasks)
    print("[PASS] Evaluator compatibility")
    print()
    print("REAL DENSE RETRIEVAL SMOKE TEST PASS")


if __name__ == "__main__":
    main()

