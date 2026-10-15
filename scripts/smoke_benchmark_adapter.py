"""使用人工 BEIR fixture 验证 benchmark 转换与现有检索链路兼容性。"""

import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarks.beir_adapter import BEIRAdapter  # noqa: E402
from src.evaluation.evaluator import evaluate_retriever  # noqa: E402
from src.evaluation.tasks import load_tasks_jsonl  # noqa: E402
from src.retrieval.bm25_retriever import BM25Retriever  # noqa: E402
from src.retrieval.corpus import PaperCorpus  # noqa: E402


def main() -> None:
    """转换 fixture，并验证标准 loader、Retriever 和 Evaluator 无需修改。"""

    print("BENCHMARK ADAPTER SMOKE TEST")
    print()
    source = PROJECT_ROOT / "tests" / "fixtures" / "beir_sample"
    output = PROJECT_ROOT / "results" / "benchmark_adapter_smoke"
    report = BEIRAdapter("beir-sample").convert(source, output)
    assert report.num_documents == 6
    print("[PASS] Corpus converted")
    assert report.num_queries == 4
    print("[PASS] Queries converted")
    assert report.num_qrels == 6
    print("[PASS] Qrels mapped")
    test_tasks = load_tasks_jsonl(output / "test_tasks.jsonl")
    assert test_tasks[0].relevance_grades is not None
    print("[PASS] Graded relevance")
    manifest = json.loads((output / "benchmark_manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_format"] == "beir-style" and manifest["split_source"] == "original"
    print("[PASS] Manifest generated")
    corpus = PaperCorpus.from_jsonl(output / "papers.jsonl")
    assert len(corpus) == 6
    print("[PASS] Project PaperCorpus loaded")
    all_tasks = []
    for split in ("train", "dev", "test"):
        all_tasks.extend(load_tasks_jsonl(output / f"{split}_tasks.jsonl"))
    assert len(all_tasks) == 4
    print("[PASS] Project Task loader loaded")
    retriever = BM25Retriever(corpus)
    assert retriever.retrieve(all_tasks[0].question, top_k=3)
    print("[PASS] Existing Retriever compatible")
    assert evaluate_retriever(retriever, test_tasks, top_k=6).num_tasks == len(test_tasks)
    print("[PASS] Existing Evaluator compatible")
    print()
    print("BENCHMARK ADAPTER SMOKE TEST PASS")


if __name__ == "__main__":
    main()

