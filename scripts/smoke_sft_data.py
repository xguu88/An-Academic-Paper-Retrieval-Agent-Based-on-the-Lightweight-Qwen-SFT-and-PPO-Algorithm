"""使用人工 fixture 完全离线验证 Hybrid expert 与 SFT 数据链路。"""

import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.actions import ActionType, AgentAction  # noqa: E402
from src.agent.expert_planner import ExpertPlanner  # noqa: E402
from src.agent.parser import parse_action  # noqa: E402
from src.environment.capabilities import EnvironmentCapabilities  # noqa: E402
from src.environment.config import EnvironmentConfig  # noqa: E402
from src.environment.literature_env import LiteratureEnvironment  # noqa: E402
from src.evaluation.tasks import RetrievalTask  # noqa: E402
from src.retrieval.bm25_retriever import BM25Retriever  # noqa: E402
from src.retrieval.corpus import PaperCorpus  # noqa: E402
from src.retrieval.dense_retriever import DenseRetriever  # noqa: E402
from src.retrieval.hybrid_retriever import HybridConfig, HybridRetriever  # noqa: E402
from src.retrieval.testing import HashingTextEmbedder  # noqa: E402
from src.retrieval.types import RetrievalSource  # noqa: E402
from src.training.sft_audit import audit_sft_samples, validate_sft_source_path  # noqa: E402
from src.training.sft_examples import trajectory_to_sft_examples  # noqa: E402
from src.training.sft_manifest import build_sft_manifest  # noqa: E402


FIXTURES = PROJECT_ROOT / "tests" / "fixtures"
PROCESSED = PROJECT_ROOT / "data" / "benchmarks" / "scifact" / "processed"


def main() -> None:
    """逐项检查 Action、Environment、Expert、prompt、audit 与隔离。"""

    print("SFT DATA SMOKE TEST")
    print()
    action = parse_action('{"action":"HybridSearch","query":"retrieval","top_k":10}')
    assert action.action is ActionType.HYBRID_SEARCH
    print("[PASS] HybridSearch action")

    corpus = PaperCorpus.from_jsonl(FIXTURES / "papers.jsonl")
    bm25 = BM25Retriever(corpus)
    dense = DenseRetriever(corpus, HashingTextEmbedder(dimension=256))
    hybrid = HybridRetriever(
        bm25,
        dense,
        HybridConfig(bm25_top_n=10, dense_top_n=10, final_top_k=10),
    )
    capabilities = EnvironmentCapabilities.scifact()
    environment = LiteratureEnvironment(
        corpus,
        bm25,
        dense,
        hybrid_retriever=hybrid,
        config=EnvironmentConfig(max_steps=6, max_searches=3, max_reads=2),
        capabilities=capabilities,
    )
    task = RetrievalTask("bm25", "BM25 lexical keyword document ranking", ["p-bm25"])
    environment.reset(task)
    _, observation, _, _, _ = environment.step(
        AgentAction(ActionType.HYBRID_SEARCH, query=task.question, top_k=10)
    )
    assert observation.retrieved_results
    assert all(item.source is RetrievalSource.RRF for item in observation.retrieved_results)
    print("[PASS] Hybrid environment execution")

    plan = ExpertPlanner(environment).plan(task)
    assert plan.primary is not None and plan.primary.score.success
    print("[PASS] Expert trajectory generation")

    examples = trajectory_to_sft_examples(plan.primary, corpus, capabilities)
    assert examples and "Original query:" in examples[0].messages[1]["content"]
    print("[PASS] Prompt serialization")

    for example in examples:
        parse_action(example.messages[2]["content"])
    print("[PASS] Action target validation")

    serialized = json.dumps([example.to_dict() for example in examples], ensure_ascii=False)
    assert "relevant_paper_ids" not in serialized and "qrels" not in serialized
    print("[PASS] Ground-truth isolation")

    assert len(examples) == len(plan.primary.trajectory.steps)
    print("[PASS] Trajectory expansion")

    audit = audit_sft_samples([example.to_dict() for example in examples], [task])
    assert audit.parse_success_rate == 1.0 and audit.leakage_count == 0
    print("[PASS] Dataset audit")

    manifest = build_sft_manifest(
        train_source=FIXTURES / "tasks.jsonl",
        dev_source=FIXTURES / "tasks.jsonl",
        train_output=FIXTURES / "papers.jsonl",
        dev_output=FIXTURES / "papers.jsonl",
        reward_config=PROJECT_ROOT / "configs" / "environment.yaml",
        retrieval_config=PROJECT_ROOT / "configs" / "retrieval.yaml",
        num_train_samples=len(examples),
        num_dev_samples=len(examples),
    )
    assert manifest["action_schema_version"] and not manifest["frozen_test_used"]
    print("[PASS] Manifest")

    validate_sft_source_path(PROCESSED / "rl_train_tasks.jsonl")
    validate_sft_source_path(PROCESSED / "rl_dev_tasks.jsonl")
    try:
        validate_sft_source_path(PROCESSED / "test_tasks.jsonl")
    except PermissionError:
        pass
    else:
        raise AssertionError("frozen test path must be rejected")
    print("[PASS] Train/dev isolation")
    print()
    print("SFT DATA SMOKE TEST PASS")


if __name__ == "__main__":
    main()

