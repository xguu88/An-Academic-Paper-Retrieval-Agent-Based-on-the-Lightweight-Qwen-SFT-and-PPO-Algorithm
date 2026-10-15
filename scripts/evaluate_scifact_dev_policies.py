"""只在 SciFact RL dev split 上运行四种离线规则策略。"""

import json
from pathlib import Path
import sys

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.baselines import (  # noqa: E402
    HybridOnlyPolicy,
    KeywordOnlyPolicy,
    KeywordThenSemanticPolicy,
    SemanticOnlyPolicy,
)
from src.benchmarks.runtime import build_retrievers  # noqa: E402
from src.environment.capabilities import EnvironmentCapabilities  # noqa: E402
from src.environment.config import EnvironmentConfig  # noqa: E402
from src.environment.literature_env import LiteratureEnvironment  # noqa: E402
from src.evaluation.agent_evaluator import evaluate_agent_policy  # noqa: E402
from src.evaluation.tasks import load_tasks_jsonl  # noqa: E402
from src.retrieval.corpus import PaperCorpus  # noqa: E402


BENCHMARK = PROJECT_ROOT / "data" / "benchmarks" / "scifact" / "processed"
OUTPUT = PROJECT_ROOT / "results" / "agent_baselines" / "scifact_dev.json"


def main() -> None:
    """构建固定 BM25/BGE 后评估 dev；绝不读取或运行 test_tasks。"""

    retrieval_config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "retrieval.yaml").read_text(encoding="utf-8")
    )
    environment_config = EnvironmentConfig.from_mapping(
        yaml.safe_load(
            (PROJECT_ROOT / "configs" / "environment.yaml").read_text(encoding="utf-8")
        )
    )
    corpus = PaperCorpus.from_jsonl(BENCHMARK / "papers.jsonl")
    tasks = load_tasks_jsonl(BENCHMARK / "rl_dev_tasks.jsonl")
    retrievers, build_times = build_retrievers(
        corpus,
        retrieval_config,
        BENCHMARK / "indexes" / "dense",
        requested={"bm25", "dense", "hybrid"},
    )
    environment = LiteratureEnvironment(
        corpus,
        retrievers["bm25"],
        retrievers["dense"],
        hybrid_retriever=retrievers["hybrid"],
        config=environment_config,
        capabilities=EnvironmentCapabilities.scifact(),
    )
    policies = [
        KeywordOnlyPolicy(),
        SemanticOnlyPolicy(),
        HybridOnlyPolicy(),
        KeywordThenSemanticPolicy(),
    ]
    results = {}
    for policy in policies:
        metrics, trajectories = evaluate_agent_policy(policy, environment, tasks, top_k=20)
        results[policy.name] = metrics.to_dict()
        # 序列化全部轨迹以验证数据契约，但汇总文件只保存指标和样例。
        sample = trajectories[0].to_dict()
        json.dumps(sample, ensure_ascii=False)
        results[policy.name]["trajectory_example"] = sample
        print(f"[PASS] {policy.name}: {len(trajectories)} dev trajectories", flush=True)
    report = {
        "dataset": "scifact",
        "split": "rl_dev",
        "num_documents": len(corpus),
        "num_tasks": len(tasks),
        "candidate_memory": "latest retrieval ranking; cumulative retrieved IDs only deduplicate novelty",
        "capabilities": EnvironmentCapabilities.scifact().to_dict(),
        "environment_config": {
            "max_steps": environment_config.max_steps,
            "max_searches": environment_config.max_searches,
            "max_reads": environment_config.max_reads,
            "search_top_k": environment_config.search_top_k,
            "stop_success_k": environment_config.stop_success_k,
        },
        "build_times": build_times.to_dict(),
        "policies": results,
        "test_used": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("SCIFACT DEV RULE-BASED BASELINES PASS")
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
