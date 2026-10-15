"""只用 SciFact RL train/dev 生成 expert trajectories 和无泄漏 SFT JSONL。"""

import json
from copy import deepcopy
from pathlib import Path
import sys
from typing import Any

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.expert_planner import ExpertPlan, ExpertPlanner, ExpertPlannerConfig  # noqa: E402
from src.agent.prompt_serializer import PromptSerializerConfig  # noqa: E402
from src.benchmarks.runtime import build_retrievers  # noqa: E402
from src.environment.capabilities import EnvironmentCapabilities  # noqa: E402
from src.environment.config import EnvironmentConfig  # noqa: E402
from src.environment.literature_env import LiteratureEnvironment  # noqa: E402
from src.evaluation.tasks import RetrievalTask, load_tasks_jsonl  # noqa: E402
from src.retrieval.base import Retriever  # noqa: E402
from src.retrieval.corpus import PaperCorpus  # noqa: E402
from src.retrieval.hybrid_retriever import HybridConfig, HybridRetriever  # noqa: E402
from src.retrieval.types import RetrievalResult  # noqa: E402
from src.training.sft_audit import audit_sft_samples, validate_sft_source_path  # noqa: E402
from src.training.sft_examples import trajectory_to_sft_examples  # noqa: E402
from src.training.sft_manifest import build_sft_manifest  # noqa: E402


BENCHMARK = PROJECT_ROOT / "data" / "benchmarks" / "scifact" / "processed"
TRAIN_SOURCE = BENCHMARK / "rl_train_tasks.jsonl"
DEV_SOURCE = BENCHMARK / "rl_dev_tasks.jsonl"
CORPUS_PATH = BENCHMARK / "papers.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "data" / "sft" / "scifact"
RESULTS_DIR = PROJECT_ROOT / "results" / "sft_generation"
RETRIEVAL_CONFIG_PATH = PROJECT_ROOT / "configs" / "retrieval.yaml"
ENVIRONMENT_CONFIG_PATH = PROJECT_ROOT / "configs" / "environment.yaml"


class CachingRetriever(Retriever):
    """缓存固定 query/top_k 排名，使五策略搜索不重复运行相同 Retriever。"""

    def __init__(self, retriever: Retriever) -> None:
        self.retriever = retriever
        self.cache: dict[tuple[str, int], list[RetrievalResult]] = {}

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """首次真实检索，随后返回结果深拷贝防止轨迹间相互修改。"""

        key = (query, top_k)
        if key not in self.cache:
            self.cache[key] = self.retriever.retrieve(query, top_k)
        return deepcopy(self.cache[key])


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """写入稳定 UTF-8 JSONL。"""

    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _quality_report(
    split: str,
    tasks: list[RetrievalTask],
    plans: list[ExpertPlan],
    samples: list[dict[str, Any]],
    audit: dict[str, Any],
) -> dict[str, Any]:
    """汇总 primary/alternative 轨迹质量、hard task 和数据审计。"""

    experts = [
        expert
        for plan in plans
        for expert in ([plan.primary] if plan.primary is not None else []) + plan.alternatives
    ]
    scores = [expert.score for expert in experts]
    return {
        "split": split,
        "num_tasks": len(tasks),
        "num_trajectories": len(experts),
        "num_primary_trajectories": sum(plan.primary is not None for plan in plans),
        "num_alternative_trajectories": sum(len(plan.alternatives) for plan in plans),
        "num_samples": len(samples),
        "success_rate": float(np.mean([score.success for score in scores])) if scores else 0.0,
        "mean_final_recall": float(np.mean([score.final_recall for score in scores])) if scores else 0.0,
        "mean_final_mrr": float(np.mean([score.final_mrr for score in scores])) if scores else 0.0,
        "mean_reward": float(np.mean([score.total_reward for score in scores])) if scores else 0.0,
        "mean_steps": float(np.mean([score.steps for score in scores])) if scores else 0.0,
        "hard_task_count": sum(plan.needs_rewrite for plan in plans),
        "needs_rewrite_count": sum(plan.needs_rewrite for plan in plans),
        "rewrite_query_coverage": 0,
        "read_paper_coverage": 0,
        **audit,
    }


def _generate_split(
    split: str,
    source: Path,
    planner: ExpertPlanner,
    corpus: PaperCorpus,
    capabilities: EnvironmentCapabilities,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """规划一个 RL split，排除 all-failed positive，并审计全部 messages。"""

    validate_sft_source_path(source)
    tasks = load_tasks_jsonl(source)
    plans: list[ExpertPlan] = []
    samples: list[dict[str, Any]] = []
    hard_tasks: list[dict[str, Any]] = []
    prompt_config = PromptSerializerConfig(max_candidate_papers=10, max_abstract_chars=400)
    for index, task in enumerate(tasks, start=1):
        plan = planner.plan(task)
        plans.append(plan)
        if plan.needs_rewrite:
            hard_tasks.append(
                {
                    "split": split,
                    "task_id": task.task_id,
                    "question": task.question,
                    "needs_rewrite": True,
                    "attempted_scores": {
                        name: score.to_dict() for name, score in plan.attempted_scores.items()
                    },
                }
            )
        else:
            selected = [plan.primary, *plan.alternatives]
            for expert in selected:
                assert expert is not None
                samples.extend(
                    example.to_dict()
                    for example in trajectory_to_sft_examples(
                        expert, corpus, capabilities, prompt_config
                    )
                )
        if index % 50 == 0 or index == len(tasks):
            print(f"{split} planning: {index}/{len(tasks)}", flush=True)
    audit_report = audit_sft_samples(samples, tasks)
    if (
        audit_report.parse_success_rate != 1.0
        or audit_report.validation_success_rate != 1.0
        or audit_report.invalid_target_count
        or audit_report.leakage_count
    ):
        raise RuntimeError(f"SFT audit failed: {audit_report.to_dict()}")
    report = _quality_report(split, tasks, plans, samples, audit_report.to_dict())
    return samples, report, hard_tasks


def main() -> None:
    """生成 train/dev JSONL、quality report、hard tasks 和可追溯 manifest。"""

    validate_sft_source_path(TRAIN_SOURCE)
    validate_sft_source_path(DEV_SOURCE)
    retrieval_config = yaml.safe_load(RETRIEVAL_CONFIG_PATH.read_text(encoding="utf-8"))
    environment_config = EnvironmentConfig.from_mapping(
        yaml.safe_load(ENVIRONMENT_CONFIG_PATH.read_text(encoding="utf-8"))
    )
    corpus = PaperCorpus.from_jsonl(CORPUS_PATH)
    base, _ = build_retrievers(
        corpus,
        retrieval_config,
        BENCHMARK / "indexes" / "dense",
        requested={"bm25", "dense"},
    )#配置bm25与dense检索方法
    bm25 = CachingRetriever(base["bm25"])
    dense = CachingRetriever(base["dense"])
    rrf = retrieval_config["rrf"]#输出混合检索的分数
    hybrid = CachingRetriever(
        HybridRetriever(
            bm25,
            dense,
            HybridConfig(
                bm25_top_n=rrf.get("bm25_top_n", 50),
                dense_top_n=rrf.get("dense_top_n", 50),
                rrf_k=rrf.get("k", 60),
                final_top_k=max(20, rrf.get("final_top_k", 20)),
            ),
        )#混合检索结果
    )
    capabilities = EnvironmentCapabilities.scifact()
    environment = LiteratureEnvironment(
        corpus,
        bm25,
        dense,
        hybrid_retriever=hybrid,
        config=environment_config,
        capabilities=capabilities,
    )#真正执行搜索、排序、返回topk的函数
    planner = ExpertPlanner(
        environment,
        ExpertPlannerConfig(top_k=10, max_alternative_trajectories=1),
    )#使用五种专家策略进行输出打分，并且返回最优方案
    train_samples, train_report, train_hard = _generate_split(
        "train", TRAIN_SOURCE, planner, corpus, capabilities
    )#产生最终数据
    dev_samples, dev_report, dev_hard = _generate_split(
        "dev", DEV_SOURCE, planner, corpus, capabilities
    )
    train_task_ids = {sample["task_id"] for sample in train_samples}
    dev_task_ids = {sample["task_id"] for sample in dev_samples}
    if train_task_ids & dev_task_ids:
        raise RuntimeError("SFT train/dev sample task_id 发生交集")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train_output = OUTPUT_DIR / "train.jsonl"
    dev_output = OUTPUT_DIR / "dev.jsonl"
    _write_jsonl(train_output, train_samples)
    _write_jsonl(dev_output, dev_samples)
    (RESULTS_DIR / "scifact_train_report.json").write_text(
        json.dumps(train_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (RESULTS_DIR / "scifact_dev_report.json").write_text(
        json.dumps(dev_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (RESULTS_DIR / "hard_tasks.json").write_text(
        json.dumps(train_hard + dev_hard, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = build_sft_manifest(
        train_source=TRAIN_SOURCE,
        dev_source=DEV_SOURCE,
        train_output=train_output,
        dev_output=dev_output,
        reward_config=ENVIRONMENT_CONFIG_PATH,
        retrieval_config=RETRIEVAL_CONFIG_PATH,
        num_train_samples=len(train_samples),
        num_dev_samples=len(dev_samples),
    )
    (OUTPUT_DIR / "sft_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("SCIFACT SFT DATA BUILD PASS")
    print(f"train samples={len(train_samples)} dev samples={len(dev_samples)}")
    print(f"hard tasks={len(train_hard) + len(dev_hard)}")


if __name__ == "__main__":
    main()
