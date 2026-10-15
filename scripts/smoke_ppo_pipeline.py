"""完全离线的 2-task、2-step PPO 闭环 smoke，不读取 SciFact frozen test。"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.environment.capabilities import EnvironmentCapabilities
from src.environment.config import EnvironmentConfig
from src.environment.literature_env import LiteratureEnvironment
from src.evaluation.tasks import RetrievalTask
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.corpus import PaperCorpus
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridConfig, HybridRetriever
from src.retrieval.testing import HashingTextEmbedder
from src.training.ppo_rollout import PPORolloutCollector
from src.training.ppo_testing import TinyTrainablePPOPolicy
from src.training.ppo_trainer import PPOTrainer
from src.training.ppo_types import PPOConfig


def build_environment() -> LiteratureEnvironment:
    """只加载 tests 人工 fixture，明确避开所有 SciFact split。"""

    corpus = PaperCorpus.from_jsonl(ROOT / "tests" / "fixtures" / "papers.jsonl")
    bm25 = BM25Retriever(corpus)
    dense = DenseRetriever(corpus, HashingTextEmbedder(dimension=64))
    hybrid = HybridRetriever(
        bm25, dense, HybridConfig(bm25_top_n=10, dense_top_n=10, final_top_k=10)
    )
    return LiteratureEnvironment(
        corpus,
        bm25,
        dense,
        hybrid_retriever=hybrid,
        config=EnvironmentConfig(max_steps=3, max_searches=3, max_reads=2),
        capabilities=EnvironmentCapabilities.scifact(),
    )


def main() -> int:
    """验证 rollout→GAE→loss→backward→optimizer step 与 invalid JSON。"""

    torch.manual_seed(7)
    environment = build_environment()
    responses = [
        '{"action":"KeywordSearch","query":"BM25 lexical keyword document ranking","top_k":10}',
        '{"action":"Stop","candidate_ids":[]}',
        "invalid-json",
        '{"action":"Stop","candidate_ids":[]}',
    ]
    policy = TinyTrainablePPOPolicy(responses, with_reference=True)
    config = PPOConfig(
        learning_rate=0.05,
        max_steps=2,
        rollout_tasks_per_batch=2,
        ppo_epochs=1,
        minibatch_size=2,
        use_reference_policy=True,
        kl_coef=0.02,
        entropy_coef=0.0,
        seed=7,
    )
    trainer = PPOTrainer(
        policy,
        PPORolloutCollector(environment, policy, config),
        config,
    )
    before = {
        "policy": policy.policy_logprob.detach().clone(),
        "value": policy.value_bias.detach().clone(),
        "reference": policy.reference_anchor.detach().clone(),
    }
    tasks = [
        RetrievalTask(
            "smoke-good", "BM25 lexical keyword document ranking", ["p-bm25"]
        ),
        RetrievalTask("smoke-invalid", "graph retrieval", ["p-graph"]),
    ]
    report = trainer.train_batch(tasks)
    changed = not torch.equal(
        policy.policy_logprob.detach(), before["policy"]
    ) or not torch.equal(policy.value_bias.detach(), before["value"])
    assert changed, "trainable policy/value parameter did not change"
    assert torch.equal(policy.reference_anchor.detach(), before["reference"])
    assert policy.reference_anchor.grad is None
    assert report["invalid_action_rate"] > 0
    assert report["frozen_test_read"] is False
    print(
        f"episodes={report['num_episodes']} rollouts={report['num_rollouts']} "
        f"optimizer_steps={report['optimizer_steps']} invalid_rate={report['invalid_action_rate']:.3f}"
    )
    print("PPO PIPELINE SMOKE TEST PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
