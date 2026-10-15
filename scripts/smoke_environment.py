"""用人工 fixture 完全离线验证 Action→Environment→Reward→Trajectory。"""

import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.actions import ActionType, AgentAction, CitationDirection  # noqa: E402
from src.environment.capabilities import EnvironmentCapabilities  # noqa: E402
from src.environment.config import EnvironmentConfig  # noqa: E402
from src.environment.literature_env import LiteratureEnvironment  # noqa: E402
from src.evaluation.tasks import RetrievalTask  # noqa: E402
from src.retrieval.bm25_retriever import BM25Retriever  # noqa: E402
from src.retrieval.corpus import PaperCorpus  # noqa: E402
from src.retrieval.dense_retriever import DenseRetriever  # noqa: E402
from src.retrieval.testing import HashingTextEmbedder  # noqa: E402


FIXTURES = PROJECT_ROOT / "tests" / "fixtures"


def _environment(corpus: PaperCorpus, config: EnvironmentConfig | None = None) -> LiteratureEnvironment:
    """构造无网络、无模型下载的 fixture 环境。"""

    return LiteratureEnvironment(
        corpus,
        BM25Retriever(corpus),
        DenseRetriever(corpus, HashingTextEmbedder(dimension=256)),
        config=config or EnvironmentConfig(max_steps=8, max_searches=4, max_reads=3),
        capabilities=EnvironmentCapabilities.scifact(),
    )


def main() -> None:
    """逐项断言环境动作、奖励、预算和标签隔离。"""

    print("ENVIRONMENT SMOKE TEST")
    print()
    corpus = PaperCorpus.from_jsonl(FIXTURES / "papers.jsonl")
    environment = _environment(corpus)
    task = RetrievalTask("smoke", "BM25 lexical keyword document ranking", ["p-bm25"])
    state, reset_observation = environment.reset(task)
    assert state.step == 0 and not reset_observation.terminated
    print("[PASS] Reset")

    state, _, _, _, _ = environment.step(
        AgentAction(ActionType.REWRITE_QUERY, query="BM25 lexical ranking")
    )
    assert state.current_query == "BM25 lexical ranking"
    print("[PASS] RewriteQuery")

    search = AgentAction(ActionType.KEYWORD_SEARCH, query=state.current_query, top_k=10)
    state, keyword_observation, first_reward, _, first_info = environment.step(search)
    assert keyword_observation.retrieved_results
    print("[PASS] KeywordSearch")

    state, semantic_observation, _, _, _ = environment.step(
        AgentAction(ActionType.SEMANTIC_SEARCH, query=state.current_query, top_k=10)
    )
    assert semantic_observation.retrieved_results
    print("[PASS] SemanticSearch")

    assert state.candidate_paper_ids == [
        item.paper_id for item in semantic_observation.retrieved_results
    ]
    assert len(state.retrieved_paper_ids) == len(set(state.retrieved_paper_ids))
    print("[PASS] Candidate memory")

    state, read_observation, _, _, _ = environment.step(
        AgentAction(ActionType.READ_PAPER, paper_id="p-bm25")
    )
    assert read_observation.read_paper is not None
    print("[PASS] ReadPaper")

    assert first_info["reward_components"]["delta_recall"] > 0 and first_reward > 0
    print("[PASS] Reward delta")

    duplicate_environment = _environment(corpus)
    duplicate_environment.reset(task)
    duplicate_environment.step(search)
    _, _, _, _, duplicate_info = duplicate_environment.step(search)
    assert duplicate_info["reward_components"]["duplicate_query_penalty"] < 0
    print("[PASS] Duplicate penalty")

    _, rejected, rejected_reward, _, _ = duplicate_environment.step(
        AgentAction(
            ActionType.EXPAND_CITATION,
            paper_id="p-bm25",
            direction=CitationDirection.REFERENCES,
        )
    )
    assert not rejected.valid_action and rejected_reward < 0
    print("[PASS] Capability rejection")

    budget_environment = _environment(
        corpus, EnvironmentConfig(max_steps=1, max_searches=1, max_reads=1)
    )
    budget_environment.reset(task)
    _, _, _, done, budget_info = budget_environment.step(
        AgentAction(ActionType.REWRITE_QUERY, query="budget")
    )
    assert done and budget_info["reward_components"]["budget_penalty"] < 0
    print("[PASS] Budget")

    stop_environment = _environment(corpus)
    stop_environment.reset(task)
    stop_environment.step(search)
    _, stop_observation, _, done, stop_info = stop_environment.step(
        AgentAction(ActionType.STOP)
    )
    assert done and stop_info["success"] and stop_observation.terminated
    print("[PASS] Stop")

    assert stop_environment.trajectory is not None
    encoded_trajectory = json.dumps(stop_environment.trajectory.to_dict(), ensure_ascii=False)
    assert "reward_components" in encoded_trajectory
    print("[PASS] Trajectory serialization")

    secret = "never-visible-ground-truth-id"
    isolated = _environment(corpus)
    isolated_state, isolated_observation = isolated.reset(
        RetrievalTask("private", "visible question", [secret])
    )
    encoded_visible = json.dumps(
        {
            "state": isolated_state.to_dict(),
            "observation": isolated_observation.to_dict(),
        },
        ensure_ascii=False,
    )
    assert secret not in encoded_visible and "relevant_paper_ids" not in encoded_visible
    print("[PASS] Ground-truth isolation")
    print()
    print("ENVIRONMENT SMOKE TEST PASS")


if __name__ == "__main__":
    main()

