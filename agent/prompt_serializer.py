"""把 Agent 可见状态稳定序列化为未来 Qwen 的 user prompt。"""

import json
from dataclasses import dataclass
from typing import Any, Sequence

from src.agent.state import AgentState
from src.environment.capabilities import EnvironmentCapabilities
from src.retrieval.corpus import PaperCorpus


PROMPT_SERIALIZER_VERSION = "visible-state-v1"


@dataclass(frozen=True, slots=True)
class PromptSerializerConfig:
    """限制候选论文数量与摘要字符数，防止 context 无界增长。"""

    max_candidate_papers: int = 10
    max_abstract_chars: int = 400

    def validate(self) -> None:
        """检查两个截断参数均为正整数。"""

        for name in ("max_candidate_papers", "max_abstract_chars"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} 必须是大于 0 的整数")


def _latest_result_metadata(
    observation_history: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """从最近到最早查找每个候选的可见 source/rank，不读取 reward 或 info。"""

    found: dict[str, dict[str, Any]] = {}
    for observation in reversed(observation_history):
        for result in observation.get("retrieved_results", []):
            paper_id = result.get("paper_id")
            if isinstance(paper_id, str) and paper_id not in found:
                found[paper_id] = result
    return found


def visible_candidate_ids(
    state: AgentState,
    config: PromptSerializerConfig | None = None,
) -> list[str]:
    """返回 prompt 中实际展示的候选 ID，供 label leakage audit 使用。"""

    selected = config or PromptSerializerConfig()
    selected.validate()
    return list(state.candidate_paper_ids[: selected.max_candidate_papers])


def serialize_agent_prompt(
    state: AgentState,
    observation_history: Sequence[dict[str, Any]],
    capabilities: EnvironmentCapabilities,
    corpus: PaperCorpus,
    config: PromptSerializerConfig | None = None,
) -> str:
    """只序列化可见 query、预算、动作历史、候选、阅读和最近观察。"""

    selected = config or PromptSerializerConfig()
    selected.validate()
    result_metadata = _latest_result_metadata(observation_history)
    lines = [
        f"Original query: {state.original_query}",
        f"Current query: {state.current_query}",
        f"Step: {state.step}/{state.max_steps}",
        "Remaining budget: "
        + json.dumps(
            {
                "steps": state.remaining_steps,
                "searches_used": state.search_count,
                "reads_used": state.read_count,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        "Capabilities: "
        + json.dumps(capabilities.to_dict(), ensure_ascii=False, sort_keys=True),
        "Action history: "
        + json.dumps(
            [action.to_dict() for action in state.action_history],
            ensure_ascii=False,
            sort_keys=True,
        ),
        "Seen queries: " + json.dumps(state.seen_queries, ensure_ascii=False),
        "Candidates:",
    ]
    candidates = visible_candidate_ids(state, selected)
    if not candidates:
        lines.append("(none)")
    for fallback_rank, paper_id in enumerate(candidates, start=1):
        paper = corpus.get(paper_id)
        if paper is None:
            continue
        metadata = result_metadata.get(paper_id, {})
        source = metadata.get("source") or "unknown"
        rank = metadata.get("rank") or fallback_rank
        abstract = (paper.abstract or "")[: selected.max_abstract_chars]
        lines.extend(
            [
                f"{fallback_rank}. {paper.paper_id}",
                f"   Title: {paper.title}",
                f"   Source: {source}",
                f"   Rank: {rank}",
                f"   Abstract: {abstract}",
            ]
        )
    lines.append("Read papers:")
    if not state.read_paper_ids:
        lines.append("(none)")
    else:
        for paper_id in state.read_paper_ids:
            paper = corpus.get(paper_id)
            if paper is not None:
                lines.append(f"- {paper.paper_id}: {paper.title}")
    lines.append("Latest observation:")
    if observation_history:
        latest = observation_history[-1]
        safe_latest = {
            key: latest.get(key)
            for key in (
                "action_type",
                "query_used",
                "new_paper_ids",
                "read_paper",
                "message",
                "remaining_budget",
                "terminated",
                "valid_action",
            )
            if key in latest
        }
        lines.append(json.dumps(safe_latest, ensure_ascii=False, sort_keys=True))
    else:
        lines.append("(none)")
    lines.append("Choose the next single JSON Action.")
    return "\n".join(lines)


def serialize_action_target(action: Any) -> str:
    """把 AgentAction 转换为不含解释文字的紧凑严格 JSON target。"""

    from src.agent.actions import AgentAction

    if not isinstance(action, AgentAction):
        raise TypeError("action 必须是 AgentAction")
    return json.dumps(action.to_dict(), ensure_ascii=False, separators=(",", ":"))

