"""把 expert trajectory 展开为稳定的 visible state → next JSON Action 样本。"""

from dataclasses import asdict, dataclass
from typing import Any

from src.agent.expert_planner import ExpertTrajectory
from src.agent.prompt_serializer import (
    PromptSerializerConfig,
    serialize_action_target,
    serialize_agent_prompt,
    visible_candidate_ids,
)
from src.agent.prompts import build_system_prompt
from src.environment.capabilities import EnvironmentCapabilities
from src.retrieval.corpus import PaperCorpus


@dataclass(slots=True)
class SFTExample:
    """保存 Qwen chat messages 与不送入模型的审计 metadata。"""

    sample_id: str
    task_id: str
    trajectory_id: str
    step: int
    messages: list[dict[str, str]]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """转换为 JSONL 兼容字典。"""

        return asdict(self)


def trajectory_to_sft_examples(
    expert: ExpertTrajectory,
    corpus: PaperCorpus,
    capabilities: EnvironmentCapabilities,
    prompt_config: PromptSerializerConfig | None = None,
) -> list[SFTExample]:
    """把 s0→a0、s1→a1 展开成独立 chat 样本，不暴露 reward 到 messages。"""

    selected = prompt_config or PromptSerializerConfig()
    system_prompt = build_system_prompt(capabilities)
    history: list[dict[str, Any]] = []
    examples: list[SFTExample] = []
    for index, transition in enumerate(expert.trajectory.steps):
        state = transition.state_before
        user_prompt = serialize_agent_prompt(
            state,
            history,
            capabilities,
            corpus,
            selected,
        )
        target = serialize_action_target(transition.action)
        examples.append(
            SFTExample(
                sample_id=f"{expert.trajectory_id}:step-{index}",
                task_id=expert.trajectory.task_id,
                trajectory_id=expert.trajectory_id,
                step=index,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": target},
                ],
                metadata={
                    "action_type": transition.action.action.value,
                    "source_policy": expert.source_policy,
                    "trajectory_reward": expert.score.total_reward,
                    "trajectory_success": expert.score.success,
                    "visible_candidate_ids": visible_candidate_ids(state, selected),
                    "visible_query_texts": [
                        state.original_query,
                        state.current_query,
                        *state.seen_queries,
                    ],
                },
            )
        )
        history.append(dict(transition.observation))
    return examples

