"""与 Trainer 解耦的 LiteratureEnvironment PPO rollout collector。"""

from __future__ import annotations

from collections.abc import Sequence

from src.agent.actions import ActionType, AgentAction
from src.agent.parser import parse_action
from src.agent.prompt_serializer import PromptSerializerConfig, serialize_agent_prompt
from src.agent.prompts import build_system_prompt
from src.environment.literature_env import LiteratureEnvironment
from src.evaluation.tasks import RetrievalTask
from src.training.ppo_policy import PPOPolicyAdapter
from src.training.ppo_types import PPOConfig, PPORolloutEpisode, PPORolloutStep

FORBIDDEN_POLICY_FIELDS = (
    "relevant_paper_ids",
    "relevance_grades",
    "qrels",
    "ground_truth",
    "reward_components",
    "trajectory_reward",
    '"success":',
)


def assert_prompt_is_truth_free(prompt: str) -> None:
    """拒绝将结构化真值/reward 字段送入 PPO policy。"""

    lowered = prompt.lower()
    leaked = [field for field in FORBIDDEN_POLICY_FIELDS if field in lowered]
    if leaked:
        raise ValueError(f"PPO prompt 含禁止字段: {leaked}")


class PPORolloutCollector:
    """串行采集 task episode；解析失败转为环境可惩罚的 invalid action。"""

    def __init__(
        self,
        environment: LiteratureEnvironment,
        policy: PPOPolicyAdapter,
        config: PPOConfig,
        *,
        prompt_config: PromptSerializerConfig | None = None,
    ) -> None:
        self.environment = environment
        self.policy = policy
        self.config = config
        self.prompt_config = prompt_config or PromptSerializerConfig()

    def _messages(
        self, state: object, history: Sequence[dict[str, object]]
    ) -> tuple[str, list[dict[str, str]]]:
        user_prompt = serialize_agent_prompt(
            state,  # type: ignore[arg-type]
            history,
            self.environment.capabilities,
            self.environment.corpus,
            self.prompt_config,
        )
        assert_prompt_is_truth_free(user_prompt)
        messages = [
            {
                "role": "system",
                "content": build_system_prompt(self.environment.capabilities),
            },
            {"role": "user", "content": user_prompt},
        ]
        return user_prompt, messages

    @staticmethod
    def _parse_or_invalid(text: str) -> tuple[AgentAction, bool]:
        try:
            return parse_action(text), True
        except ValueError:
            # 复用 Environment 的 invalid-action reward，不伪装成合法 Stop。
            return AgentAction(ActionType.KEYWORD_SEARCH, query=None), False

    def collect(self, task: RetrievalTask) -> PPORolloutEpisode:
        """采集一条 episode，尊重 environment done 与 collector max_steps。"""

        state, _ = self.environment.reset(task)
        history: list[dict[str, object]] = []
        episode = PPORolloutEpisode(task_id=task.task_id)
        for step_index in range(self.config.max_steps):
            prompt, messages = self._messages(state, history)
            generation = self.policy.generate(messages)
            generation.validate()
            action, parsed_valid = self._parse_or_invalid(generation.text)
            next_state, observation, reward, env_done, _ = self.environment.step(action)
            observation_dict = observation.to_dict()
            valid_action = bool(parsed_valid and observation.valid_action)
            collector_truncated = bool(
                (step_index + 1 >= self.config.max_steps and not env_done)
                or (
                    self.config.terminate_on_invalid_action
                    and not valid_action
                    and not env_done
                )
            )
            reference = generation.reference_logprob
            if self.config.use_reference_policy and reference is None:
                raise RuntimeError(
                    "use_reference_policy=True 但 rollout 缺少 reference_logprob"
                )
            reference_action_log_ratio = (
                generation.old_logprob - reference if reference is not None else 0.0
            )
            kl_penalty = self.config.kl_coef * reference_action_log_ratio
            total_reward = float(reward - kl_penalty)
            episode.steps.append(
                PPORolloutStep(
                    task_id=task.task_id,
                    step_index=step_index,
                    prompt=prompt,
                    input_ids=list(generation.input_ids),
                    response_mask=list(generation.response_mask),
                    prompt_length=generation.prompt_length,
                    response_text=generation.text,
                    action=action,
                    old_logprob=float(generation.old_logprob),
                    reference_logprob=reference,
                    value=float(generation.value),
                    environment_reward=float(reward),
                    kl_penalty=float(kl_penalty),
                    reference_action_log_ratio=float(reference_action_log_ratio),
                    total_ppo_reward=total_reward,
                    done=bool(env_done),
                    truncated=collector_truncated,
                    valid_action=valid_action,
                    state=state.to_dict(),
                    next_state=next_state.to_dict(),
                    observation=observation_dict,
                )
            )
            history.append(observation_dict)
            state = next_state
            if env_done:
                episode.bootstrap_value = 0.0
                break
            if collector_truncated:
                _, next_messages = self._messages(state, history)
                episode.bootstrap_value = self.policy.value_for_messages(next_messages)
                break
        return episode
