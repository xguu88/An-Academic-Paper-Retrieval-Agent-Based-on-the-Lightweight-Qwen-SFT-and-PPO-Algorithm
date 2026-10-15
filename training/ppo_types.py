"""PPO 配置、生成结果、rollout transition 与训练统计类型。"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from src.agent.actions import AgentAction


@dataclass(frozen=True, slots=True)
class PPOConfig:
    """集中保存 PPO 超参数；正式入口从 YAML 构造，禁止散落硬编码。"""

    learning_rate: float = 1e-5
    max_steps: int = 6
    rollout_tasks_per_batch: int = 4
    ppo_epochs: int = 4
    minibatch_size: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_clip_range: float | None = 0.2
    policy_coef: float = 1.0
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    use_reference_policy: bool = True
    kl_coef: float = 0.02
    max_grad_norm: float = 1.0
    log_ratio_clamp: float = 20.0
    normalize_advantages: bool = True
    terminate_on_invalid_action: bool = False
    action_logprob_reduction: str = "sum"
    seed: int = 42

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> PPOConfig:
        """从 YAML 顶层或 training 小节读取配置并执行严格校验。
    将 YAML/dict 中的 PPO 参数：
    1. 提取 training 段
    2. 兼容旧字段名
    3. 过滤非 PPOConfig 字段
    4. 构造 PPOConfig 对象
    5. 检查参数是否合法
        """

        values = dict(data.get("training", data))
        aliases = {
            "mini_batch_size": "minibatch_size",
            "value_coefficient": "value_coef",
            "entropy_coefficient": "entropy_coef",
            "rollout_batch_size": "rollout_tasks_per_batch",
        }
        for old, new in aliases.items():
            if old in values and new not in values:
                values[new] = values.pop(old)
        allowed = set(cls.__dataclass_fields__)
        selected = {key: value for key, value in values.items() if key in allowed}
        config = cls(**selected)
        config.validate()
        return config

    def validate(self) -> None:
        """拒绝会产生未定义 PPO 行为的超参数。"""

        positive = (
            "learning_rate",
            "max_steps",
            "rollout_tasks_per_batch",
            "ppo_epochs",
            "minibatch_size",
            "gamma",
            "gae_lambda",
            "clip_range",
            "policy_coef",
            "value_coef",
            "max_grad_norm",
            "log_ratio_clamp",
        )
        if any(float(getattr(self, name)) <= 0 for name in positive):
            raise ValueError("PPO 正值超参数必须大于 0")
        if self.gamma > 1 or self.gae_lambda > 1:
            raise ValueError("gamma 和 gae_lambda 必须位于 (0, 1]")
        if self.entropy_coef < 0 or self.kl_coef < 0:
            raise ValueError("entropy_coef 和 kl_coef 不能为负")
        if self.value_clip_range is not None and self.value_clip_range <= 0:
            raise ValueError("value_clip_range 必须为正数或 None")
        if self.action_logprob_reduction != "sum":
            raise ValueError(
                "STRICT MACRO-ACTION PPO 要求 action_logprob_reduction=sum"
            )

    def to_dict(self) -> dict[str, Any]:
        """转换为 JSON/YAML 可序列化字典。"""

        return asdict(self)


@dataclass(slots=True)
class PolicyGeneration:
    """一次生成；old/reference 是 response_mask 内 token logprob 的严格总和。"""

    text: str
    input_ids: list[int]
    response_mask: list[int]
    old_logprob: float
    value: float
    reference_logprob: float | None = None
    prompt_length: int = 0

    def validate(self) -> None:
        """检查序列、mask、概率与 value 均可用于 PPO。"""

        if not self.input_ids or len(self.input_ids) != len(self.response_mask):
            raise ValueError("input_ids 与 response_mask 必须为等长非空序列")
        if any(value not in {0, 1} for value in self.response_mask):
            raise ValueError("response_mask 只能包含 0/1")
        if not any(self.response_mask):
            raise ValueError("assistant response mask 不能为空")
        if self.prompt_length <= 0 or self.prompt_length >= len(self.input_ids):
            raise ValueError("prompt_length 必须落在完整序列内部")
        numbers = [self.old_logprob, self.value]
        if self.reference_logprob is not None:
            numbers.append(self.reference_logprob)
        if not all(math.isfinite(value) for value in numbers):
            raise ValueError("rollout logprob/value 必须是有限数")


@dataclass(slots=True)
class PPORolloutStep:
    """一个 state→generated action→environment transition。"""

    task_id: str
    step_index: int
    prompt: str
    input_ids: list[int]
    response_mask: list[int]
    prompt_length: int
    response_text: str
    action: AgentAction
    old_logprob: float
    reference_logprob: float | None
    value: float
    environment_reward: float
    kl_penalty: float
    reference_action_log_ratio: float
    total_ppo_reward: float
    done: bool
    truncated: bool
    valid_action: bool
    state: dict[str, Any]
    next_state: dict[str, Any]
    observation: dict[str, Any]
    advantage: float = 0.0
    return_: float = 0.0


@dataclass(slots=True)
class PPORolloutEpisode:
    """一条 episode 及 truncated 尾部的 bootstrap value。"""

    task_id: str
    steps: list[PPORolloutStep] = field(default_factory=list)
    bootstrap_value: float = 0.0


@dataclass(frozen=True, slots=True)
class PPOUpdateMetrics:
    """一次 minibatch update；entropy 是 response raw-token entropy 均值。"""

    total_loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    ppo_log_ratio_mean: float
    ppo_ratio_mean: float
    clip_fraction: float
    gradient_norm: float
