"""定义训练与推理都可复用的 Agent 轨迹数据结构。"""

from dataclasses import dataclass, field
from typing import Any

from src.agent.actions import AgentAction
from src.agent.state import AgentState


@dataclass(slots=True)
class TrajectoryStep:
    """记录一次状态转移；环境尚未评分时 reward 可以为 None。"""

    step: int
    state_before: AgentState
    action: AgentAction
    observation: dict[str, Any]
    state_after: AgentState
    reward: float | None = None
    reward_components: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转换为可保存为 JSON 的纯字典。"""

        return {
            "step": self.step,
            "state_before": self.state_before.to_dict(),
            "action": self.action.to_dict(),
            "observation": dict(self.observation),
            "reward": self.reward,
            "reward_components": dict(self.reward_components),
            "state_after": self.state_after.to_dict(),
        }


@dataclass(slots=True)
class AgentTrajectory:
    """记录一个任务的完整动作轨迹和可选评估结果。"""

    task_id: str
    steps: list[TrajectoryStep] = field(default_factory=list)
    total_reward: float | None = None
    success: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        """转换为可保存为 JSON 的纯字典。"""

        return {
            "task_id": self.task_id,
            "steps": [step.to_dict() for step in self.steps],
            "total_reward": self.total_reward,
            "success": self.success,
        }
