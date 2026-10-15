"""定义 Agent 在一次检索轨迹中的纯数据、可序列化状态。"""

from dataclasses import asdict, dataclass, field
from typing import Any

from src.agent.actions import AgentAction


@dataclass(slots=True)
class AgentState:
    """只保存轨迹状态，不持有模型、Retriever 或训练框架对象。"""

    task_id: str
    original_query: str
    current_query: str
    step: int = 0
    max_steps: int = 20
    remaining_budget: float = 0.0
    action_history: list[AgentAction] = field(default_factory=list)
    retrieved_paper_ids: list[str] = field(default_factory=list)
    candidate_paper_ids: list[str] = field(default_factory=list)
    read_paper_ids: list[str] = field(default_factory=list)
    selected_paper_ids: list[str] = field(default_factory=list)
    seen_queries: list[str] = field(default_factory=list)
    search_count: int = 0
    read_count: int = 0
    terminated: bool = False
    termination_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def remaining_steps(self) -> int:
        """返回剩余动作步数，结果不会小于零。"""

        return max(0, self.max_steps - self.step)

    def to_dict(self) -> dict[str, Any]:
        """转换为不包含对象引用的 JSON 兼容字典。"""

        data = asdict(self)
        data["action_history"] = [action.to_dict() for action in self.action_history]
        return data
