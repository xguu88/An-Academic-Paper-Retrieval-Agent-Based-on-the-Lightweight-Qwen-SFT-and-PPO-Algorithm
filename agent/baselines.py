"""提供不依赖 Qwen 的确定性规则策略，用于验证环境整条轨迹。"""

from abc import ABC, abstractmethod

from src.agent.actions import ActionType, AgentAction
from src.agent.trajectory import AgentTrajectory
from src.environment.literature_env import LiteratureEnvironment
from src.evaluation.tasks import RetrievalTask


class RuleBasedPolicy(ABC):
    """按固定动作序列运行 LiteratureEnvironment 的基线策略。"""

    name: str

    @abstractmethod
    def actions(self, task: RetrievalTask, top_k: int) -> list[AgentAction]:
        """为任务生成不含模型推理的确定性动作序列。"""

    def run(
        self,
        environment: LiteratureEnvironment,
        task: RetrievalTask,
        *,
        top_k: int = 20,
    ) -> AgentTrajectory:
        """reset 后依次 step，直到 Stop 或预算终止，并返回完整轨迹。"""

        environment.reset(task)
        for action in self.actions(task, top_k):
            _, _, _, done, _ = environment.step(action)
            if done:
                break
        if environment.trajectory is None:
            raise RuntimeError("environment 未生成 trajectory")
        return environment.trajectory


class KeywordOnlyPolicy(RuleBasedPolicy):
    """BM25 KeywordSearch 后立即 Stop。"""

    name = "KeywordOnly"

    def actions(self, task: RetrievalTask, top_k: int) -> list[AgentAction]:
        """生成 KeywordSearch → Stop。"""

        return [
            AgentAction(ActionType.KEYWORD_SEARCH, query=task.question, top_k=top_k),
            AgentAction(ActionType.STOP),
        ]


class SemanticOnlyPolicy(RuleBasedPolicy):
    """Dense SemanticSearch 后立即 Stop。"""

    name = "SemanticOnly"

    def actions(self, task: RetrievalTask, top_k: int) -> list[AgentAction]:
        """生成 SemanticSearch → Stop。"""

        return [
            AgentAction(ActionType.SEMANTIC_SEARCH, query=task.question, top_k=top_k),
            AgentAction(ActionType.STOP),
        ]


class HybridOnlyPolicy(RuleBasedPolicy):
    """执行真实 RRF HybridSearch 后立即 Stop。"""

    name = "HybridOnly"

    def actions(self, task: RetrievalTask, top_k: int) -> list[AgentAction]:
        """生成 HybridSearch → Stop。"""

        return [
            AgentAction(ActionType.HYBRID_SEARCH, query=task.question, top_k=top_k),
            AgentAction(ActionType.STOP),
        ]


class KeywordThenSemanticPolicy(RuleBasedPolicy):
    """依次执行 BM25 与 Dense，不冒充 RRF Hybrid。"""

    name = "KeywordThenSemantic"

    def actions(self, task: RetrievalTask, top_k: int) -> list[AgentAction]:
        """生成 KeywordSearch → SemanticSearch → Stop。"""

        return [
            AgentAction(ActionType.KEYWORD_SEARCH, query=task.question, top_k=top_k),
            AgentAction(ActionType.SEMANTIC_SEARCH, query=task.question, top_k=top_k),
            AgentAction(ActionType.STOP),
        ]


class SemanticThenKeywordPolicy(RuleBasedPolicy):
    """先 Dense 后 BM25，用于离线 Expert 搜索空间而非默认 dev baseline。"""

    name = "SemanticThenKeyword"

    def actions(self, task: RetrievalTask, top_k: int) -> list[AgentAction]:
        """生成 SemanticSearch → KeywordSearch → Stop。"""

        return [
            AgentAction(ActionType.SEMANTIC_SEARCH, query=task.question, top_k=top_k),
            AgentAction(ActionType.KEYWORD_SEARCH, query=task.question, top_k=top_k),
            AgentAction(ActionType.STOP),
        ]
