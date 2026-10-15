"""定义不同文献环境可显式启用的动作能力。"""

from dataclasses import asdict, dataclass
from typing import Any

from src.agent.actions import ActionType


@dataclass(frozen=True, slots=True)
class EnvironmentCapabilities:
    """声明环境支持哪些动作，避免用不存在的数据能力冒充可用工具。"""

    rewrite_query: bool = True
    keyword_search: bool = True
    semantic_search: bool = True
    hybrid_search: bool = True
    rerank: bool = False
    expand_citation: bool = False
    read_paper: bool = True
    stop: bool = True

    def supports(self, action_type: ActionType) -> bool:
        """返回给定 ActionType 是否在当前环境启用。"""

        mapping = {
            ActionType.REWRITE_QUERY: self.rewrite_query,
            ActionType.KEYWORD_SEARCH: self.keyword_search,
            ActionType.SEMANTIC_SEARCH: self.semantic_search,
            ActionType.HYBRID_SEARCH: self.hybrid_search,
            ActionType.RERANK: self.rerank,
            ActionType.EXPAND_CITATION: self.expand_citation,
            ActionType.READ_PAPER: self.read_paper,
            ActionType.STOP: self.stop,
        }
        return mapping[action_type]

    def to_dict(self) -> dict[str, Any]:
        """转换为 JSON 兼容字典。"""

        return asdict(self)

    @classmethod
    def scifact(cls) -> "EnvironmentCapabilities":
        """返回不虚构 citation graph、默认禁用 reranker 的 SciFact 能力。"""

        return cls(rerank=False, expand_citation=False)
