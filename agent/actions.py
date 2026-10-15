"""定义 Agent 的结构化动作协议及其可序列化数据结构。"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ActionType(str, Enum):
    """文献检索 Agent 支持的八种动作类型。"""

    REWRITE_QUERY = "RewriteQuery"
    KEYWORD_SEARCH = "KeywordSearch"
    SEMANTIC_SEARCH = "SemanticSearch"
    HYBRID_SEARCH = "HybridSearch"
    RERANK = "Rerank"
    EXPAND_CITATION = "ExpandCitation"
    READ_PAPER = "ReadPaper"
    STOP = "Stop"


class CitationDirection(str, Enum):
    """引用图扩展方向。"""

    REFERENCES = "references"
    CITATIONS = "citations"


@dataclass(slots=True)
class AgentAction:
    """统一动作载体；字段是否可用由独立 validator 按动作类型判断。"""

    action: ActionType
    query: str | None = None
    paper_id: str | None = None
    direction: CitationDirection | None = None
    candidate_ids: list[str] = field(default_factory=list)
    top_k: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """按动作协议输出字典，只包含该动作实际使用的字段。"""

        data: dict[str, Any] = {"action": self.action.value}
        if self.query is not None:
            data["query"] = self.query
        if self.paper_id is not None:
            data["paper_id"] = self.paper_id
        if self.direction is not None:
            data["direction"] = self.direction.value
        if self.candidate_ids:
            data["candidate_ids"] = list(self.candidate_ids)
        elif self.action is ActionType.STOP:
            data["candidate_ids"] = []
        if self.top_k is not None:
            data["top_k"] = self.top_k
        return data
