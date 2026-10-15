"""定义未来 Qwen 可复用、按 Environment capability 动态生成的系统指令。"""

import json

from src.agent.actions import ActionType
from src.environment.capabilities import EnvironmentCapabilities


SYSTEM_PROMPT_VERSION = "scifact-agent-system-v1"


_ACTION_SCHEMAS: dict[ActionType, dict[str, object]] = {
    ActionType.REWRITE_QUERY: {"action": "RewriteQuery", "query": "string"},
    ActionType.KEYWORD_SEARCH: {
        "action": "KeywordSearch",
        "query": "string",
        "top_k": "optional positive integer",
    },
    ActionType.SEMANTIC_SEARCH: {
        "action": "SemanticSearch",
        "query": "string",
        "top_k": "optional positive integer",
    },
    ActionType.HYBRID_SEARCH: {
        "action": "HybridSearch",
        "query": "string",
        "top_k": "optional positive integer",
    },
    ActionType.RERANK: {
        "action": "Rerank",
        "candidate_ids": ["paper_id"],
        "top_k": "optional positive integer",
    },
    ActionType.EXPAND_CITATION: {
        "action": "ExpandCitation",
        "paper_id": "string",
        "direction": "references or citations",
    },
    ActionType.READ_PAPER: {"action": "ReadPaper", "paper_id": "string"},
    ActionType.STOP: {"action": "Stop", "candidate_ids": "optional list[paper_id]"},
}


def build_system_prompt(capabilities: EnvironmentCapabilities) -> str:
    """仅列出当前能力允许的 Action，并要求输出唯一严格 JSON。"""

    available = [action for action in ActionType if capabilities.supports(action)]
    schemas = [json.dumps(_ACTION_SCHEMAS[action], ensure_ascii=False) for action in available]
    return "\n".join(
        [
            "你是一个学术文献检索 Agent。",
            "你必须只输出一个合法 JSON Action，不得输出解释、Markdown 或推理过程。",
            "只能从下面当前可用的 Action schema 中选择：",
            *[f"- {schema}" for schema in schemas],
        ]
    )

