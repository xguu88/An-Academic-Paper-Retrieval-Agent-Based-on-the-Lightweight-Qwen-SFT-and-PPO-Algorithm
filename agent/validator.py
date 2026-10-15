"""实现与 AgentAction 数据载体解耦的严格业务合法性校验。"""

from typing import Any

from src.agent.actions import ActionType, AgentAction, CitationDirection
from src.agent.exceptions import InvalidActionError, MissingActionArgumentError


#约束每个变量传入参数的格式
ACTION_FIELDS: dict[ActionType, tuple[set[str], set[str]]] = {
    ActionType.REWRITE_QUERY: ({"query"}, set()),#第一个参数必传，第二个可选
    ActionType.KEYWORD_SEARCH: ({"query"}, {"top_k"}),#两个参数都必传
    ActionType.SEMANTIC_SEARCH: ({"query"}, {"top_k"}),
    ActionType.HYBRID_SEARCH: ({"query"}, {"top_k"}),
    ActionType.RERANK: ({"candidate_ids"}, {"top_k"}),
    ActionType.EXPAND_CITATION: ({"paper_id", "direction"}, set()),
    ActionType.READ_PAPER: ({"paper_id"}, set()),
    ActionType.STOP: (set(), {"candidate_ids"}),
}


def validate_action_payload(payload: Any) -> None:
    """在构造数据类前校验 JSON 根类型、字段集合和必填字段。"""

    if not isinstance(payload, dict):
        raise InvalidActionError("Action JSON 根节点必须是对象")
    if "action" not in payload:
        raise MissingActionArgumentError("缺少必填字段: action")
    raw_action = payload["action"]
    if not isinstance(raw_action, str):
        raise InvalidActionError("action 必须是字符串")
    try:
        action_type = ActionType(raw_action)
    except ValueError as exc:
        raise InvalidActionError(f"不存在的 action: {raw_action!r}") from exc

    required, optional = ACTION_FIELDS[action_type]
    supplied = set(payload) - {"action"}
    unknown = supplied - required - optional
    if unknown:
        names = ", ".join(sorted(unknown))
        raise InvalidActionError(f"{action_type.value} 不允许字段: {names}")
    missing = required - supplied
    if missing:
        names = ", ".join(sorted(missing))
        raise MissingActionArgumentError(f"{action_type.value} 缺少必填字段: {names}")


def validate_action(action: AgentAction) -> None:
    """严格校验每种动作的数据类型、非空约束和无关字段。"""

    if not isinstance(action, AgentAction):
        raise InvalidActionError("待校验对象必须是 AgentAction")
    if not isinstance(action.action, ActionType):
        raise InvalidActionError("action 必须是 ActionType")

    used_fields = {
        "query": action.query is not None,
        "paper_id": action.paper_id is not None,
        "direction": action.direction is not None,
        "candidate_ids": bool(action.candidate_ids),
        "top_k": action.top_k is not None,
    }
    required, optional = ACTION_FIELDS[action.action]
    illegal = {name for name, used in used_fields.items() if used and name not in required | optional}
    if illegal:
        names = ", ".join(sorted(illegal))
        raise InvalidActionError(f"{action.action.value} 不允许参数: {names}")

    if "query" in required:
        if action.query is None:
            raise MissingActionArgumentError(f"{action.action.value} 缺少必填字段: query")
        if not isinstance(action.query, str) or not action.query.strip():
            raise InvalidActionError("query 必须是非空字符串")

    if "paper_id" in required:
        if action.paper_id is None:
            raise MissingActionArgumentError(f"{action.action.value} 缺少必填字段: paper_id")
        if not isinstance(action.paper_id, str) or not action.paper_id.strip():
            raise InvalidActionError("paper_id 必须是非空字符串")

    if action.action is ActionType.EXPAND_CITATION:
        if action.direction is None:
            raise MissingActionArgumentError("ExpandCitation 缺少必填字段: direction")
        if not isinstance(action.direction, CitationDirection):
            raise InvalidActionError("direction 只能是 references 或 citations")

    if not isinstance(action.candidate_ids, list):
        raise InvalidActionError("candidate_ids 必须是字符串列表")
    if action.action is ActionType.RERANK and not action.candidate_ids:
        raise InvalidActionError("Rerank 的 candidate_ids 不能为空")
    if action.candidate_ids:
        if any(not isinstance(item, str) or not item.strip() for item in action.candidate_ids):
            raise InvalidActionError("candidate_ids 必须是非空字符串列表")

    if action.top_k is not None:
        if isinstance(action.top_k, bool) or not isinstance(action.top_k, int):
            raise InvalidActionError("top_k 必须是整数")
        if action.top_k <= 0:
            raise InvalidActionError("top_k 必须大于 0")
