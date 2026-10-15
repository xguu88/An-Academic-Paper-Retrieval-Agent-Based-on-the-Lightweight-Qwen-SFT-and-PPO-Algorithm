"""从模型文本中提取、构造并严格验证结构化 Action JSON。"""

import json
import re
from typing import Any

from src.agent.actions import ActionType, AgentAction, CitationDirection
from src.agent.exceptions import ActionParseError, InvalidActionError
from src.agent.validator import validate_action, validate_action_payload


_FENCED_JSON = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)


def _decode_json_object(text: str) -> Any:
    """解析完整 JSON，或从少量前后自然语言中提取唯一 JSON 值。将json转变成python对象"""

    stripped = text.strip()
    if not stripped:
        raise ActionParseError("动作文本为空")

    fenced = _FENCED_JSON.findall(stripped)
    if len(fenced) > 1:
        raise ActionParseError("动作文本包含多个 JSON 代码块")
    candidate = fenced[0].strip() if fenced else stripped
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()#创建一个 JSON 解码器对象，用来把JSON 格式字符串 → Python 对象（字典、列表、数字、字符串等）
    decoded: list[Any] = []
    index = 0
    while index < len(candidate):
        character = candidate[index]
        if character not in "[{":
            index += 1
            continue
        try:
            value, end = decoder.raw_decode(candidate, index)
        except json.JSONDecodeError:
            index += 1
            continue
        decoded.append(value)
        index = end
    if len(decoded) != 1:
        raise ActionParseError("无法从文本中唯一解析 Action JSON")
    return decoded[0]


def _build_action(payload: dict[str, Any]) -> AgentAction:
    """将已通过字段集合检查的字典转换为 AgentAction。"""

    try:
        direction = (
            CitationDirection(payload["direction"])
            if "direction" in payload and isinstance(payload["direction"], str)
            else payload.get("direction")
        )
    except ValueError as exc:
        raise InvalidActionError("direction 只能是 references 或 citations") from exc

    return AgentAction(
        action=ActionType(payload["action"]),
        query=payload.get("query"),
        paper_id=payload.get("paper_id"),
        direction=direction,
        candidate_ids=payload.get("candidate_ids", []),
        top_k=payload.get("top_k"),
    )


def parse_action(text: str) -> AgentAction:
    """解析模型文本；语法失败抛 ParseError，语义失败抛 InvalidActionError。"""

    if not isinstance(text, str):
        raise ActionParseError("动作输入必须是字符串")
    payload = _decode_json_object(text)
    validate_action_payload(payload)
    action = _build_action(payload)
    validate_action(action)
    return action
