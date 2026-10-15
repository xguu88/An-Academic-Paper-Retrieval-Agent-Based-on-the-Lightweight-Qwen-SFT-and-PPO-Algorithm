"""定义结构化动作协议使用的项目自定义异常。"""


class ActionProtocolError(ValueError):
    """所有动作协议错误的基类。"""


class ActionParseError(ActionProtocolError):
    """输入中不存在可唯一解析的 JSON 对象。"""


class InvalidActionError(ActionProtocolError):
    """JSON 可以解析，但动作名称、字段或参数值非法。"""


class MissingActionArgumentError(InvalidActionError):
    """动作缺少其协议规定的必填参数。"""

