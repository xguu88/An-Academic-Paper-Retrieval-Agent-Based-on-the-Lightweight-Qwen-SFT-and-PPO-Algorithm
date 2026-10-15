"""提供无需外部资源的 Unicode 规范化轻量检索 tokenizer。"""

import re
import unicodedata


_TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """执行 NFKC、大小写归一化，并保留 Unicode 字母与数字 token。"""

    if not isinstance(text, str):
        raise TypeError("text 必须是字符串")
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return [token for token in _TOKEN_PATTERN.findall(normalized) if token]

