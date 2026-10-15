"""定义离线论文语料使用的统一、轻量且可序列化的数据结构。"""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class PaperDocument:
    """一篇论文的元数据；未知标量使用 None，引用关系保存 paper_id。"""

    paper_id: str
    title: str
    abstract: str | None = None
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    keywords: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转换为可直接写入 JSONL 的纯字典。"""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PaperDocument":
        """从论文 JSON 对象恢复数据类。"""

        return cls(**data)

