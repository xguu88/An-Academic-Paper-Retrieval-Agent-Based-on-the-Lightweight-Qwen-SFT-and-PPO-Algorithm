"""提供论文 JSONL 的严格字段校验与带上下文错误信息。"""

from collections.abc import Iterable, Mapping
from typing import Any

from src.retrieval.paper import PaperDocument


PAPER_FIELDS = {
    "paper_id",
    "title",
    "abstract",
    "authors",
    "year",
    "venue",
    "doi",
    "arxiv_id",
    "keywords",
    "references",
    "citations",
    "metadata",
}


class CorpusValidationError(ValueError):
    """携带 JSONL 行号、可用 paper_id 和具体原因的语料错误。"""

    def __init__(
        self,
        reason: str,
        *,
        line_number: int | None = None,
        paper_id: str | None = None,
    ) -> None:
        """构造便于命令行定位坏数据的错误消息。"""

        self.reason = reason
        self.line_number = line_number
        self.paper_id = paper_id
        location = f" at line {line_number}" if line_number is not None else ""
        identity = paper_id if paper_id is not None else "<unknown>"
        super().__init__(f"Invalid paper{location}: paper_id={identity}; reason={reason}")


def _validate_string_list(value: Any, field_name: str) -> None:
    """检查值为元素均非空的 list[str]，不接受元组等替代类型。"""

    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"{field_name} must be list[str]")


def validate_paper(
    paper: Mapping[str, Any] | PaperDocument,
    *,
    line_number: int | None = None,
) -> PaperDocument:
    """严格校验一篇论文并返回独立 PaperDocument，不静默转换类型。"""

    if isinstance(paper, PaperDocument):
        data = paper.to_dict()
    elif isinstance(paper, Mapping):
        data = dict(paper)
    else:
        raise CorpusValidationError(
            "JSON value must be an object",
            line_number=line_number,
        )

    raw_id = data.get("paper_id")
    context_id = raw_id if isinstance(raw_id, str) else None
    try:
        unknown = set(data) - PAPER_FIELDS
        if unknown:
            raise ValueError(f"unknown fields: {', '.join(sorted(unknown))}")
        if "paper_id" not in data:
            raise ValueError("paper_id is required")
        if not isinstance(raw_id, str) or not raw_id.strip():
            raise ValueError("paper_id must be a non-empty string")
        title = data.get("title")
        if not isinstance(title, str):
            raise ValueError("title must be a string")
        abstract = data.get("abstract")
        if not title.strip() and (not isinstance(abstract, str) or not abstract.strip()):
            raise ValueError("title and abstract cannot both be empty")

        for field_name in ("authors", "keywords", "references", "citations"):
            _validate_string_list(data.get(field_name, []), field_name)

        year = data.get("year")
        if year is not None and (isinstance(year, bool) or not isinstance(year, int)):
            raise ValueError("year must be int or null")
        for field_name in ("abstract", "venue", "doi", "arxiv_id"):
            value = data.get(field_name)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{field_name} must be string or null")
        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be an object")
    except ValueError as exc:
        raise CorpusValidationError(
            str(exc),
            line_number=line_number,
            paper_id=context_id,
        ) from exc

    normalized = {
        "paper_id": data["paper_id"],
        "title": data["title"],
        "abstract": data.get("abstract"),
        "authors": list(data.get("authors", [])),
        "year": data.get("year"),
        "venue": data.get("venue"),
        "doi": data.get("doi"),
        "arxiv_id": data.get("arxiv_id"),
        "keywords": list(data.get("keywords", [])),
        "references": list(data.get("references", [])),
        "citations": list(data.get("citations", [])),
        "metadata": dict(data.get("metadata", {})),
    }
    return PaperDocument(**normalized)


def validate_corpus(papers: Iterable[PaperDocument]) -> None:
    """校验论文集合并检测重复 paper_id。"""

    seen: set[str] = set()
    for position, paper in enumerate(papers, start=1):
        validated = validate_paper(paper, line_number=position)
        if validated.paper_id in seen:
            raise CorpusValidationError(
                "duplicate paper_id",
                line_number=position,
                paper_id=validated.paper_id,
            )
        seen.add(validated.paper_id)
