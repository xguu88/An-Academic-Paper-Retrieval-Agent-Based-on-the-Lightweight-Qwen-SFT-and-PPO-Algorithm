"""实现以 paper_id 字典为核心存储的严格论文 JSONL 语料库。"""

import json
from collections.abc import Iterator
from pathlib import Path

from src.retrieval.paper import PaperDocument
from src.retrieval.validation import CorpusValidationError, validate_paper


class PaperCorpus:
    """支持稳定遍历、按 ID 查询与严格加载的内存论文语料库。"""

    def __init__(self, papers: list[PaperDocument]) -> None:
        """从已验证论文建立索引，并再次保护 paper_id 唯一性。"""

        self._papers: dict[str, PaperDocument] = {}
        for position, raw_paper in enumerate(papers, start=1):
            paper = validate_paper(raw_paper, line_number=position)
            if paper.paper_id in self._papers:
                raise CorpusValidationError(
                    "duplicate paper_id",
                    line_number=position,
                    paper_id=paper.paper_id,
                )
            self._papers[paper.paper_id] = paper

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "PaperCorpus":
        """逐行读取 UTF-8 JSONL，所有错误均包含行号与可用论文 ID。"""

        corpus_path = Path(path)
        papers: list[PaperDocument] = []
        seen: set[str] = set()
        with corpus_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise CorpusValidationError("blank line is not allowed", line_number=line_number)
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise CorpusValidationError(
                        f"invalid JSON: {exc.msg}",
                        line_number=line_number,
                    ) from exc
                paper = validate_paper(raw, line_number=line_number)
                if paper.paper_id in seen:
                    raise CorpusValidationError(
                        "duplicate paper_id",
                        line_number=line_number,
                        paper_id=paper.paper_id,
                    )
                seen.add(paper.paper_id)
                papers.append(paper)
        return cls(papers)

    def get(self, paper_id: str) -> PaperDocument | None:
        """按 paper_id 查询，未知 ID 返回 None。"""

        return self._papers.get(paper_id)

    def require(self, paper_id: str) -> PaperDocument:
        """按 paper_id 查询，未知 ID 抛出带 ID 的 KeyError。"""

        try:
            return self._papers[paper_id]
        except KeyError as exc:
            raise KeyError(f"Paper not found: {paper_id}") from exc

    def __len__(self) -> int:
        """返回语料中的论文数量。"""

        return len(self._papers)

    def __iter__(self) -> Iterator[PaperDocument]:
        """按 JSONL 原始顺序遍历论文。"""

        return iter(self._papers.values())

