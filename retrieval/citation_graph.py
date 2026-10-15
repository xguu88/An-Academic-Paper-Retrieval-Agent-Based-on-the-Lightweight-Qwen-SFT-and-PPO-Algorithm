"""实现纯内存论文引用图，明确 source 引用 target 的有向关系。"""

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any

from src.agent.actions import CitationDirection
from src.retrieval.paper import PaperDocument


class CitationGraph:
    """维护正向 references 与反向 citations 邻接表。"""

    def __init__(self) -> None:
        """创建空论文表和双向邻接表。"""

        self._papers: dict[str, PaperDocument] = {}
        self._references: dict[str, set[str]] = defaultdict(set)
        self._citations: dict[str, set[str]] = defaultdict(set)

    def add_paper(self, paper: PaperDocument, *, include_relations: bool = True) -> None:
        """加入论文；可选择是否同步其 references 与 citations 声明。"""

        if not isinstance(paper, PaperDocument):
            raise TypeError("paper 必须是 PaperDocument")
        if not paper.paper_id.strip():
            raise ValueError("paper_id 不能为空")
        self._papers[paper.paper_id] = paper
        if not include_relations:
            return
        for target_id in paper.references:
            self.add_edge(paper.paper_id, target_id)
        for source_id in paper.citations:
            self.add_edge(source_id, paper.paper_id)

    def add_edge(self, source_id: str, target_id: str) -> None:
        """加入 source -> target，含义是 source 论文引用了 target 论文。"""

        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError("source_id 必须是非空字符串")
        if not isinstance(target_id, str) or not target_id.strip():
            raise ValueError("target_id 必须是非空字符串")
        self._references[source_id].add(target_id)
        self._citations[target_id].add(source_id)

    def get_references(self, paper_id: str) -> list[str]:
        """返回该论文引用的目标论文 ID，按 ID 排序以保证结果稳定。"""

        return sorted(self._references.get(paper_id, set()))

    def get_citations(self, paper_id: str) -> list[str]:
        """返回引用该论文的来源论文 ID，按 ID 排序以保证结果稳定。"""

        return sorted(self._citations.get(paper_id, set()))

    def expand(
        self,
        paper_id: str,
        direction: CitationDirection | str,
        max_results: int,
    ) -> list[str]:
        """按 references 或 citations 方向展开一跳邻居。"""

        if isinstance(max_results, bool) or not isinstance(max_results, int) or max_results <= 0:
            raise ValueError("max_results 必须是大于 0 的整数")
        try:
            normalized = CitationDirection(direction)
        except ValueError as exc:
            raise ValueError("direction 只能是 references 或 citations") from exc
        neighbors = (
            self.get_references(paper_id)
            if normalized is CitationDirection.REFERENCES
            else self.get_citations(paper_id)
        )
        return neighbors[:max_results]

    def get_paper(self, paper_id: str) -> PaperDocument | None:
        """返回已登记论文；仅有边但没有元数据时返回 None。"""

        return self._papers.get(paper_id)


@dataclass(slots=True)
class CitationGraphStats:
    """记录基于 references 构建引用图时的离线质量统计。"""

    num_papers: int
    num_edges: int
    num_missing_targets: int
    missing_target_examples: list[str]
    self_citations: int
    duplicate_edges: int

    def to_dict(self) -> dict[str, Any]:
        """转换为可直接写入 JSON 报告的字典。"""

        return asdict(self)


def build_citation_graph(
    papers: list[PaperDocument],
    *,
    max_missing_examples: int = 20,
) -> tuple[CitationGraph, CitationGraphStats]:
    """只根据 PaperDocument.references 构图，并统计缺失目标等问题。"""

    if max_missing_examples < 0:
        raise ValueError("max_missing_examples 不能小于 0")
    graph = CitationGraph()
    corpus_ids = {paper.paper_id for paper in papers}
    for paper in papers:
        graph.add_paper(paper, include_relations=False)

    unique_edges: set[tuple[str, str]] = set()
    missing_edges: set[tuple[str, str]] = set()
    duplicate_edges = 0
    self_edges: set[tuple[str, str]] = set()
    for paper in papers:
        for target_id in paper.references:
            edge = (paper.paper_id, target_id)
            if edge in unique_edges:
                duplicate_edges += 1
                continue
            unique_edges.add(edge)
            graph.add_edge(*edge)
            if paper.paper_id == target_id:
                self_edges.add(edge)
            if target_id not in corpus_ids:
                missing_edges.add(edge)

    examples = [
        f"{source_id}->{target_id}"
        for source_id, target_id in sorted(missing_edges)[:max_missing_examples]
    ]
    stats = CitationGraphStats(
        num_papers=len(papers),
        num_edges=len(unique_edges),
        num_missing_targets=len(missing_edges),
        missing_target_examples=examples,
        self_citations=len(self_edges),
        duplicate_edges=duplicate_edges,
    )
    return graph, stats

