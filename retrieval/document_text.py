"""集中定义各 Retriever 共用的论文基础检索文本。"""

from src.retrieval.paper import PaperDocument


def retrieval_text_fields(paper: PaperDocument) -> dict[str, str]:
    """返回标题、摘要和关键词三个规范文本字段。"""

    return {
        "title": paper.title,
        "abstract": paper.abstract or "",
        "keywords": " ".join(paper.keywords),
    }


def build_retrieval_text(paper: PaperDocument) -> str:
    """按固定顺序拼接基础检索文本，不包含论文全文。"""

    fields = retrieval_text_fields(paper)
    return "\n".join(value for value in fields.values() if value)

