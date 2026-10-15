"""计算不受 JSONL 行顺序影响的稳定论文语料 SHA256 指纹。"""

import hashlib
import json
from collections.abc import Iterable

from src.retrieval.paper import PaperDocument


def corpus_fingerprint(papers: Iterable[PaperDocument]) -> str:
    """按 paper_id 排序并哈希检索相关字段的稳定 JSON 表示。"""

    records = [
        {
            "paper_id": paper.paper_id,
            "title": paper.title,
            "abstract": paper.abstract,
            "keywords": list(paper.keywords),
        }
        for paper in papers
    ]
    records.sort(key=lambda item: item["paper_id"])
    payload = json.dumps(
        records,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

