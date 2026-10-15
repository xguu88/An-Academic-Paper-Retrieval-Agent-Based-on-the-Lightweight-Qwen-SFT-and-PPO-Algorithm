"""实现可配置字段映射的通用 BEIR-style benchmark 离线转换器。"""

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.benchmarks.base import BenchmarkAdapter
from src.benchmarks.split import split_tasks
from src.benchmarks.types import (
    BenchmarkConversionReport,
    BenchmarkManifest,
    BenchmarkQrel,
    ExternalDocument,
    ExternalQuery,
)
from src.benchmarks.validation import BenchmarkValidationError, validate_benchmark
from src.evaluation.tasks import RetrievalTask
from src.retrieval.fingerprint import corpus_fingerprint
from src.retrieval.paper import PaperDocument


@dataclass(slots=True)
class BEIRFieldMapping:
    """描述外部 corpus/query 字段名，不绑定具体数据集名称。"""

    corpus_id: str = "_id"
    corpus_title: str = "title"
    corpus_text: str = "text"
    corpus_metadata: str = "metadata"
    query_id: str = "_id"
    query_text: str = "text"
    query_type: str = "query_type"
    qrel_query_id: str = "query_id"
    qrel_corpus_id: str = "corpus_id"
    qrel_relevance: str = "relevance"


class BEIRAdapter(BenchmarkAdapter):
    """把 BEIR-style corpus、queries、qrels 转为项目标准 JSONL。"""

    def __init__(
        self,
        dataset_name: str,
        *,
        field_mapping: BEIRFieldMapping | None = None,
        generated_split_seed: int = 42,
    ) -> None:
        """配置数据集显示名称、字段映射和无原 split 时的固定 seed。"""

        if not isinstance(dataset_name, str) or not dataset_name.strip():
            raise ValueError("dataset_name 必须是非空字符串")
        self.dataset_name = dataset_name.strip()
        self.mapping = field_mapping or BEIRFieldMapping()
        self.generated_split_seed = generated_split_seed

    @staticmethod
    def _resolve_file(source_dir: Path, candidates: tuple[str, ...]) -> Path:
        """按候选名称寻找必需输入文件。"""

        for name in candidates:
            path = source_dir / name
            if path.is_file():
                return path
        raise FileNotFoundError(f"找不到输入文件，候选: {', '.join(candidates)}")

    @staticmethod
    def _read_jsonl(path: Path) -> list[Any]:
        """逐行读取 JSONL，并在错误中保留文件名和行号。"""

        records: list[Any] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise BenchmarkValidationError(f"{path.name}:{line_number}: blank line")
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise BenchmarkValidationError(
                        f"{path.name}:{line_number}: invalid JSON: {exc.msg}"
                    ) from exc
        return records

    def _parse_documents(self, path: Path) -> list[ExternalDocument]:
        """按字段映射解析外部 corpus，不虚构作者、年份或 citation。"""

        documents: list[ExternalDocument] = []
        for line_number, raw in enumerate(self._read_jsonl(path), start=1):
            if not isinstance(raw, dict):
                raise BenchmarkValidationError(f"{path.name}:{line_number}: corpus record must be object")
            document_id = raw.get(self.mapping.corpus_id)
            title = raw.get(self.mapping.corpus_title, "")
            text = raw.get(self.mapping.corpus_text, "")
            metadata = raw.get(self.mapping.corpus_metadata, {})
            documents.append(ExternalDocument(document_id, title, text, metadata))
        return documents

    def _parse_queries(self, path: Path) -> list[ExternalQuery]:
        """按字段映射解析 query，并仅保留显式提供的 query_type。"""

        queries: list[ExternalQuery] = []
        for line_number, raw in enumerate(self._read_jsonl(path), start=1):
            if not isinstance(raw, dict):
                raise BenchmarkValidationError(f"{path.name}:{line_number}: query record must be object")
            queries.append(
                ExternalQuery(
                    raw.get(self.mapping.query_id),
                    raw.get(self.mapping.query_text),
                    raw.get(self.mapping.query_type),
                )
            )
        return queries

    @staticmethod
    def _split_from_path(path: Path) -> str | None:
        """仅把 train/dev/test 文件名解释为原始 split。"""

        stem = path.stem.lower()
        return stem if stem in {"train", "dev", "test"} else None

    def _parse_qrels_jsonl(self, path: Path, split: str | None) -> list[BenchmarkQrel]:
        """解析 JSONL qrels，支持可配置字段名和常见 score 别名。"""

        qrels: list[BenchmarkQrel] = []
        for line_number, raw in enumerate(self._read_jsonl(path), start=1):
            if not isinstance(raw, dict):
                raise BenchmarkValidationError(f"{path.name}:{line_number}: qrel must be object")
            query_id = raw.get(self.mapping.qrel_query_id, raw.get("query-id"))
            corpus_id = raw.get(self.mapping.qrel_corpus_id, raw.get("corpus-id"))
            relevance = raw.get(
                self.mapping.qrel_relevance,
                raw.get("score", raw.get("relevance")),
            )
            qrels.append(BenchmarkQrel(query_id, corpus_id, relevance, split))
        return qrels

    def _parse_qrels_tsv(self, path: Path, split: str | None) -> list[BenchmarkQrel]:
        """解析带表头或无表头的三列 BEIR qrels TSV。"""

        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle, delimiter="\t"))
        if not rows:
            raise BenchmarkValidationError(f"{path.name}: qrels file is empty")
        normalized_header = [cell.strip().lower().replace("-", "_") for cell in rows[0]]
        query_names = {self.mapping.qrel_query_id.lower(), "query_id", "queryid"}
        corpus_names = {self.mapping.qrel_corpus_id.lower(), "corpus_id", "doc_id", "docid"}
        relevance_names = {self.mapping.qrel_relevance.lower(), "score", "relevance"}
        has_header = bool(set(normalized_header) & query_names)
        if has_header:
            def find_index(names: set[str]) -> int:
                for index, name in enumerate(normalized_header):
                    if name in names:
                        return index
                raise BenchmarkValidationError(f"{path.name}: qrels header 缺少字段 {sorted(names)}")

            query_index = find_index(query_names)
            corpus_index = find_index(corpus_names)
            relevance_index = find_index(relevance_names)
            data_rows = rows[1:]
        else:
            query_index, corpus_index, relevance_index = 0, 1, 2
            data_rows = rows
        qrels: list[BenchmarkQrel] = []
        for line_number, row in enumerate(data_rows, start=2 if has_header else 1):
            if len(row) <= max(query_index, corpus_index, relevance_index):
                raise BenchmarkValidationError(f"{path.name}:{line_number}: qrel columns missing")
            try:
                relevance = float(row[relevance_index])
            except ValueError as exc:
                raise BenchmarkValidationError(
                    f"{path.name}:{line_number}: relevance is not numeric"
                ) from exc
            qrels.append(
                BenchmarkQrel(row[query_index], row[corpus_index], relevance, split)
            )
        return qrels

    def _qrel_files(self, source_dir: Path) -> list[Path]:
        """发现 qrels/ 目录或根目录下的等价 qrels 文件。"""

        qrels_dir = source_dir / "qrels"
        if qrels_dir.is_dir():
            files = sorted(path for path in qrels_dir.iterdir() if path.suffix.lower() in {".tsv", ".jsonl"})
            if not files:
                raise FileNotFoundError("qrels/ 目录中没有 .tsv 或 .jsonl 文件")
            return files
        for name in ("qrels.tsv", "qrels.jsonl", "qrels"):
            path = source_dir / name
            if path.is_file():
                return [path]
        raise FileNotFoundError("找不到 qrels/、qrels.tsv、qrels.jsonl 或 qrels")

    def _parse_qrels(self, source_dir: Path) -> list[BenchmarkQrel]:
        """解析全部 qrels 文件，并从 train/dev/test 文件名保留 split。"""

        qrels: list[BenchmarkQrel] = []
        for path in self._qrel_files(source_dir):
            split = self._split_from_path(path)
            if path.suffix.lower() == ".jsonl":
                qrels.extend(self._parse_qrels_jsonl(path, split))
            else:
                qrels.extend(self._parse_qrels_tsv(path, split))
        return qrels

    @staticmethod
    def _grade_key(grade: float) -> str:
        """把整数 grade 输出为简洁字符串，否则保留浮点表示。"""

        return str(int(grade)) if float(grade).is_integer() else str(grade)

    @staticmethod
    def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
        """以 UTF-8 稳定写入项目标准 JSONL。"""

        content = "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records
        )
        path.write_text(content, encoding="utf-8")

    def convert(self, source_dir: Path, output_dir: Path) -> BenchmarkConversionReport:
        """转换 BEIR-style 数据，并生成 papers、split tasks 与 manifest。"""

        source_dir = Path(source_dir)
        output_dir = Path(output_dir)
        corpus_path = self._resolve_file(source_dir, ("corpus.jsonl", "corpus"))
        queries_path = self._resolve_file(source_dir, ("queries.jsonl", "queries"))
        documents = self._parse_documents(corpus_path)
        queries = self._parse_queries(queries_path)
        qrels = self._parse_qrels(source_dir)
        validate_benchmark(documents, queries, qrels)

        papers = [
            PaperDocument(
                paper_id=document.document_id,
                title=document.title,
                abstract=document.text,
                metadata=dict(document.metadata),
            )
            for document in documents
        ]
        qrels_by_query: dict[str, dict[str, float]] = defaultdict(dict)
        split_by_query: dict[str, str] = {}
        for qrel in qrels:
            qrels_by_query[qrel.query_id][qrel.corpus_id] = float(qrel.relevance)
            if qrel.split is not None:
                previous = split_by_query.setdefault(qrel.query_id, qrel.split)
                if previous != qrel.split:
                    raise BenchmarkValidationError(
                        f"query {qrel.query_id!r} appears in multiple qrel splits"
                    )
        tasks = []
        for query in queries:
            grades = qrels_by_query[query.query_id]
            relevant_ids = [paper_id for paper_id, grade in grades.items() if grade > 0]
            tasks.append(
                RetrievalTask(
                    task_id=query.query_id,
                    question=query.text,
                    relevant_paper_ids=relevant_ids,
                    relevance_grades=grades,
                    query_type=query.query_type,
                )
            )
        if split_by_query and len(split_by_query) != len(tasks):
            raise BenchmarkValidationError("qrels split 信息不完整，不能混合 original/generated split")
        split_result = split_tasks(
            tasks,
            explicit_splits=split_by_query or None,
            seed=self.generated_split_seed,
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        self._write_jsonl(output_dir / "papers.jsonl", [paper.to_dict() for paper in papers])
        for split_name in ("train", "dev", "test"):
            split_tasks_list = getattr(split_result, split_name)
            self._write_jsonl(
                output_dir / f"{split_name}_tasks.jsonl",
                [task.to_dict() for task in split_tasks_list],
            )
        distribution = Counter(self._grade_key(float(qrel.relevance)) for qrel in qrels)
        report = BenchmarkConversionReport(
            num_documents=len(documents),
            num_queries=len(queries),
            num_qrels=len(qrels),
            relevance_distribution=dict(sorted(distribution.items())),
        )
        manifest = BenchmarkManifest(
            dataset_name=self.dataset_name,
            source_format="beir-style",
            num_documents=report.num_documents,
            num_queries=report.num_queries,
            num_qrels=report.num_qrels,
            converted_at=datetime.now(timezone.utc).isoformat(),
            corpus_fingerprint=corpus_fingerprint(papers),
            split_source=split_result.source,
            split_seed=split_result.seed,
            splits=split_result.counts(),
        )
        (output_dir / "benchmark_manifest.json").write_text(
            json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output_dir / "conversion_report.json").write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return report
