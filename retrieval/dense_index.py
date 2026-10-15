"""实现 numpy Dense 索引的离线保存、加载和一致性校验。"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


class DenseIndexValidationError(ValueError):
    """索引文件损坏、过期或与当前语料/模型不匹配。"""


@dataclass(slots=True)
class DenseIndexMetadata:
    """保存复用 Dense 向量前必须匹配的索引元数据。"""

    embedding_model: str
    dimension: int
    normalize_embeddings: bool
    num_documents: int
    corpus_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        """转换为稳定 JSON 字典。"""

        return asdict(self)


@dataclass(slots=True)
class DenseIndexBundle:
    """加载后的文档向量、论文顺序和索引元数据。"""

    embeddings: np.ndarray
    paper_ids: list[str]
    metadata: DenseIndexMetadata


def _validate_index_components(
    embeddings: np.ndarray,
    paper_ids: list[str],
    metadata: DenseIndexMetadata,
) -> None:
    """校验数组形状、ID 唯一性和 metadata 内部一致性。"""

    if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2:
        raise DenseIndexValidationError("embeddings 必须是二维 numpy.ndarray")
    if not np.issubdtype(embeddings.dtype, np.number):
        raise DenseIndexValidationError("embeddings 必须是数值数组")
    if not np.all(np.isfinite(embeddings)):
        raise DenseIndexValidationError("embeddings 包含 NaN 或 Infinity")
    if not isinstance(paper_ids, list) or any(
        not isinstance(paper_id, str) or not paper_id.strip() for paper_id in paper_ids
    ):
        raise DenseIndexValidationError("paper_ids 必须是非空字符串列表")
    if len(set(paper_ids)) != len(paper_ids):
        raise DenseIndexValidationError("paper_ids 包含重复 ID")
    if embeddings.shape[0] != len(paper_ids):
        raise DenseIndexValidationError("embedding rows 与 paper_ids 数量不一致")
    if embeddings.shape[1] != metadata.dimension:
        raise DenseIndexValidationError("embedding dimension 与 metadata 不一致")
    if len(paper_ids) != metadata.num_documents:
        raise DenseIndexValidationError("paper_ids 数量与 metadata.num_documents 不一致")
    if (
        isinstance(metadata.dimension, bool)
        or not isinstance(metadata.dimension, int)
        or metadata.dimension <= 0
        or isinstance(metadata.num_documents, bool)
        or not isinstance(metadata.num_documents, int)
        or metadata.num_documents < 0
    ):
        raise DenseIndexValidationError("metadata 中的维度或文档数量非法")
    if not isinstance(metadata.normalize_embeddings, bool):
        raise DenseIndexValidationError("metadata.normalize_embeddings 必须是布尔值")
    if (
        not isinstance(metadata.embedding_model, str)
        or not metadata.embedding_model.strip()
        or not isinstance(metadata.corpus_fingerprint, str)
        or not metadata.corpus_fingerprint.strip()
    ):
        raise DenseIndexValidationError("metadata 缺少模型标识或 corpus fingerprint")


def save_dense_index(
    index_dir: str | Path,
    embeddings: np.ndarray,
    paper_ids: list[str],
    metadata: DenseIndexMetadata,
) -> None:
    """保存 embeddings.npy、paper_ids.json 和 metadata.json。"""

    array = np.asarray(embeddings)
    _validate_index_components(array, paper_ids, metadata)
    directory = Path(index_dir)
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "embeddings.npy", array, allow_pickle=False)
    (directory / "paper_ids.json").write_text(
        json.dumps(paper_ids, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (directory / "metadata.json").write_text(
        json.dumps(metadata.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_dense_index(
    index_dir: str | Path,
    *,
    expected_corpus_fingerprint: str,
    expected_embedding_model: str,
    expected_dimension: int | None = None,
) -> DenseIndexBundle:
    """加载索引并拒绝损坏、语料过期或模型标识不匹配。"""

    directory = Path(index_dir)
    required = ["embeddings.npy", "paper_ids.json", "metadata.json"]
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise DenseIndexValidationError(f"Dense index 缺少文件: {', '.join(missing)}")
    try:
        embeddings = np.load(directory / "embeddings.npy", allow_pickle=False)
        paper_ids_raw = json.loads((directory / "paper_ids.json").read_text(encoding="utf-8"))
        metadata_raw = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        metadata = DenseIndexMetadata(**metadata_raw)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise DenseIndexValidationError(f"Dense index 无法解析: {exc}") from exc
    if not isinstance(paper_ids_raw, list):
        raise DenseIndexValidationError("paper_ids.json 根节点必须是列表")
    paper_ids = list(paper_ids_raw)
    _validate_index_components(embeddings, paper_ids, metadata)
    if metadata.corpus_fingerprint != expected_corpus_fingerprint:
        raise DenseIndexValidationError("corpus fingerprint 不匹配，拒绝使用过期索引")
    if metadata.embedding_model != expected_embedding_model:
        raise DenseIndexValidationError("embedding model identifier 不匹配")
    if expected_dimension is not None and metadata.dimension != expected_dimension:
        raise DenseIndexValidationError("expected embedding dimension 不匹配")
    return DenseIndexBundle(
        embeddings=np.asarray(embeddings, dtype=np.float32),
        paper_ids=paper_ids,
        metadata=metadata,
    )
