"""提供不会自动安装依赖或默认下载模型的 SentenceTransformer 适配层。"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.retrieval.embedding import TextEmbedder


class EmbeddingDependencyError(RuntimeError):
    """当前环境缺少 sentence-transformers 时抛出的错误。"""


class EmbeddingModelUnavailableError(RuntimeError):
    """模型路径不存在且未显式允许远程模型 ID 时抛出的错误。"""


class SentenceTransformerEmbedder(TextEmbedder):
    """延迟加载 SentenceTransformer，并暴露统一 TextEmbedder 接口。"""

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        device: str = "cpu",
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        allow_remote: bool = False,
    ) -> None:
        """加载本地模型；只有 allow_remote=True 才允许未来使用远程模型 ID。"""

        if not str(model_name_or_path).strip():
            raise ValueError("model_name_or_path 不能为空")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size 必须是大于 0 的整数")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingDependencyError(
                "缺少 sentence_transformers；本项目不会自动安装"
            ) from exc

        model_path = Path(model_name_or_path)
        is_local = model_path.exists()
        if not is_local and not allow_remote:
            raise EmbeddingModelUnavailableError(
                "未找到本地 embedding 模型；为防止自动下载，未加载远程模型 ID"
            )

        identifier = str(model_path.resolve()) if is_local else str(model_name_or_path)
        kwargs: dict[str, Any] = {"device": device}
        if is_local:
            kwargs["local_files_only"] = True
        self._model = SentenceTransformer(identifier, **kwargs)
        if hasattr(self._model, "get_embedding_dimension"):
            dimension = self._model.get_embedding_dimension()
        else:
            dimension = self._model.get_sentence_embedding_dimension()
        if not isinstance(dimension, int) or dimension <= 0:
            raise RuntimeError("SentenceTransformer 返回了非法 embedding dimension")
        self._dimension = dimension
        self._model_identifier = identifier
        self._batch_size = batch_size
        self._normalize = normalize_embeddings

    @property
    def dimension(self) -> int:
        """返回模型声明的向量维度。"""

        return self._dimension

    @property
    def model_identifier(self) -> str:
        """返回本地绝对路径或显式远程模型 ID。"""

        return self._model_identifier

    @property
    def normalize_embeddings(self) -> bool:
        """返回 encode 使用的归一化选项。"""

        return self._normalize

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """使用 SentenceTransformer 批量编码，并严格检查输出形状。"""

        if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
            raise TypeError("texts 必须是字符串序列")
        if any(not isinstance(text, str) for text in texts):
            raise TypeError("texts 中的每个元素必须是字符串")
        vectors = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=self._normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        array = np.asarray(vectors, dtype=np.float32)
        expected = (len(texts), self.dimension)
        if array.shape != expected:
            raise RuntimeError(f"embedding shape 非法: expected={expected}, actual={array.shape}")
        return array
