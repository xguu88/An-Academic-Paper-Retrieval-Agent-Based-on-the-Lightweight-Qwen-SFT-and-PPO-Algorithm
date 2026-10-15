"""提供严格本地加载的 sentence-transformers CrossEncoder PairScorer。"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from src.retrieval.pair_scorer import PairScorer


class SentenceTransformerCrossEncoderScorer(PairScorer):
    """封装本地 CrossEncoder；默认且当前实现均禁止远程模型下载。"""

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        device: str = "cpu",
        batch_size: int = 16,
        allow_remote: bool = False,
    ) -> None:
        """检查本地目录后以 local_files_only=True 加载模型。"""

        if allow_remote:
            raise ValueError("本项目禁止 CrossEncoder 自动远程下载")
        model_path = Path(model_name_or_path).expanduser()
        if not model_path.is_dir():
            raise FileNotFoundError(f"本地 CrossEncoder 模型目录不存在: {model_path}")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size 必须是大于 0 的整数")
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(
            str(model_path),
            device=device,
            local_files_only=True,
        )
        self.model_name_or_path = str(model_path.resolve())
        self.device = device
        self.batch_size = batch_size
        self.allow_remote = False

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        """批量预测相关性分数，并规范为一维 float32 数组。"""

        if isinstance(pairs, (str, bytes)) or not isinstance(pairs, Sequence):
            raise TypeError("pairs 必须是 (query, document) 序列")
        normalized: list[tuple[str, str]] = []
        for pair in pairs:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise TypeError("每个 pair 必须是长度为 2 的 tuple")
            query, document = pair
            if not isinstance(query, str) or not isinstance(document, str):
                raise TypeError("pair 中 query 和 document 必须是字符串")
            normalized.append((query, document))
        if not normalized:
            return np.empty((0,), dtype=np.float32)
        scores = self._model.predict(
            normalized,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        values = np.asarray(scores, dtype=np.float32).reshape(-1)
        if values.shape != (len(normalized),) or not np.all(np.isfinite(values)):
            raise ValueError("CrossEncoder 输出 shape 非法或包含非有限值")
        return values

