"""提供仅验证 Dense Retrieval 工程流程的确定性测试编码器。"""

import hashlib
from collections.abc import Sequence

import numpy as np

from src.retrieval.embedding import TextEmbedder
from src.retrieval.pair_scorer import PairScorer
from src.retrieval.tokenization import tokenize


class HashingTextEmbedder(TextEmbedder):
    """确定性 token hashing 编码器；不是正式语义模型，不代表检索效果。"""

    def __init__(self, dimension: int = 256, normalize: bool = True) -> None:
        """设置固定向量维度及是否输出 L2 归一化向量。"""

        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("dimension 必须是大于 0 的整数")
        self._dimension = dimension
        self._normalize = normalize

    @property
    def dimension(self) -> int:
        """返回 hashing 向量维度。"""

        return self._dimension

    @property
    def model_identifier(self) -> str:
        """返回包含维度的稳定测试编码器标识。"""

        return f"hashing-test-v1-d{self.dimension}"

    @property
    def normalize_embeddings(self) -> bool:
        """返回是否在 encode 中进行 L2 归一化。"""

        return self._normalize

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """将相同 token 映射到相同维度，使词项重叠产生基础相似性。"""

        if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
            raise TypeError("texts 必须是字符串序列")
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError("texts 中的每个元素必须是字符串")
            for token in tokenize(text):
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:8], "big") % self.dimension
                sign = 1.0 if digest[8] % 2 == 0 else -1.0
                vectors[row, index] += sign
        if self._normalize:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            np.divide(vectors, norms, out=vectors, where=norms > 0)
        return vectors


class DeterministicPairScorer(PairScorer):
    """按精确文本对查表的测试打分器；不代表真实语义重排能力。"""

    def __init__(
        self,
        scores: dict[tuple[str, str], float] | None = None,
        *,
        default_score: float = 0.0,
    ) -> None:
        """保存固定映射；相同输入在不同运行中得到完全相同输出。"""

        self.scores = dict(scores or {})
        self.default_score = float(default_score)

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        """按输入顺序查表，未知文本对使用固定默认分数。"""

        if isinstance(pairs, (str, bytes)) or not isinstance(pairs, Sequence):
            raise TypeError("pairs 必须是 (query, document) 序列")
        values: list[float] = []
        for pair in pairs:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise TypeError("每个 pair 必须是长度为 2 的 tuple")
            query, document = pair
            if not isinstance(query, str) or not isinstance(document, str):
                raise TypeError("pair 中 query 和 document 必须是字符串")
            values.append(float(self.scores.get(pair, self.default_score)))
        return np.asarray(values, dtype=np.float32)
