"""定义 DenseRetriever 唯一依赖的文本向量编码抽象。"""

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np


class TextEmbedder(ABC):
    """把文本序列编码成形状为 [N, D] 的浮点 numpy 数组。"""

    @abstractmethod
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """批量编码文本，并保持输入顺序。"""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """返回固定向量维度。"""

    @property
    def model_identifier(self) -> str:
        """返回用于索引一致性检查的稳定编码器标识。"""

        return type(self).__name__

    @property
    def normalize_embeddings(self) -> bool:
        """声明编码结果是否已经进行 L2 归一化。"""

        return False

