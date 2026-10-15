"""定义与具体模型库解耦的 query-document 成对打分接口。"""

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np


class PairScorer(ABC):
    """只负责为文本对计算相关性分数，不负责召回或排序。"""

    @abstractmethod
    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        """按输入顺序返回一维、有限的相关性分数数组。"""

