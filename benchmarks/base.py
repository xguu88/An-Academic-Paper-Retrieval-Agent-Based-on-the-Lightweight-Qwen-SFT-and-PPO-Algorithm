"""定义外部 benchmark 转换器的统一抽象接口。"""

from abc import ABC, abstractmethod
from pathlib import Path

from src.benchmarks.types import BenchmarkConversionReport


class BenchmarkAdapter(ABC):
    """只负责外部格式到项目标准 papers/tasks JSONL 的转换。"""

    @abstractmethod
    def convert(self, source_dir: Path, output_dir: Path) -> BenchmarkConversionReport:
        """转换 benchmark 并返回可审计统计，不直接构建 Retriever。"""

