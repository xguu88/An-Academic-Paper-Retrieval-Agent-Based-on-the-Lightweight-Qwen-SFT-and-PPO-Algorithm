"""提供只返回 input_ids/attention_mask/labels 的 assistant-only SFT Dataset。"""

import json
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from src.training.sft_tokenization import tokenize_sft_example


class SFTDataset(Dataset):#更改为训练需要的数据格式
    """按需应用真实 chat template；metadata 仅保留为可选 debug 字段。"""

    def __init__(
        self,
        path: str | Path,
        tokenizer: Any,
        *,
        max_seq_length: int,
        include_debug_metadata: bool = False,
    ) -> None:
        """加载 JSONL 索引，不执行不安全的尾部截断。"""

        self.samples = [
            json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
        ]
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.include_debug_metadata = include_debug_metadata

    def __len__(self) -> int:
        """返回样本数。"""

        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """返回训练张量前的整数列表，metadata 默认不返回。"""

        sample = self.samples[index]
        encoded = tokenize_sft_example(
            self.tokenizer, sample, max_seq_length=self.max_seq_length
        )
        item: dict[str, Any] = {
            "input_ids": encoded.input_ids,
            "attention_mask": encoded.attention_mask,
            "labels": encoded.labels,
        }
        if self.include_debug_metadata:
            item["metadata"] = {
                "sample_id": sample.get("sample_id"),
                "task_id": sample.get("task_id"),
            }
        return item

