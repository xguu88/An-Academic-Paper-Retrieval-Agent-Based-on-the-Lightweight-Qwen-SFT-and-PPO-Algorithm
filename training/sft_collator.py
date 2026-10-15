"""只负责 SFT batch padding，不重新创建 chat 文本或 labels。"""

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(slots=True)
class SFTDataCollator:
    """input 用 pad_token_id，attention 用 0，labels 用 -100 右侧填充。"""

    pad_token_id: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """把不同长度整数列表转换为 batch-first LongTensor。"""

        if not features:
            raise ValueError("features 不能为空")
        max_length = max(len(feature["input_ids"]) for feature in features)
        input_ids, attention_masks, labels = [], [], []
        for feature in features:
            length = len(feature["input_ids"])
            padding = max_length - length
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * padding)
            attention_masks.append(feature["attention_mask"] + [0] * padding)
            labels.append(feature["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

