"""基于 Qwen hidden state 的独立标量 value head。"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class ScalarValueHead(nn.Module):#**选出 prompt 末尾那一个 token 的向量，映射成一个标量价值 V (s)**，代表当前状态的预估回报。
    """将最后一个 prompt token 的 hidden state 映射为 V(s)。
    # 选出 prompt 末尾那个 token 的 hidden state，
# 通过一个线性层映射成标量 V(s)
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size 必须为正整数")
        self.projection = nn.Linear(hidden_size, 1)

    def forward(
        self, hidden_states: torch.Tensor, prompt_lengths: torch.Tensor
    ) -> torch.Tensor:
        """从 [B,T,H] 中选取 prompt_lengths-1 位置，返回 [B] value。"""

        if hidden_states.ndim != 3 or prompt_lengths.ndim != 1:
            raise ValueError("hidden_states=[B,T,H]，prompt_lengths=[B]")
        if hidden_states.shape[0] != prompt_lengths.shape[0]:
            raise ValueError("value batch size 不一致")
        indexes = prompt_lengths.to(hidden_states.device, dtype=torch.long) - 1
        if torch.any(indexes < 0) or torch.any(indexes >= hidden_states.shape[1]):
            raise ValueError("prompt length 超出 hidden state 序列")
        batch = torch.arange(hidden_states.shape[0], device=hidden_states.device)
        selected = hidden_states[batch, indexes]
        # Critic 保持 FP32；只转换 [B,H]，避免把完整 Qwen hidden states 扩为 FP32。
        selected = selected.to(dtype=self.projection.weight.dtype)
        return self.projection(selected).squeeze(-1)


def extract_last_hidden(model_output: Any) -> torch.Tensor:
    """从 Transformers causal LM output 中提取最后一层 hidden state。"""

    hidden_states = getattr(model_output, "hidden_states", None)
    if not hidden_states:
        raise ValueError("模型 forward 必须启用 output_hidden_states=True")
    return hidden_states[-1]


def value_parameter_report(
    policy_model: nn.Module, value_head: nn.Module
) -> dict[str, int | float]:
    """分别统计 policy 与 value head 的 trainable/all 参数。"""

    policy_all = sum(parameter.numel() for parameter in policy_model.parameters())
    policy_trainable = sum(
        parameter.numel()
        for parameter in policy_model.parameters()
        if parameter.requires_grad
    )
    value_all = sum(parameter.numel() for parameter in value_head.parameters())
    value_trainable = sum(
        parameter.numel()
        for parameter in value_head.parameters()
        if parameter.requires_grad
    )
    all_parameters = policy_all + value_all
    trainable = policy_trainable + value_trainable
    return {
        "all_parameters": all_parameters,
        "trainable_parameters": trainable,
        "trainable_ratio": trainable / all_parameters if all_parameters else 0.0,
        "policy_all_parameters": policy_all,
        "policy_trainable_parameters": policy_trainable,
        "value_head_parameters": value_all,
        "value_head_trainable_parameters": value_trainable,
    }
