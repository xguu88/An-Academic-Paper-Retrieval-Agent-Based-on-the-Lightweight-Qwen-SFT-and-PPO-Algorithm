"""独立、可单测的 Generalized Advantage Estimation。"""

from __future__ import annotations

import torch


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    *,
    bootstrap_value: float | torch.Tensor = 0.0,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """计算一条 trajectory 的 advantages 与 returns。

    terminal step 的 done=1 会屏蔽下一状态；truncated episode 最后一项 done=0，
    并通过 bootstrap_value 使用 V(s_{T})。
    """

    if rewards.ndim != 1 or values.ndim != 1 or dones.ndim != 1:
        raise ValueError("rewards/values/dones 必须是一维")
    if not (rewards.shape == values.shape == dones.shape) or rewards.numel() == 0:
        raise ValueError("GAE 输入必须等长且非空")
    if not 0 < gamma <= 1 or not 0 < gae_lambda <= 1:
        raise ValueError("gamma/gae_lambda 必须位于 (0,1]")
    if not torch.isfinite(rewards).all() or not torch.isfinite(values).all():
        raise ValueError("GAE 输入含 NaN/Inf")
    done_values = dones.to(dtype=values.dtype)
    if torch.any((done_values != 0) & (done_values != 1)):
        raise ValueError("dones 只能包含 0/1")
    next_value = torch.as_tensor(
        bootstrap_value, dtype=values.dtype, device=values.device
    )
    advantages = torch.zeros_like(values)
    running = torch.zeros((), dtype=values.dtype, device=values.device)
    for index in range(rewards.numel() - 1, -1, -1):
        not_done = 1.0 - done_values[index]
        delta = rewards[index] + gamma * next_value * not_done - values[index]
        running = delta + gamma * gae_lambda * not_done * running
        advantages[index] = running
        next_value = values[index]
    return advantages, advantages + values


def normalize_advantages(
    advantages: torch.Tensor, epsilon: float = 1e-8
) -> torch.Tensor:
    """标准化 advantage；常数向量稳定返回全零。"""

    if advantages.numel() == 0 or not torch.isfinite(advantages).all():
        raise ValueError("advantages 必须非空且有限")
    std = advantages.std(unbiased=False)
    if std <= epsilon:
        return torch.zeros_like(advantages)
    return (advantages - advantages.mean()) / (std + epsilon)
