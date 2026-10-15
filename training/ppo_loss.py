"""标准 clipped PPO policy/value objective；KL 仅在 reward shaping 中使用。"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class PPOLossOutput:
    """保留可反向传播 total loss 与 detached 诊断量。"""

    total_loss: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy: torch.Tensor
    approx_kl: torch.Tensor
    log_ratio_mean: torch.Tensor
    ratio_mean: torch.Tensor
    clip_fraction: torch.Tensor


def clipped_ppo_loss(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    new_values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    entropies: torch.Tensor,
    *,
    clip_range: float = 0.2,
    value_clip_range: float | None = 0.2,
    policy_coef: float = 1.0,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    log_ratio_clamp: float = 20.0,
) -> PPOLossOutput:
    """计算 action-level clipped PPO；不会再次施加 reference KL。"""

    tensors = (
        new_logprobs,
        old_logprobs,
        advantages,
        new_values,
        old_values,
        returns,
        entropies,
    )
    if any(tensor.ndim != 1 for tensor in tensors):
        raise ValueError("PPO loss 输入必须是一维")
    if (
        len({tuple(tensor.shape) for tensor in tensors}) != 1
        or not new_logprobs.numel()
    ):
        raise ValueError("PPO loss 输入必须等长且非空")
    if any(not torch.isfinite(tensor).all() for tensor in tensors):
        raise ValueError("PPO loss 输入含 NaN/Inf")
    if clip_range <= 0 or value_coef < 0 or entropy_coef < 0 or log_ratio_clamp <= 0:
        raise ValueError("PPO loss 系数非法")

    old_logprobs = old_logprobs.detach()#转成张量，可供训练
    advantages = advantages.detach()
    old_values = old_values.detach()
    returns = returns.detach()
    log_ratio = new_logprobs - old_logprobs
    if not torch.isfinite(log_ratio).all():
        raise ValueError("PPO log_ratio 含 NaN/Inf")
    # 仅防止 exp 数值溢出；PPO objective 仍由 1±clip_range 执行统计裁剪。
    safe_log_ratio = log_ratio.clamp(-log_ratio_clamp, log_ratio_clamp)
    ratio = torch.exp(safe_log_ratio)
    if not torch.isfinite(ratio).all():
        raise ValueError("PPO ratio 含 NaN/Inf")
    unclipped = ratio * advantages
    clipped_ratio = ratio.clamp(1.0 - clip_range, 1.0 + clip_range)
    policy_loss = -torch.minimum(unclipped, clipped_ratio * advantages).mean()

    if value_clip_range is None:
        value_loss = 0.5 * (new_values - returns).pow(2).mean()
    else:
        clipped_values = old_values + (new_values - old_values).clamp(
            -value_clip_range, value_clip_range
        )
        value_loss = (
            0.5
            * torch.maximum(
                (new_values - returns).pow(2), (clipped_values - returns).pow(2)
            ).mean()
        )
    entropy = entropies.mean()
    total = policy_coef * policy_loss + value_coef * value_loss - entropy_coef * entropy
    if not torch.isfinite(total):
        raise ValueError("PPO total loss 含 NaN/Inf")
    approx_kl = ((ratio - 1.0) - safe_log_ratio).mean()
    log_ratio_mean = log_ratio.mean()
    ratio_mean = ratio.mean()
    clip_fraction = ((ratio - 1.0).abs() > clip_range).float().mean()
    return PPOLossOutput(
        total,
        policy_loss,
        value_loss,
        entropy,
        approx_kl,
        log_ratio_mean,
        ratio_mean,
        clip_fraction,
    )
