"""集中实现 causal LM assistant-token log probability 与 entropy。"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class CausalActionStats:
    """完整 assistant macro-action 的概率及 token 级诊断量。"""

    action_logprobs: torch.Tensor
    mean_token_entropies: torch.Tensor
    token_counts: torch.Tensor
    token_logprobs: torch.Tensor
    shifted_response_mask: torch.Tensor


def masked_causal_action_stats(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    response_mask: torch.Tensor,
) -> CausalActionStats:
    """计算严格 macro-action SUM logprob 与 response-token 诊断量。

    `logits[:, t]` 预测 `input_ids[:, t+1]`，因此 response mask 同样
    右移一位。prompt 与 padding token 必须在 response_mask 中为 0。
    entropy 保持为 response positions 上 raw-softmax token entropy 的均值；
    它不是 sequence entropy。


    把模型对整段 Action 的 token 预测概率，合成为 PPO 需要的“一个完整宏动作的 logprob”，同时算 Action token 的平均 entropy。
    """

    if logits.ndim != 3 or input_ids.ndim != 2 or response_mask.ndim != 2:
        raise ValueError("logits 必须为 [B,T,V]，input_ids/mask 必须为 [B,T]")
    if logits.shape[:2] != input_ids.shape or input_ids.shape != response_mask.shape:
        raise ValueError("logits/input_ids/response_mask 序列 shape 不一致")
    if not torch.isfinite(logits).all():
        raise ValueError("logits 含 NaN/Inf")
    shifted_logits = logits[:, :-1, :]
    shifted_targets = input_ids[:, 1:]
    active_mask = response_mask[:, 1:].to(dtype=torch.bool)
    counts = active_mask.sum(dim=-1).to(dtype=torch.float32)
    if torch.any(counts <= 0):
        raise ValueError("每个样本必须至少包含一个受监督 response token")
    # 只对 response positions 计算 FP32 softmax：保留长 sequence SUM 精度，
    # 同时避免把整个 prompt 的 [T,V] 概率张量扩为 FP32。
    active_logits = shifted_logits[active_mask].to(dtype=torch.float32)
    active_targets = shifted_targets[active_mask]
    active_log_probs = torch.log_softmax(active_logits, dim=-1)
    active_token_logprobs = active_log_probs.gather(
        -1, active_targets.unsqueeze(-1)
    ).squeeze(-1)
    active_probabilities = active_log_probs.exp()
    active_token_entropies = -(active_probabilities * active_log_probs).sum(dim=-1)
    token_log_probs = torch.zeros(
        active_mask.shape, dtype=torch.float32, device=logits.device
    ).masked_scatter(active_mask, active_token_logprobs)
    token_entropies = torch.zeros_like(token_log_probs).masked_scatter(
        active_mask, active_token_entropies
    )
    shifted_mask = active_mask.to(dtype=torch.float32)
    action_logprobs = token_log_probs.sum(dim=-1)
    mean_token_entropies = token_entropies.sum(dim=-1) / counts
    return CausalActionStats(
        action_logprobs=action_logprobs,
        mean_token_entropies=mean_token_entropies,
        token_counts=counts,
        token_logprobs=token_log_probs,
        shifted_response_mask=shifted_mask,
    )


def masked_causal_token_stats(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    response_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """兼容接口：返回 action SUM logprob、mean token entropy、token 数。"""

    stats = masked_causal_action_stats(logits, input_ids, response_mask)
    return stats.action_logprobs, stats.mean_token_entropies, stats.token_counts


def build_response_mask(length: int, prompt_length: int) -> list[int]:
    """为完整 prompt+assistant token 序列构造 response-only 0/1 mask。"""

    if prompt_length <= 0 or prompt_length >= length:
        raise ValueError("prompt_length 必须位于 (0, length)")
    return [0] * prompt_length + [1] * (length - prompt_length)
