"""PPO rollout buffer：逐 episode GAE、冻结 old stats 与变长序列 padding。"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch

from src.training.ppo_advantage import compute_gae, normalize_advantages
from src.training.ppo_types import PPOConfig, PPORolloutEpisode, PPORolloutStep


@dataclass(slots=True)
class PPOMinibatch:
    """Trainer update 所需的 padded tensor batch。"""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    response_mask: torch.Tensor
    prompt_lengths: torch.Tensor
    old_logprobs: torch.Tensor
    old_values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor

    def to(self, device: torch.device) -> PPOMinibatch:
        """把全部 tensor 移到同一设备。"""

        return PPOMinibatch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            response_mask=self.response_mask.to(device),
            prompt_lengths=self.prompt_lengths.to(device),
            old_logprobs=self.old_logprobs.to(device),
            old_values=self.old_values.to(device),
            advantages=self.advantages.to(device),
            returns=self.returns.to(device),
        )


class PPORolloutBuffer:
    """保存 rollout snapshot；old_logprob/value 在 update 中始终 detached。"""

    def __init__(self, config: PPOConfig, pad_token_id: int = 0) -> None:
        self.config = config
        self.pad_token_id = int(pad_token_id)
        self.steps: list[PPORolloutStep] = []

    def add_episode(self, episode: PPORolloutEpisode) -> None:
        """逐 episode 计算 GAE，避免跨 task 传播 advantage。"""

        if not episode.steps:
            raise ValueError("不能加入空 episode")
        rewards = torch.tensor([step.total_ppo_reward for step in episode.steps])
        values = torch.tensor([step.value for step in episode.steps])
        dones = torch.tensor([float(step.done) for step in episode.steps])
        advantages, returns = compute_gae(
            rewards,
            values,
            dones,
            bootstrap_value=episode.bootstrap_value,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )
        for step, advantage, return_ in zip(episode.steps, advantages, returns):
            step.advantage = float(advantage.item())
            step.return_ = float(return_.item())
            self.steps.append(step)

    def finalize(self) -> None:
        """可选地对完整 rollout batch 标准化 advantage。"""

        if not self.steps:
            raise ValueError("rollout buffer 为空")
        if self.config.normalize_advantages:
            values = normalize_advantages(
                torch.tensor([step.advantage for step in self.steps])
            )
            for step, value in zip(self.steps, values):
                step.advantage = float(value.item())

    def _collate(self, indexes: torch.Tensor) -> PPOMinibatch:
        selected = [self.steps[index] for index in indexes.tolist()]
        max_length = max(len(step.input_ids) for step in selected)
        input_ids, attention, masks = [], [], []
        for step in selected:
            padding = max_length - len(step.input_ids)
            input_ids.append(step.input_ids + [self.pad_token_id] * padding)
            attention.append([1] * len(step.input_ids) + [0] * padding)
            masks.append(step.response_mask + [0] * padding)
        return PPOMinibatch(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            attention_mask=torch.tensor(attention, dtype=torch.long),
            response_mask=torch.tensor(masks, dtype=torch.long),
            prompt_lengths=torch.tensor([step.prompt_length for step in selected]),
            old_logprobs=torch.tensor([step.old_logprob for step in selected])
            .detach()
            .clone(),
            old_values=torch.tensor([step.value for step in selected]).detach().clone(),
            advantages=torch.tensor([step.advantage for step in selected])
            .detach()
            .clone(),
            returns=torch.tensor([step.return_ for step in selected]).detach().clone(),
        )

    def minibatches(
        self, *, generator: torch.Generator | None = None
    ) -> Iterator[PPOMinibatch]:
        """随机生成一个 PPO epoch 的 minibatches。"""

        if not self.steps:
            raise ValueError("rollout buffer 为空")
        order = torch.randperm(len(self.steps), generator=generator)
        for start in range(0, len(order), self.config.minibatch_size):
            yield self._collate(order[start : start + self.config.minibatch_size])
