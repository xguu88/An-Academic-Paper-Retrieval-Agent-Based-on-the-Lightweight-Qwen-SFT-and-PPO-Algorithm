"""仅供单元测试和 CPU smoke 的微型可训练 PPO policy。"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from src.training.ppo_policy import PPOPolicyAdapter
from src.training.ppo_types import PolicyGeneration


class TinyTrainablePPOPolicy(PPOPolicyAdapter):
    """用两个标量模拟 policy/value，绝不代表真实 Qwen 质量。"""

    def __init__(
        self,
        responses: Sequence[str],
        *,
        with_reference: bool = False,
    ) -> None:
        super().__init__()
        if not responses:
            raise ValueError("responses 不能为空")
        self.responses = list(responses)
        self.policy_logprob = nn.Parameter(torch.tensor(-1.0))
        self.value_bias = nn.Parameter(torch.tensor(0.0))
        self.reference_anchor = nn.Parameter(torch.tensor(-1.1), requires_grad=False)
        self.with_reference = with_reference
        self.calls = 0

    @staticmethod
    def _tokenize(text: str) -> list[int]:
        return [10 + (byte % 200) for byte in text.encode("utf-8")]

    def generate(self, messages: Sequence[dict[str, str]]) -> PolicyGeneration:
        """按队列返回 JSON 文本并构造可变长度 response mask。"""

        text = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        prompt_ids = [1, 2, 3]
        response_ids = self._tokenize(text) or [4]
        result = PolicyGeneration(
            text=text,
            input_ids=prompt_ids + response_ids,
            response_mask=[0] * len(prompt_ids) + [1] * len(response_ids),
            prompt_length=len(prompt_ids),
            old_logprob=float(self.policy_logprob.detach().item()),
            reference_logprob=(
                float(self.reference_anchor.detach().item())
                if self.with_reference
                else None
            ),
            value=float(self.value_bias.detach().item()),
        )
        result.validate()
        return result

    def value_for_messages(self, messages: Sequence[dict[str, str]]) -> float:
        return float(self.value_bias.detach().item())

    def evaluate_actions(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_mask: torch.Tensor,
        prompt_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = input_ids.shape[0]
        if response_mask.shape != input_ids.shape or torch.any(
            response_mask * (1 - attention_mask)
        ):
            raise ValueError("padding 不得进入 response mask")
        logprobs = self.policy_logprob.expand(batch)
        entropies = (0.5 + self.policy_logprob * 0.0).expand(batch)
        values = self.value_bias.expand(batch)
        return logprobs, entropies, values
