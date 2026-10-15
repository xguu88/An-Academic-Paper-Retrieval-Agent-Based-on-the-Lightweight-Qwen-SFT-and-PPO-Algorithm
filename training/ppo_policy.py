"""PPO policy adapter：Qwen generation、response logprob、reference KL 与 value。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.training.ppo_logprob import build_response_mask, masked_causal_action_stats
from src.training.ppo_types import PolicyGeneration
from src.training.ppo_value import ScalarValueHead, extract_last_hidden
from src.training.sft_tokenization import apply_qwen_chat_template


class PPOPolicyAdapter(nn.Module, ABC):
    """Rollout 与 Trainer 共用的最小策略接口。"""

    @abstractmethod
    def generate(self, messages: Sequence[dict[str, str]]) -> PolicyGeneration:
        """生成一次 assistant JSON response，并冻结 rollout 统计。"""

    @abstractmethod
    def value_for_messages(self, messages: Sequence[dict[str, str]]) -> float:
        """计算未终止 state 的 bootstrap value。"""

    @abstractmethod
    def evaluate_actions(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_mask: torch.Tensor,
        prompt_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 new action logprob、entropy、V(s)。"""


class QwenPPOPolicy(PPOPolicyAdapter):
    """共享 Qwen policy backbone，并在最后 prompt token 上接独立 value head。"""

    def __init__(
        self,
        policy_model: nn.Module,
        tokenizer: Any,
        value_head: ScalarValueHead,
        *,
        reference_model: nn.Module | None = None,
        max_new_tokens: int = 128,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
    ) -> None:
        super().__init__()
        self.policy_model = policy_model
        self.tokenizer = tokenizer
        self.value_head = value_head
        self.reference_model = reference_model
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        if not self.do_sample:
            raise ValueError(
                "STRICT PPO 要求 do_sample=true 以保留 on-policy exploration"
            )
        if self.temperature != 1.0 or self.top_p != 1.0 or self.top_k != 0:
            raise ValueError(
                "STRICT PPO 要求 raw-policy sampling: temperature=1, top_p=1, top_k=0"
            )
        if reference_model is not None:
            for parameter in reference_model.parameters():
                parameter.requires_grad_(False)
        self.train(False)

    def train(self, mode: bool = True) -> QwenPPOPolicy:
        """保持 Qwen/reference 永久 eval，同时允许 FP32 value head 参与训练。

        PPO update 不需要 train mode 才能建立 autograd。显式覆盖递归 train，
        防止 LoRA dropout 改变 old/new probability 或重新激活 reference dropout。
        """

        #设置policy关闭dropout、打开backward，value_head打开训练模式
        self.training = mode
        self.policy_model.eval()
        self.value_head.train(mode)
        if self.reference_model is not None:
            self.reference_model.eval()
        self.assert_reference_invariant()
        return self

    def assert_reference_invariant(self) -> None:
        """硬检查 reference 始终 eval 且完全无梯度。"""

        if self.reference_model is None:
            return
        if self.reference_model.training:
            raise RuntimeError("reference policy 必须永久保持 eval mode")
        if any(
            parameter.requires_grad for parameter in self.reference_model.parameters()
        ):
            raise RuntimeError("reference policy 参数必须全部 requires_grad=False")

    @property
    def device(self) -> torch.device:
        """返回 policy 首个参数所在设备。"""

        return next(self.policy_model.parameters()).device

    def _prompt_ids(self, messages: Sequence[dict[str, str]]) -> list[int]:
        return list(
            apply_qwen_chat_template(
                self.tokenizer, messages, tokenize=True, add_generation_prompt=True
            )
        )

    @staticmethod
    def _forward_model(
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        output_hidden_states: bool,
    ) -> Any:
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            use_cache=False,
        )

    def generate(self, messages: Sequence[dict[str, str]]) -> PolicyGeneration:
        """只对实际生成的 assistant token 计算 old/reference logprob。"""
        #产生训练的模型value、policy、reference

        prompt_ids = self._prompt_ids(messages)#转成id
        prompt = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)#转成张量
        attention = torch.ones_like(prompt)
        self.policy_model.eval()
        self.assert_reference_invariant()
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        with torch.no_grad():#关闭 PyTorch 的计算图构建，不记录梯度
            generated = self.policy_model.generate(
                input_ids=prompt,#就是state转成序列化的json输入
                attention_mask=attention,#哪些 token 是真实内容，需要参与注意力计算；哪些 token 只是 padding，需要忽略。
                **generation_kwargs,
            )

            if generated.shape[1] <= prompt.shape[1]:
                raise RuntimeError("Qwen 未生成 assistant token")
            full_attention = torch.ones_like(generated)
            mask = torch.tensor(
                [build_response_mask(generated.shape[1], prompt.shape[1])],
                dtype=torch.long,
                device=self.device,
            )#只训练和评价模型生成的 Action token，前面的 State/Prompt 只是条件，不算作 Action。
            output = self._forward_model(
                self.policy_model, generated,#generated = Prompt/State + 模型生成的 Action
                full_attention, output_hidden_states=True
            )#对 Prompt+Action 做前向传播，得到 logits 和 hidden states
            old_stats = masked_causal_action_stats(output.logits, generated, mask)## 只利用 Action token 计算整个 Action 的 old logprob 和 token entropy
            prompt_lengths = torch.tensor([prompt.shape[1]], device=self.device)#记录Prompt 到哪里结束。
            value = self.value_head(extract_last_hidden(output), prompt_lengths)#执行 Action 之前，这个 State 本身预计未来能得到多少收益。
            reference_logprob = None
            if self.reference_model is not None:
                reference_output = self._forward_model(
                    self.reference_model,
                    generated,
                    full_attention,
                    output_hidden_states=False,
                )
                reference_logprob = float(
                    masked_causal_action_stats(
                        reference_output.logits, generated, mask
                    ).action_logprobs.item()
                )
        response_ids = generated[0, prompt.shape[1] :].tolist()
        result = PolicyGeneration(
            text=self.tokenizer.decode(response_ids, skip_special_tokens=True).strip(),
            input_ids=generated[0].tolist(),
            response_mask=mask[0].tolist(),
            old_logprob=float(old_stats.action_logprobs.item()),
            reference_logprob=reference_logprob,
            value=float(value.item()),
            prompt_length=prompt.shape[1],
        )
        result.validate()
        return result

    def value_for_messages(self, messages: Sequence[dict[str, str]]) -> float:
        """使用 state prompt 最后一个 token 的 hidden state 计算 V(s)。"""

        ids = self._prompt_ids(messages)
        self.policy_model.eval()
        self.assert_reference_invariant()
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        attention = torch.ones_like(input_ids)
        with torch.no_grad():
            output = self._forward_model(
                self.policy_model, input_ids, attention, output_hidden_states=True
            )
            value = self.value_head(
                extract_last_hidden(output),
                torch.tensor([len(ids)], device=self.device),
            )
        return float(value.item())

    def evaluate_actions(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_mask: torch.Tensor,
        prompt_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """更新阶段重新 forward，返回带梯度的 new logprob/entropy/value。"""

        self.policy_model.eval()
        self.assert_reference_invariant()
        output = self._forward_model(
            self.policy_model, input_ids, attention_mask, output_hidden_states=True
        )
        stats = masked_causal_action_stats(output.logits, input_ids, response_mask)
        values = self.value_head(extract_last_hidden(output), prompt_lengths)
        return stats.action_logprobs, stats.mean_token_entropies, values

    def optimizer_parameters(self) -> list[nn.Parameter]:
        """只返回 policy LoRA 与 value head 的可训练参数；reference 永不包含。"""

        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def save_checkpoint(self, output_dir: str | Path) -> None:
        """保存 policy adapter/tokenizer 和独立 value head。"""

        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        if not hasattr(self.policy_model, "save_pretrained"):
            raise TypeError("policy_model 不支持 save_pretrained")
        self.policy_model.save_pretrained(target / "policy_adapter")
        self.tokenizer.save_pretrained(target / "policy_adapter")
        torch.save(self.value_head.state_dict(), target / "value_head.pt")
