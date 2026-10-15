"""统一推理 policy interface，以及 mock/Qwen 实现。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from src.agent.actions import ActionType, AgentAction
from src.agent.parser import parse_action
from src.agent.state import AgentState
from src.agent.validator import validate_action
from src.training.sft_tokenization import apply_qwen_chat_template


@dataclass(frozen=True, slots=True)
class ActionDecision:
    """Policy 输出及解析状态；invalid 文本保留用于 trace。"""

    action: AgentAction
    raw_text: str
    parsed_valid: bool
    error: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class TextGeneration:
    text: str
    input_tokens: int
    output_tokens: int


class AgentPolicy(ABC):
    """InferenceRunner 唯一依赖的 policy 抽象。"""

    @abstractmethod
    def generate_action(
        self,
        state: AgentState,
        messages: Sequence[dict[str, str]],
    ) -> ActionDecision:
        """从可见 state prompt 生成一个结构化 AgentAction。"""


class FakeAgentPolicy(AgentPolicy):
    """开发模式固定执行 Semantic→Keyword→Stop，不依赖 checkpoint。"""

    def generate_action(
        self,
        state: AgentState,
        messages: Sequence[dict[str, str]],
    ) -> ActionDecision:
        if state.step == 0:
            action = AgentAction(
                ActionType.SEMANTIC_SEARCH, query=state.current_query, top_k=10
            )
        elif state.step == 1:
            action = AgentAction(
                ActionType.KEYWORD_SEARCH, query=state.current_query, top_k=10
            )
        else:
            action = AgentAction(ActionType.STOP)
        return ActionDecision(action, _compact_action(action), True)


def _compact_action(action: AgentAction) -> str:
    import json

    return json.dumps(action.to_dict(), ensure_ascii=False, separators=(",", ":"))


class QwenAgentPolicy(AgentPolicy):
    """封装 tokenizer/chat template/generate/parse/validate，不依赖 PPOTrainer。"""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        max_new_tokens: int = 128,
        do_sample: bool = False,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def generate_action(
        self,
        state: AgentState,
        messages: Sequence[dict[str, str]],
    ) -> ActionDecision:
        generated = self.generate_text(messages)
        text = generated.text
        try:
            action = parse_action(text)
            validate_action(action)
            return ActionDecision(
                action, text, True, input_tokens=generated.input_tokens,
                output_tokens=generated.output_tokens,
            )
        except ValueError as exc:
            invalid = AgentAction(ActionType.KEYWORD_SEARCH, query=None)
            return ActionDecision(
                invalid, text, False, str(exc), generated.input_tokens,
                generated.output_tokens,
            )

    def generate_text(
        self, messages: Sequence[dict[str, str]], *, max_new_tokens: int | None = None
    ) -> TextGeneration:
        """Generate local-model text and expose exact tokenizer counts."""

        prompt_ids = apply_qwen_chat_template(
            self.tokenizer, messages, tokenize=True, add_generation_prompt=True
        )
        inputs = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        attention = torch.ones_like(inputs)
        self.model.eval()
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens or self.max_new_tokens,
            "do_sample": self.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if self.do_sample:
            generation_kwargs.update(temperature=self.temperature, top_p=self.top_p)
        with torch.no_grad():
            output = self.model.generate(
                input_ids=inputs,
                attention_mask=attention,
                **generation_kwargs,
            )
        generated_ids = output[0, inputs.shape[1] :].tolist()
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return TextGeneration(text, len(prompt_ids), len(generated_ids))
