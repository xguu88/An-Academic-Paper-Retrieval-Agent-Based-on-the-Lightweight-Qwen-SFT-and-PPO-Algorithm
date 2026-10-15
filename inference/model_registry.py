"""base/sft/ppo policy registry；API 不感知 checkpoint 加载细节。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml

from src.inference.policy import AgentPolicy, QwenAgentPolicy
from src.models.qwen_loader import load_qwen_causal_lm, load_qwen_tokenizer


class ModelRegistry:
    """注册稳定模型名称并返回对应推理 policy。"""

    ALLOWED = ("base", "sft", "ppo")

    def __init__(self) -> None:
        self._policies: dict[str, AgentPolicy] = {}
        self._unavailable_reasons: dict[str, str] = {}

    def register(self, name: str, policy: AgentPolicy) -> None:
        if name not in self.ALLOWED:
            raise ValueError(f"未知模型类型: {name}")
        self._policies[name] = policy
        self._unavailable_reasons.pop(name, None)

    def mark_unavailable(self, name: str, reason: str) -> None:
        if name not in self.ALLOWED:
            raise ValueError(f"未知模型类型: {name}")
        self._policies.pop(name, None)
        self._unavailable_reasons[name] = reason

    def get(self, name: str) -> AgentPolicy:
        try:
            return self._policies[name]
        except KeyError as exc:
            raise KeyError(f"模型 {name!r} 尚未配置") from exc

    def list_models(self) -> list[dict[str, object]]:
        models: list[dict[str, object]] = []
        for name in self.ALLOWED:
            available = name in self._policies
            item: dict[str, object] = {"name": name, "available": available}
            if not available:
                item["reason"] = self._unavailable_reasons.get(
                    name, "checkpoint not available"
                )
            models.append(item)
        return models


def load_base_qwen_policy(
    model_path: str | Path, *, generation: dict[str, Any] | None = None
) -> QwenAgentPolicy:
    """Load exactly one local base model for online inference."""

    local_path = Path(model_path).expanduser().resolve()
    if not local_path.is_dir():
        raise FileNotFoundError("Base Qwen model path does not exist")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    tokenizer, _ = load_qwen_tokenizer(local_path)
    model, _ = load_qwen_causal_lm(local_path, dtype=dtype)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    selected = generation or {}
    return QwenAgentPolicy(
        model,
        tokenizer,
        max_new_tokens=int(selected.get("max_new_tokens", 128)),
        do_sample=bool(selected.get("do_sample", False)),
        temperature=float(selected.get("temperature", 0.2)),
        top_p=float(selected.get("top_p", 0.9)),
    )


def load_qwen_adapter_policy(
    base_model_path: str | Path,
    adapter_path: str | Path,
    *,
    generation: dict[str, Any] | None = None,
) -> QwenAgentPolicy:
    """Load an available local adapter without changing base/online action semantics."""

    resolved_adapter = Path(adapter_path).expanduser().resolve()
    if not resolved_adapter.is_dir():
        raise FileNotFoundError(f"adapter checkpoint not available: {resolved_adapter}")
    policy = load_base_qwen_policy(base_model_path, generation=generation)
    from peft import PeftModel

    policy.model = PeftModel.from_pretrained(
        policy.model, resolved_adapter, is_trainable=False
    ).to(policy.device).eval()
    for parameter in policy.model.parameters():
        parameter.requires_grad_(False)
    return policy


def load_qwen_registry(config_path: str | Path) -> ModelRegistry:
    """按配置加载 base/SFT/PPO checkpoints；该函数不依赖训练器。"""

    config_file = Path(config_path).expanduser().resolve()
    project_root = config_file.parent.parent
    config: dict[str, Any] = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    generation = config.get("generation", {})
    registry = ModelRegistry()
    for name in ModelRegistry.ALLOWED:
        entry = config.get("models", {}).get(name, {})
        if not entry.get("enabled", False):
            continue
        raw_base = Path(entry["base_model_path"]).expanduser()
        base_path = (
            raw_base.resolve()
            if raw_base.is_absolute()
            else (project_root / raw_base).resolve()
        )
        tokenizer, _ = load_qwen_tokenizer(base_path)
        model, _ = load_qwen_causal_lm(base_path, dtype=dtype)
        adapter_path = entry.get("adapter_path")
        if adapter_path:
            from peft import PeftModel

            raw_adapter = Path(adapter_path).expanduser()
            resolved_adapter = (
                raw_adapter.resolve()
                if raw_adapter.is_absolute()
                else (project_root / raw_adapter).resolve()
            )
            model = PeftModel.from_pretrained(
                model, resolved_adapter, is_trainable=False
            )
        model.to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        registry.register(
            name,
            QwenAgentPolicy(
                model,
                tokenizer,
                max_new_tokens=int(generation.get("max_new_tokens", 128)),
                do_sample=bool(generation.get("do_sample", False)),
                temperature=float(generation.get("temperature", 0.2)),
                top_p=float(generation.get("top_p", 0.9)),
            ),
        )
    return registry
