"""提供 LoRA 注入前的模块审计与训练参数门禁。"""

from dataclasses import asdict, dataclass
from typing import Any, Sequence


DEFAULT_LORA_TARGETS = (#设置的需要微调的参数
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass(slots=True)
class LoRATargetAudit:
    """记录真实模型中 LoRA 目标模块的发现情况。"""

    requested_targets: list[str]
    present_targets: list[str]
    missing_targets: list[str]
    matched_module_names: dict[str, list[str]]

    def to_dict(self) -> dict[str, Any]:
        """转换为 JSON 可序列化字典。"""

        return asdict(self)


def inspect_lora_target_modules(
    model: Any,
    targets: Sequence[str] = DEFAULT_LORA_TARGETS,
) -> LoRATargetAudit:
    """检查模块名后缀，禁止在未确认结构时盲目写死目标层。"""

    matches = {target: [] for target in targets}
    for name, _module in model.named_modules():
        leaf_name = name.rsplit(".", 1)[-1]
        if leaf_name in matches:
            matches[leaf_name].append(name)
    present = [target for target in targets if matches[target]]
    missing = [target for target in targets if not matches[target]]
    return LoRATargetAudit(
        requested_targets=list(targets),
        present_targets=present,
        missing_targets=missing,
        matched_module_names=matches,
    )


def inject_lora(
    model: Any,
    *,
    targets: Sequence[str] = DEFAULT_LORA_TARGETS,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
) -> Any:
    """仅在目标层全部存在且 PEFT 可用时注入 LoRA。"""

    audit = inspect_lora_target_modules(model, targets)
    if audit.missing_targets:
        raise ValueError(f"LoRA target modules missing: {audit.missing_targets}")
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise RuntimeError("PEFT NOT INSTALLED: LoRA injection blocked") from exc
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=list(targets),
    )
    return get_peft_model(model, config)#融合参数


def trainable_parameter_stats(model: Any) -> dict[str, int | float]:
    """统计总参数与可训练参数，供 base freeze 门禁使用。"""

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_ratio": trainable / total if total else 0.0,
    }


def validate_lora_trainability(model: Any) -> dict[str, int | float]:
    """确认只有名称含 lora_ 的参数可训练，防止误解冻基础模型。"""

    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if not trainable_names:
        raise ValueError("LoRA 模型没有可训练参数")
    unexpected = [name for name in trainable_names if "lora_" not in name]
    if unexpected:
        raise ValueError(f"发现非 LoRA 可训练参数: {unexpected[:10]}")
    return trainable_parameter_stats(model)
