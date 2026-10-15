"""基于真实 chat template 分段计算 assistant-only labels 与长度审计。"""

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np


class SFTSequenceTooLongError(ValueError):
    """序列超过 max_seq_length，且不能安全截断 assistant target。"""


@dataclass(slots=True)
class TokenizedSFTExample:
    """保存模型输入以及模板边界推导出的 assistant span。"""

    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    prompt_tokens: int
    assistant_tokens: int


def chat_template_kwargs(tokenizer: Any) -> tuple[dict[str, Any], str]:
    """只在真实 template 声明 enable_thinking 时显式关闭 thinking。"""

    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str) or not template.strip():
        raise ValueError("tokenizer.chat_template 不存在")
    if "enable_thinking" in template:
        return {"enable_thinking": False}, "template_enable_thinking_false"
    return {}, "template_has_no_thinking_switch"


def apply_qwen_chat_template(
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
    *,
    tokenize: bool,
    add_generation_prompt: bool,
) -> Any:
    """渲染真实模板；需要 token 时统一走显式文本编码。"""

    kwargs, _ = chat_template_kwargs(tokenizer)
    rendered = tokenizer.apply_chat_template(
        list(messages),
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        **kwargs,
    )
    if not tokenize:
        return rendered
    return tokenizer(
        rendered,
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]


def tokenize_sft_example(
    tokenizer: Any,
    sample: Mapping[str, Any],
    *,
    max_seq_length: int | None = None,
) -> TokenizedSFTExample:
    """先渲染模板文本再显式编码，以稳定定位 assistant token span。"""

    messages = sample.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        raise ValueError("SFT sample 必须包含 system/user/assistant 三条 messages")
    if [message.get("role") for message in messages] != ["system", "user", "assistant"]:
        raise ValueError("messages role 顺序必须是 system,user,assistant")
    # 不依赖 apply_chat_template(tokenize=True) 在不同 Transformers 版本中的
    # 返回容器；统一先取得文本，再使用 tokenizer 显式编码。
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    prompt_text = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    full_ids = list(
        tokenizer(
            full_text,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
    )
    prompt_ids = list(
        tokenizer(
            prompt_text,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("assistant prefix 与完整序列不匹配")
    assistant_start = len(prompt_ids)
    if assistant_start >= len(full_ids):
        raise ValueError("assistant span 为空")
    if max_seq_length is not None and len(full_ids) > max_seq_length:
        raise SFTSequenceTooLongError(
            f"sequence length {len(full_ids)} exceeds {max_seq_length}; "
            "refuse tail truncation because assistant target must remain complete"
        )
    # assistant turn 的终止 token 属于 full_ids 后缀，因此也接受监督。
    labels = [-100] * assistant_start + full_ids[assistant_start:]
    return TokenizedSFTExample(
        input_ids=full_ids,
        attention_mask=[1] * len(full_ids),
        labels=labels,
        prompt_tokens=assistant_start,
        assistant_tokens=len(full_ids) - assistant_start,
    )


def _stats(values: list[int]) -> dict[str, float | int]:
    """计算 token length 的 mean/P50/P90/P95/P99/max。"""

    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": int(array.max()),
    }


def audit_token_lengths(tokenizer: Any, samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """对全部 chat-template 序列统计总/prompt/assistant tokens 与阈值超限。"""

    encoded = [tokenize_sft_example(tokenizer, sample) for sample in samples]
    totals = [len(item.input_ids) for item in encoded]
    prompts = [item.prompt_tokens for item in encoded]
    assistants = [item.assistant_tokens for item in encoded]
    thresholds = {}
    for threshold in (2048, 3072, 4096, 6144, 8192):
        count = sum(length > threshold for length in totals)
        thresholds[str(threshold)] = {"count": count, "ratio": count / len(totals)}
    return {
        "num_samples": len(samples),
        "total_tokens": _stats(totals),
        "prompt_tokens": _stats(prompts),
        "assistant_target_tokens": _stats(assistants),
        "exceeds": thresholds,
    }


def audit_assistant_masks(
    tokenizer: Any,
    samples: Sequence[Mapping[str, Any]],
    *,
    sample_count: int = 20,
) -> dict[str, Any]:
    """统计监督比例，并稳定抽取前后均匀样本解码 supervised token。"""

    encoded = [tokenize_sft_example(tokenizer, sample) for sample in samples]
    total = sum(len(item.input_ids) for item in encoded)
    supervised = sum(item.assistant_tokens for item in encoded)
    # 固定随机种子保证审计可复现，同时避免只检查文件头部样本。
    rng = np.random.default_rng(42)
    indexes = np.sort(
        rng.choice(len(samples), size=min(sample_count, len(samples)), replace=False)
    )
    decoded = []
    for index in indexes.tolist():
        item = encoded[index]
        token_ids = [label for label in item.labels if label != -100]
        decoded.append(
            {
                "sample_id": samples[index].get("sample_id"),
                "decoded_supervised": tokenizer.decode(token_ids, skip_special_tokens=False),
                "assistant_target": samples[index]["messages"][2]["content"],
            }
        )
    return {
        "total_tokens": total,
        "supervised_tokens": supervised,
        "masked_tokens": total - supervised,
        "supervised_ratio": supervised / total,
        "decoded_samples": decoded,
    }
