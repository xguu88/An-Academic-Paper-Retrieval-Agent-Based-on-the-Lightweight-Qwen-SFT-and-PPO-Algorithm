"""严格审计 SFT messages 的标签泄漏、Action schema、重复和 prompt 长度。"""

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.agent.actions import ActionType
from src.agent.parser import parse_action
from src.agent.validator import validate_action
from src.evaluation.tasks import RetrievalTask


FORBIDDEN_MESSAGE_TERMS = (
    "relevant_paper_ids",
    "relevance_grades",
    "qrels",
    "oracle",
    "ground_truth",
    "reward_components",
    "trajectory_reward",
    "trajectory_success",
    "first_relevant_rank",
    '"reward":',
    '"success":',
)


@dataclass(slots=True)
class SFTAuditReport:
    """保存 schema、泄漏、分布、重复和字符/词长度审计结果。"""

    num_samples: int
    parse_success_rate: float
    validation_success_rate: float
    invalid_target_count: int
    leakage_count: int
    duplicate_prompt_count: int
    duplicate_target_count: int
    action_distribution: dict[str, dict[str, float | int]]
    mean_prompt_chars: float
    p50_prompt_chars: float
    p95_prompt_chars: float
    max_prompt_chars: int
    mean_prompt_words: float
    p50_prompt_words: float
    p95_prompt_words: float
    max_prompt_words: int
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        """转换为 JSON 兼容字典。"""

        return asdict(self)


def _message_map(sample: Mapping[str, Any]) -> dict[str, str]:
    """验证 system/user/assistant 三条消息并按 role 建立映射。"""

    messages = sample.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        raise ValueError("每个 SFT sample 必须恰好包含 system/user/assistant 三条消息")
    mapped: dict[str, str] = {}
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("message 必须是对象")
        role, content = message.get("role"), message.get("content")
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            raise ValueError("message role/content 非法")
        mapped[role] = content
    if set(mapped) != {"system", "user", "assistant"}:
        raise ValueError("message roles 必须且只能是 system/user/assistant")
    return mapped


def audit_sft_samples(
    samples: Sequence[Mapping[str, Any]],
    tasks: Sequence[RetrievalTask],
) -> SFTAuditReport:
    """审计 messages；相关 ID 仅在可见候选或原始可见 query 中出现时合法。"""

    if not samples:
        raise ValueError("samples 不能为空")
    truth = {task.task_id: set(task.relevant_paper_ids) for task in tasks}
    prompts: list[str] = []
    targets: list[str] = []
    char_lengths: list[int] = []
    word_lengths: list[int] = []
    action_counts: Counter[str] = Counter()
    parse_success = validation_success = invalid = leakage = 0
    for sample in samples:
        messages = _message_map(sample)
        combined = "\n".join(messages.values()).lower()
        if any(term in combined for term in FORBIDDEN_MESSAGE_TERMS):
            leakage += 1
        metadata = sample.get("metadata", {})
        visible_ids = set(metadata.get("visible_candidate_ids", []))
        visible_queries = metadata.get("visible_query_texts", [])
        allowed_query_text = "\n".join(
            text for text in visible_queries if isinstance(text, str)
        )
        task_id = sample.get("task_id")
        for relevant_id in truth.get(task_id, set()):
            appears = relevant_id in messages["user"] or relevant_id in messages["assistant"]
            naturally_visible = relevant_id in visible_ids or relevant_id in allowed_query_text
            if appears and not naturally_visible:
                leakage += 1
                break
        target = messages["assistant"]
        try:
            action = parse_action(target)
            parse_success += 1
            validate_action(action)
            validation_success += 1
            action_counts[action.action.value] += 1
        except ValueError:
            invalid += 1
        prompt = messages["system"] + "\n" + messages["user"]
        prompts.append(hashlib.sha256(prompt.encode("utf-8")).hexdigest())
        targets.append(target)
        char_lengths.append(len(prompt))
        word_lengths.append(len(prompt.split()))
    count = len(samples)
    distribution = {
        action.value: {
            "count": action_counts.get(action.value, 0),
            "percentage": action_counts.get(action.value, 0) / count,
        }
        for action in ActionType
    }
    warnings: list[str] = []
    stop_ratio = action_counts.get(ActionType.STOP.value, 0) / count
    if stop_ratio > 0.60:
        warnings.append("Stop action percentage exceeds 60%")
    search_counts = [
        action_counts.get(action.value, 0)
        for action in (
            ActionType.KEYWORD_SEARCH,
            ActionType.SEMANTIC_SEARCH,
            ActionType.HYBRID_SEARCH,
        )
    ]
    if sum(search_counts) and max(search_counts) / sum(search_counts) > 0.80:
        warnings.append("One search action exceeds 80% of search targets")
    chars = np.asarray(char_lengths, dtype=np.float64)
    words = np.asarray(word_lengths, dtype=np.float64)
    return SFTAuditReport(
        num_samples=count,
        parse_success_rate=parse_success / count,
        validation_success_rate=validation_success / count,
        invalid_target_count=invalid,
        leakage_count=leakage,
        duplicate_prompt_count=count - len(set(prompts)),
        duplicate_target_count=count - len(set(targets)),
        action_distribution=distribution,
        mean_prompt_chars=float(chars.mean()),
        p50_prompt_chars=float(np.percentile(chars, 50)),
        p95_prompt_chars=float(np.percentile(chars, 95)),
        max_prompt_chars=int(chars.max()),
        mean_prompt_words=float(words.mean()),
        p50_prompt_words=float(np.percentile(words, 50)),
        p95_prompt_words=float(np.percentile(words, 95)),
        max_prompt_words=int(words.max()),
        warnings=warnings,
    )


def validate_sft_source_path(
    path: str | Path,
    *,
    allow_test_override: bool = False,
) -> Path:
    """默认硬拒绝 test_tasks.jsonl，只有显式 override 才允许。"""

    resolved = Path(path).resolve()
    if resolved.name == "test_tasks.jsonl" and not allow_test_override:
        raise PermissionError("禁止使用 frozen test 生成 SFT 数据；需显式 override")
    if resolved.name not in {"rl_train_tasks.jsonl", "rl_dev_tasks.jsonl", "test_tasks.jsonl"}:
        raise ValueError("SFT source 必须是 RL train/dev；test 仅可显式 override")
    return resolved


def sha256_file(path: str | Path) -> str:
    """返回数据或配置文件的 SHA-256。"""

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_json_sha256(value: Any) -> str:
    """对 JSON 兼容对象计算排序键稳定 fingerprint。"""

    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def prompt_hash(sample: Mapping[str, Any]) -> str:
    """对模型输入的 system+user 计算稳定 SHA-256，不包含 assistant target。"""

    messages = _message_map(sample)
    prompt = messages["system"] + "\n" + messages["user"]
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def analyze_conflicting_supervision(
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """区分同 prompt 同 target 的 exact duplicate 与不同 target 的 conflict。"""

    groups: dict[str, list[Mapping[str, Any]]] = {}
    for sample in samples:
        groups.setdefault(prompt_hash(sample), []).append(sample)
    duplicate_groups = 0
    exact_duplicate_count = 0
    conflicting_groups: list[dict[str, Any]] = []
    for digest, group in groups.items():
        if len(group) > 1:
            duplicate_groups += 1
        targets = [_message_map(sample)["assistant"] for sample in group]
        counts = Counter(targets)
        exact_duplicate_count += sum(count - 1 for count in counts.values())
        if len(counts) > 1:
            conflicting_groups.append(
                {
                    "prompt_hash": digest,
                    "sample_ids": [sample.get("sample_id") for sample in group],
                    "task_ids": [sample.get("task_id") for sample in group],
                    "trajectory_ids": [sample.get("trajectory_id") for sample in group],
                    "targets": targets,
                    "action_types": [
                        sample.get("metadata", {}).get("action_type") for sample in group
                    ],
                    "source_policy": [
                        sample.get("metadata", {}).get("source_policy") for sample in group
                    ],
                }
            )
    return {
        "num_samples": len(samples),
        "num_unique_prompts": len(groups),
        "duplicate_prompt_groups": duplicate_groups,
        "exact_duplicate_sample_count": exact_duplicate_count,
        "conflicting_prompt_groups": len(conflicting_groups),
        "conflicting_sample_count": sum(
            len(group["sample_ids"]) for group in conflicting_groups
        ),
        "conflicts": conflicting_groups,
    }


def _trajectory_roles(samples: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """按生成顺序把每个 task 首个 trajectory 标为 primary，其余标为 alternative。"""

    first_by_task: dict[str, str] = {}
    roles: dict[str, str] = {}
    for sample in samples:
        task_id = str(sample.get("task_id"))
        trajectory_id = str(sample.get("trajectory_id"))
        if task_id not in first_by_task:
            first_by_task[task_id] = trajectory_id
        roles[trajectory_id] = (
            "primary" if first_by_task[task_id] == trajectory_id else "alternative"
        )
    return roles


def _canonical_sample_key(sample: Mapping[str, Any], role: str) -> tuple[float, ...]:
    """优先 primary，再使用可用 expert metadata 和少步骤选择 canonical 样本。"""

    metadata = sample.get("metadata", {})
    score = metadata.get("expert_score", {})
    return (
        float(role == "primary"),
        float(score.get("success", metadata.get("trajectory_success", False))),
        float(score.get("final_recall", 0.0)),
        float(score.get("final_mrr", 0.0)),
        float(score.get("final_hit", 0.0)),
        float(score.get("total_reward", metadata.get("trajectory_reward", 0.0))),
        -float(score.get("steps", sample.get("step", 0))),
        -float(score.get("searches", 0)),
    )


def resolve_conflicting_supervision(
    samples: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """每个 identical prompt 只保留一个 canonical target/sample，不删其他历史不同样本。"""

    roles = _trajectory_roles(samples)
    groups: dict[str, list[Mapping[str, Any]]] = {}
    order: list[str] = []
    for sample in samples:
        digest = prompt_hash(sample)
        if digest not in groups:
            order.append(digest)
        groups.setdefault(digest, []).append(sample)
    retained: list[dict[str, Any]] = []
    removed_exact = removed_conflicting = 0
    for digest in order:
        group = groups[digest]
        targets = {_message_map(sample)["assistant"] for sample in group}
        canonical = max(
            group,
            key=lambda sample: _canonical_sample_key(
                sample, roles.get(str(sample.get("trajectory_id")), "alternative")
            ),
        )
        canonical_target = _message_map(canonical)["assistant"]
        same_target = [
            sample for sample in group if _message_map(sample)["assistant"] == canonical_target
        ]
        chosen = max(
            same_target,
            key=lambda sample: _canonical_sample_key(
                sample, roles.get(str(sample.get("trajectory_id")), "alternative")
            ),
        )
        copied = json.loads(json.dumps(chosen, ensure_ascii=False))
        copied.setdefault("metadata", {})["expert_role"] = roles.get(
            str(chosen.get("trajectory_id")), "alternative"
        )
        copied["metadata"]["supervision_resolution"] = "canonical_prompt_target_v1"
        retained.append(copied)
        removed_conflicting += sum(
            _message_map(sample)["assistant"] != canonical_target for sample in group
        )
        removed_exact += len(same_target) - 1
    return retained, {
        "input_samples": len(samples),
        "output_samples": len(retained),
        "deduplicated_count": len(samples) - len(retained),
        "removed_exact_duplicate_count": removed_exact,
        "removed_conflicting_target_count": removed_conflicting,
        "resolution_rule": "primary expert, then lexicographic expert metadata; keep one sample per prompt",
    }
