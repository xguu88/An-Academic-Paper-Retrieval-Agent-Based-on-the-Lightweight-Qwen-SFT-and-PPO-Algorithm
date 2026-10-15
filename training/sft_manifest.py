"""生成可追溯 SFT dataset manifest 与配置/数据 fingerprint。"""

from pathlib import Path
from typing import Any

from src.agent.prompt_serializer import PROMPT_SERIALIZER_VERSION
from src.agent.prompts import SYSTEM_PROMPT_VERSION
from src.training.sft_audit import sha256_file


ACTION_SCHEMA_VERSION = "action-protocol-v2-hybrid-search"


def build_sft_manifest(
    *,
    train_source: Path,
    dev_source: Path,
    train_output: Path,
    dev_output: Path,
    reward_config: Path,
    retrieval_config: Path,
    num_train_samples: int,
    num_dev_samples: int,
) -> dict[str, Any]:
    """记录输入、配置、serializer/schema 版本以及生成 JSONL 的 SHA-256。"""

    return {
        "train_task_source_fingerprint": sha256_file(train_source),
        "dev_task_source_fingerprint": sha256_file(dev_source),
        "prompt_serializer_version": PROMPT_SERIALIZER_VERSION,
        "system_prompt_version": SYSTEM_PROMPT_VERSION,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "reward_config_fingerprint": sha256_file(reward_config),
        "retrieval_config_fingerprint": sha256_file(retrieval_config),
        "num_train_samples": num_train_samples,
        "num_dev_samples": num_dev_samples,
        "train_data_sha256": sha256_file(train_output),
        "dev_data_sha256": sha256_file(dev_output),
        "frozen_test_used": False,
    }

