"""从 train_v2 确定性抽取多动作 tiny overfit 数据，不访问 dev/test。"""

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SOURCE = ROOT / "data/sft/scifact/train_v2.jsonl"
OUTPUT = ROOT / "data/sft/scifact/tiny_overfit.jsonl"
MANIFEST = ROOT / "data/sft/scifact/tiny_overfit_manifest.json"
ACTION_QUOTAS = {
    "KeywordSearch": 4,
    "SemanticSearch": 4,
    "HybridSearch": 4,
    "Stop": 4,
}


def sha256_file(path: Path) -> str:
    """计算输入文件指纹。"""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    """仅从训练集按固定顺序抽取 16 条样本并保存来源清单。"""

    samples = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines()]
    selected: dict[str, list[dict]] = defaultdict(list)
    seen_tasks: set[str] = set()
    for sample in samples:
        action_type = sample.get("metadata", {}).get("action_type")
        if action_type not in ACTION_QUOTAS:
            continue
        task_id = str(sample.get("task_id"))
        if task_id in seen_tasks or len(selected[action_type]) >= ACTION_QUOTAS[action_type]:
            continue
        selected[action_type].append(sample)
        seen_tasks.add(task_id)
    missing = {
        action: ACTION_QUOTAS[action] - len(selected[action])
        for action in ACTION_QUOTAS
        if len(selected[action]) < ACTION_QUOTAS[action]
    }
    if missing:
        raise RuntimeError(f"训练集无法满足 tiny action quotas: {missing}")
    output_samples = [sample for action in ACTION_QUOTAS for sample in selected[action]]
    OUTPUT.write_text(
        "".join(json.dumps(sample, ensure_ascii=False) + "\n" for sample in output_samples),
        encoding="utf-8",
    )
    manifest = {
        "source": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "source_sha256": sha256_file(SOURCE),
        "output": str(OUTPUT.relative_to(ROOT)).replace("\\", "/"),
        "num_samples": len(output_samples),
        "action_distribution": {action: len(selected[action]) for action in ACTION_QUOTAS},
        "task_ids": [sample["task_id"] for sample in output_samples],
        "uses_dev": False,
        "uses_frozen_test": False,
        "uses_qrels_as_input": False,
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("TINY SFT DATA PREPARATION PASS")
    print(json.dumps(manifest["action_distribution"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
