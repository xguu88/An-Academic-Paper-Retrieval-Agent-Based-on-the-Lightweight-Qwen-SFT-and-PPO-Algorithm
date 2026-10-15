"""从冻结的 SciFact train 确定性拆分 RL train/dev，绝不改写 test。"""

import hashlib
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.tasks import load_tasks_jsonl  # noqa: E402


PROCESSED = PROJECT_ROOT / "data" / "benchmarks" / "scifact" / "processed"
SOURCE_TRAIN = PROCESSED / "train_tasks.jsonl"
SOURCE_TEST = PROCESSED / "test_tasks.jsonl"
RL_TRAIN = PROCESSED / "rl_train_tasks.jsonl"
RL_DEV = PROCESSED / "rl_dev_tasks.jsonl"
MANIFEST = PROCESSED / "rl_split_manifest.json"
DEV_RATIO = 0.15
SEED = 42


def _file_sha256(path: Path) -> str:
    """计算文件字节级 SHA-256，用于证明原始 test 未被修改。"""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _split_key(task_id: str) -> str:
    """使用 seed+task_id 稳定哈希，避免依赖平台随机数实现。"""

    return hashlib.sha256(f"{SEED}:{task_id}".encode("utf-8")).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    """以稳定 UTF-8 JSONL 写入拆分结果。"""

    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    """生成 RL split、检查三组 ID 互斥并保存 manifest。"""

    test_hash_before = _file_sha256(SOURCE_TEST)
    train_tasks = load_tasks_jsonl(SOURCE_TRAIN)
    test_tasks = load_tasks_jsonl(SOURCE_TEST)
    ordered = sorted(train_tasks, key=lambda task: (_split_key(task.task_id), task.task_id))
    dev_count = round(len(ordered) * DEV_RATIO)
    dev_tasks = ordered[:dev_count]
    rl_train_tasks = ordered[dev_count:]
    _write_jsonl(RL_TRAIN, [task.to_dict() for task in rl_train_tasks])
    _write_jsonl(RL_DEV, [task.to_dict() for task in dev_tasks])

    train_ids = {task.task_id for task in rl_train_tasks}
    dev_ids = {task.task_id for task in dev_tasks}
    test_ids = {task.task_id for task in test_tasks}
    intersections = {
        "train_dev": sorted(train_ids & dev_ids),
        "train_test": sorted(train_ids & test_ids),
        "dev_test": sorted(dev_ids & test_ids),
    }
    if any(intersections.values()):
        raise RuntimeError(f"RL split 发生 task_id 泄漏: {intersections}")
    if _file_sha256(SOURCE_TEST) != test_hash_before:
        raise RuntimeError("冻结的 test_tasks.jsonl 被意外修改")

    manifest = {
        "source_train_count": len(train_tasks),
        "rl_train_count": len(rl_train_tasks),
        "rl_dev_count": len(dev_tasks),
        "test_count": len(test_tasks),
        "dev_ratio": DEV_RATIO,
        "seed": SEED,
        "split_method": "sha256(seed:task_id), ascending; first round(N*dev_ratio) to dev",
        "source_train_sha256": _file_sha256(SOURCE_TRAIN),
        "test_sha256": test_hash_before,
        "disjoint": {name: len(values) == 0 for name, values in intersections.items()},
    }
    MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("SCIFACT RL SPLIT PASS")
    print(f"source_train={len(train_tasks)} rl_train={len(rl_train_tasks)} rl_dev={len(dev_tasks)}")
    print(f"test={len(test_tasks)} unchanged_sha256={test_hash_before}")


if __name__ == "__main__":
    main()

