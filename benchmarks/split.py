"""保留显式 benchmark split，或生成带 seed 的确定性 train/dev/test split。"""

import random
from collections.abc import Mapping, Sequence

from src.benchmarks.types import BenchmarkSplitResult
from src.evaluation.tasks import RetrievalTask


def split_tasks(
    tasks: Sequence[RetrievalTask],
    *,
    explicit_splits: Mapping[str, str] | None = None,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> BenchmarkSplitResult:
    """优先保留原 split；否则按排序后任务和固定 seed 确定性拆分。"""

    if not tasks:
        raise ValueError("tasks 不能为空")
    if explicit_splits is not None:
        task_ids = {task.task_id for task in tasks}
        if set(explicit_splits) != task_ids:
            raise ValueError("explicit_splits 必须覆盖且只能覆盖全部 task_id")
        invalid = {value for value in explicit_splits.values() if value not in {"train", "dev", "test"}}
        if invalid:
            raise ValueError(f"存在非法 split: {sorted(invalid)}")
        buckets = {"train": [], "dev": [], "test": []}
        for task in tasks:
            buckets[explicit_splits[task.task_id]].append(task)
        return BenchmarkSplitResult(**buckets, source="original", seed=None)

    if len(ratios) != 3 or any(ratio < 0 for ratio in ratios) or abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("ratios 必须是和为 1 的三个非负数")
    ordered = sorted(tasks, key=lambda task: task.task_id)
    random.Random(seed).shuffle(ordered)
    count = len(ordered)
    train_count = int(count * ratios[0])
    dev_count = int(count * ratios[1])
    if count >= 3:
        dev_count = max(1, dev_count)
        test_count = max(1, count - train_count - dev_count)
        train_count = count - dev_count - test_count
    else:
        test_count = count - train_count - dev_count
    train = ordered[:train_count]
    dev = ordered[train_count : train_count + dev_count]
    test = ordered[train_count + dev_count :]
    return BenchmarkSplitResult(train, dev, test, source="generated", seed=seed)

