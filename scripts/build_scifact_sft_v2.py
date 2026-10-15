"""清理 identical prompt 的 exact duplicate/conflict，保留原始 SFT JSONL。"""

import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.tasks import load_tasks_jsonl  # noqa: E402
from src.training.sft_audit import (  # noqa: E402
    analyze_conflicting_supervision,
    audit_sft_samples,
    resolve_conflicting_supervision,
    sha256_file,
)
from src.training.sft_manifest import build_sft_manifest  # noqa: E402


DATA = PROJECT_ROOT / "data" / "sft" / "scifact"
PROCESSED = PROJECT_ROOT / "data" / "benchmarks" / "scifact" / "processed"
RESULTS = PROJECT_ROOT / "results" / "sft_generation"


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    """读取 UTF-8 SFT JSONL。"""

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    """写入新的 v2 JSONL，绝不覆盖原始 train/dev。"""

    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    """分析原数据、按 primary 优先消冲突并重新执行全部 SFT audit。"""

    RESULTS.mkdir(parents=True, exist_ok=True)
    conflict_report: dict[str, object] = {}
    outputs: dict[str, Path] = {}
    sample_counts: dict[str, int] = {}
    retained_task_ids: dict[str, set[str]] = {}
    for split, source_name in (("train", "rl_train_tasks.jsonl"), ("dev", "rl_dev_tasks.jsonl")):
        original_path = DATA / f"{split}.jsonl"
        original = _read_jsonl(original_path)
        before = analyze_conflicting_supervision(original)
        resolved, resolution = resolve_conflicting_supervision(original)
        after = analyze_conflicting_supervision(resolved)
        tasks = load_tasks_jsonl(PROCESSED / source_name)
        audit = audit_sft_samples(resolved, tasks)
        if (
            after["conflicting_prompt_groups"] != 0
            or after["exact_duplicate_sample_count"] != 0
            or audit.parse_success_rate != 1.0
            or audit.validation_success_rate != 1.0
            or audit.leakage_count != 0
        ):
            raise RuntimeError(f"{split} v2 gate failed")
        output = DATA / f"{split}_v2.jsonl"
        _write_jsonl(output, resolved)
        outputs[split] = output
        sample_counts[split] = len(resolved)
        retained_task_ids[split] = {str(sample["task_id"]) for sample in resolved}
        conflict_report[split] = {
            "before": before,
            "resolution": resolution,
            "after": after,
            "audit": audit.to_dict(),
        }
    overlap = retained_task_ids["train"] & retained_task_ids["dev"]
    if overlap:
        raise RuntimeError(f"SFT v2 train/dev task overlap: {len(overlap)}")
    conflict_report["train_dev_task_overlap"] = 0
    conflict_report["frozen_test_used"] = False
    (RESULTS / "conflicting_supervision.json").write_text(
        json.dumps(conflict_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = build_sft_manifest(
        train_source=PROCESSED / "rl_train_tasks.jsonl",
        dev_source=PROCESSED / "rl_dev_tasks.jsonl",
        train_output=outputs["train"],
        dev_output=outputs["dev"],
        reward_config=PROJECT_ROOT / "configs" / "environment.yaml",
        retrieval_config=PROJECT_ROOT / "configs" / "retrieval.yaml",
        num_train_samples=sample_counts["train"],
        num_dev_samples=sample_counts["dev"],
    )
    manifest.update(
        {
            "dataset_version": "sft-v2-conflict-resolved",
            "parent_manifest_sha256": sha256_file(DATA / "sft_manifest.json"),
            "conflicting_prompt_groups": 0,
            "exact_duplicate_prompt_target_pairs": 0,
            "train_dev_task_overlap": 0,
        }
    )
    (DATA / "sft_manifest_v2.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("SCIFACT SFT V2 BUILD PASS")
    print(f"train_v2={sample_counts['train']} dev_v2={sample_counts['dev']}")


if __name__ == "__main__":
    main()

