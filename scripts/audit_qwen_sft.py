"""执行 Qwen SFT tokenizer、长度与 assistant mask 的严格本地预检。"""

import importlib.metadata
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.qwen_loader import load_qwen_tokenizer
from src.training.sft_tokenization import (
    apply_qwen_chat_template,
    audit_assistant_masks,
    audit_token_lengths,
    chat_template_kwargs,
)


MODEL_PATH = Path("C:/Users/swukw/Desktop/agent_project/models/Qwen3-1.7B")
TRAIN_PATH = ROOT / "data/sft/scifact/train_v2.jsonl"
DEV_PATH = ROOT / "data/sft/scifact/dev_v2.jsonl"
RESULT_DIR = ROOT / "results/sft_generation"


def package_version(name: str) -> str | None:
    """读取已安装包版本，缺失时返回 None。"""

    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def load_jsonl(path: Path) -> list[dict]:
    """加载 SFT JSONL。"""

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def recommend_max_seq_length(audits: dict[str, dict]) -> dict:
    """依据 train/dev 的 P99 与 max 在候选长度中给出保守建议。"""

    p99 = max(report["total_tokens"]["p99"] for report in audits.values())
    maximum = max(report["total_tokens"]["max"] for report in audits.values())
    recommendation = next((value for value in (2048, 4096, 8192) if p99 <= value), None)
    return {
        "max_p99": p99,
        "max_observed": maximum,
        "recommended_max_seq_length": recommendation,
        "zero_truncation_at_recommendation": (
            recommendation is not None and maximum <= recommendation
        ),
        "note": "若仍有超长样本，应先缩减 visible candidate abstracts，不得截断 assistant target。",
    }


def write_blocked(reason: str, environment: dict) -> None:
    """保存明确的阻塞状态，绝不伪造 tokenizer 统计。"""

    blocked = {"status": "blocked", "reason": reason}
    (RESULT_DIR / "qwen_preflight.json").write_text(
        json.dumps({**environment, **blocked}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for filename in ("token_length_audit.json", "assistant_mask_audit.json"):
        (RESULT_DIR / filename).write_text(
            json.dumps(blocked, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def main() -> int:
    """只用本地模型文件运行 tokenizer 审计；模型缺失时安全停止。"""

    import transformers

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    environment = {
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "transformers_import_path": transformers.__file__,
        "peft": package_version("peft"),
        "trl": package_version("trl"),
        "accelerate": package_version("accelerate"),
        "bitsandbytes": package_version("bitsandbytes"),
        "cuda_available": torch.cuda.is_available(),
        "model_path": str(MODEL_PATH),
        "local_files_only": True,
    }
    if not MODEL_PATH.is_dir():
        write_blocked("QWEN MODEL NOT FOUND", environment)
        print("QWEN MODEL NOT FOUND")
        if not torch.cuda.is_available():
            print("FORMAL SFT TRAINING BLOCKED: CUDA unavailable")
        return 0
    tokenizer, model_info = load_qwen_tokenizer(MODEL_PATH)
    train, dev = load_jsonl(TRAIN_PATH), load_jsonl(DEV_PATH)
    template_kwargs, thinking_mode = chat_template_kwargs(tokenizer)
    rendered = apply_qwen_chat_template(
        tokenizer, train[0]["messages"], tokenize=False, add_generation_prompt=False
    )
    full_ids = apply_qwen_chat_template(
        tokenizer, train[0]["messages"], tokenize=True, add_generation_prompt=False
    )
    prompt_ids = apply_qwen_chat_template(
        tokenizer, train[0]["messages"][:2], tokenize=True, add_generation_prompt=True
    )
    if list(full_ids[: len(prompt_ids)]) != list(prompt_ids):
        raise RuntimeError("chat template assistant boundary 不一致")
    if train[0]["messages"][2]["content"] not in rendered:
        raise RuntimeError("chat template 未保留 assistant JSON target")
    if "<think>" in rendered or "</think>" in rendered:
        raise RuntimeError("non-thinking SFT template 仍产生 think 标签")
    token_audits = {
        "train_v2": audit_token_lengths(tokenizer, train),
        "dev_v2": audit_token_lengths(tokenizer, dev),
    }
    token_report = {
        "status": "pass",
        "model_info": model_info.to_dict(),
        "thinking_mode": thinking_mode,
        "chat_template_kwargs": template_kwargs,
        "first_rendered_chat": rendered,
        "chat_template_audit": {
            "system_boundary_present": train[0]["messages"][0]["content"] in rendered,
            "user_boundary_present": train[0]["messages"][1]["content"] in rendered,
            "assistant_boundary_token_index": len(prompt_ids),
            "assistant_target_present": train[0]["messages"][2]["content"] in rendered,
            "final_token_ids": list(full_ids[-8:]),
            "final_tokens_decoded": tokenizer.decode(
                full_ids[-8:], skip_special_tokens=False
            ),
            "tokenizer_eos_token_id": tokenizer.eos_token_id,
            "tokenizer_eos_token": tokenizer.eos_token,
        },
        "splits": token_audits,
        "decision": recommend_max_seq_length(token_audits),
    }
    mask_report = {
        "status": "pass",
        "train_v2": audit_assistant_masks(tokenizer, train, sample_count=20),
        "dev_v2": audit_assistant_masks(tokenizer, dev, sample_count=20),
    }
    (RESULT_DIR / "token_length_audit.json").write_text(
        json.dumps(token_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (RESULT_DIR / "assistant_mask_audit.json").write_text(
        json.dumps(mask_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (RESULT_DIR / "qwen_preflight.json").write_text(
        json.dumps({**environment, "status": "pass"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("QWEN TOKENIZER AUDIT PASS")
    if not torch.cuda.is_available():
        print("FORMAL SFT TRAINING BLOCKED: CUDA unavailable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
