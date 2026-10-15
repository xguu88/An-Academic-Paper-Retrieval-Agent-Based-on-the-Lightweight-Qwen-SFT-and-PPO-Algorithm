"""真实 Qwen + LoRA SFT 链路 smoke；默认测试不会调用本脚本。"""

import gc
import json
import math
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.qwen_loader import load_qwen_causal_lm, load_qwen_tokenizer
from src.training.lora_utils import (
    inject_lora,
    inspect_lora_target_modules,
    validate_lora_trainability,
)
from src.training.sft_collator import SFTDataCollator
from src.training.sft_dataset import SFTDataset
from src.training.sft_tokenization import apply_qwen_chat_template, tokenize_sft_example


MODEL_PATH = Path("C:/Users/swukw/Desktop/agent_project/models/Qwen3-1.7B")
TINY_PATH = ROOT / "data/sft/scifact/tiny_overfit.jsonl"


def pass_line(message: str) -> None:
    """输出统一的 smoke 成功行。"""

    print(f"[PASS] {message}")


def main() -> int:
    """运行到 backward 的最小真实链路；缺少模型或 PEFT 时明确阻塞。"""

    print("SFT PIPELINE SMOKE TEST")
    if not MODEL_PATH.is_dir():
        print("QWEN MODEL NOT FOUND")
        print("SFT PIPELINE SMOKE TEST BLOCKED")
        if not torch.cuda.is_available():
            print("FORMAL SFT TRAINING BLOCKED: CUDA unavailable")
        return 0
    try:
        import peft  # noqa: F401
    except ImportError:
        print("PEFT NOT INSTALLED")
        print("SFT PIPELINE SMOKE TEST BLOCKED")
        return 0
    tokenizer, _ = load_qwen_tokenizer(MODEL_PATH)
    pass_line("Tokenizer loaded")
    sample = json.loads(TINY_PATH.read_text(encoding="utf-8").splitlines()[0])
    rendered = apply_qwen_chat_template(
        tokenizer, sample["messages"], tokenize=False, add_generation_prompt=False
    )
    if not isinstance(rendered, str) or sample["messages"][2]["content"] not in rendered:
        raise RuntimeError("chat template 未包含 assistant target")
    pass_line("Chat template")
    encoded = tokenize_sft_example(tokenizer, sample, max_seq_length=4096)
    pass_line("Token length")
    if not any(label != -100 for label in encoded.labels):
        raise RuntimeError("assistant labels 为空")
    pass_line("Assistant mask")
    dataset = SFTDataset(TINY_PATH, tokenizer, max_seq_length=4096)
    pass_line("Dataset")
    collator = SFTDataCollator(tokenizer.pad_token_id)
    batch = collator([dataset[0], dataset[1]])
    pass_line("Collator")
    model, _ = load_qwen_causal_lm(MODEL_PATH, torch_dtype=torch.float32)
    model.to("cpu")
    pass_line("Model loaded")
    model.eval()
    with torch.no_grad():
        output = model(**batch)
    if not math.isfinite(float(output.loss)) or float(output.loss) <= 0:
        raise RuntimeError("forward loss 非有限正数")
    if not torch.isfinite(output.logits).all():
        raise RuntimeError("forward logits 非有限值")
    pass_line("Forward loss finite")
    audit = inspect_lora_target_modules(model)
    if audit.missing_targets:
        raise RuntimeError(f"LoRA targets missing: {audit.missing_targets}")
    model = inject_lora(model)
    pass_line("LoRA injected")
    validate_lora_trainability(model)
    pass_line("Trainable params")
    model.train()
    train_output = model(**batch)
    train_output.loss.backward()
    pass_line("Backward")
    del model
    gc.collect()
    command = [
        sys.executable,
        str(ROOT / "scripts/tiny_overfit_qwen_lora.py"),
        "--steps",
        "30",
    ]
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode != 0:
        print("SFT PIPELINE SMOKE TEST FAILED: tiny overfit gate")
        return completed.returncode
    pass_line("Adapter save/load")
    pass_line("Action generation validation")
    print("SFT PIPELINE SMOKE TEST PASS")
    if not torch.cuda.is_available():
        print("FORMAL SFT TRAINING BLOCKED: CUDA unavailable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
