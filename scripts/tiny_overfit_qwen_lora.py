"""运行固定 tiny 数据的真实 Qwen LoRA 端到端过拟合门禁。"""

import argparse
import gc
import json
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.sft_generation_eval import evaluate_sft_generations
from src.models.qwen_loader import load_qwen_causal_lm, load_qwen_tokenizer
from src.training.lora_utils import (
    inject_lora,
    inspect_lora_target_modules,
    validate_lora_trainability,
)
from src.training.sft_collator import SFTDataCollator
from src.training.sft_dataset import SFTDataset
from src.training.sft_tokenization import apply_qwen_chat_template


MODEL_PATH = Path("C:/Users/swukw/Desktop/agent_project/models/Qwen3-1.7B")
DATA_PATH = ROOT / "data/sft/scifact/tiny_overfit.jsonl"
OUTPUT_DIR = ROOT / "checkpoints/qwen3-1.7b/sft_tiny"


def generate_actions(model, tokenizer, samples: list[dict], device: torch.device) -> list[str]:
    """只解码新生成 token，避免把 prompt 错计为模型输出。"""

    model.eval()
    outputs: list[str] = []
    for sample in samples:
        prompt_ids = apply_qwen_chat_template(
            tokenizer,
            sample["messages"][:2],
            tokenize=True,
            add_generation_prompt=True,
        )
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_tokens = generated[0, input_ids.shape[1] :].tolist()
        outputs.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    return outputs


def train_fixed_steps(
    model,
    loader: DataLoader,
    *,
    device: torch.device,
    steps: int,
    learning_rate: float,
) -> list[float]:
    """使用固定 step 数训练，不以测试结果动态延长训练。"""

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=0.0,
    )
    losses: list[float] = []
    iterator = iter(loader)
    model.train()
    for _step in range(steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = {key: value.to(device) for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        output = model(**batch)
        loss = output.loss
        if not torch.isfinite(loss) or float(loss) <= 0:
            raise RuntimeError("tiny training loss 不是有限正数")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return losses


def main() -> int:
    """验证 train/backward/save/reload/generate 全链，并保存真实门禁结果。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    args = parser.parse_args()
    print("TINY QWEN LORA OVERFIT TEST")
    if not MODEL_PATH.is_dir():
        print("QWEN MODEL NOT FOUND")
        print("TINY QWEN LORA OVERFIT TEST BLOCKED")
        return 0
    try:
        from peft import PeftModel
    except ImportError:
        print("PEFT NOT INSTALLED")
        print("TINY QWEN LORA OVERFIT TEST BLOCKED")
        return 0
    samples = [json.loads(line) for line in DATA_PATH.read_text(encoding="utf-8").splitlines()]
    tokenizer, _ = load_qwen_tokenizer(MODEL_PATH)
    dataset = SFTDataset(DATA_PATH, tokenizer, max_seq_length=4096)
    collator = SFTDataCollator(tokenizer.pad_token_id)
    generator = torch.Generator().manual_seed(42)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        generator=generator,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    base_model, model_info = load_qwen_causal_lm(MODEL_PATH, torch_dtype=dtype)
    base_model.to(device)
    module_audit = inspect_lora_target_modules(base_model)
    if module_audit.missing_targets:
        raise RuntimeError(f"LoRA targets missing: {module_audit.missing_targets}")
    before_outputs = generate_actions(base_model, tokenizer, samples, device)
    before_metrics = evaluate_sft_generations(before_outputs, samples).to_dict()
    model = inject_lora(base_model)
    parameter_stats = validate_lora_trainability(model)
    losses = train_fixed_steps(
        model,
        loader,
        device=device,
        steps=args.steps,
        learning_rate=args.learning_rate,
    )
    after_outputs = generate_actions(model, tokenizer, samples, device)
    after_metrics = evaluate_sft_generations(after_outputs, samples).to_dict()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    del model, base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    reload_base, _ = load_qwen_causal_lm(MODEL_PATH, torch_dtype=dtype)
    reload_model = PeftModel.from_pretrained(
        reload_base,
        str(OUTPUT_DIR),
        local_files_only=True,
    ).to(device)
    reloaded_outputs = generate_actions(reload_model, tokenizer, samples, device)
    reload_consistent = reloaded_outputs == after_outputs
    window = max(1, min(5, len(losses) // 2))
    initial_mean = sum(losses[:window]) / window
    final_mean = sum(losses[-window:]) / window
    loss_declined = math.isfinite(final_mean) and final_mean < initial_mean * 0.8
    gates = {
        "loss_declined_by_at_least_20_percent": loss_declined,
        "json_parse_rate_at_least_90_percent": after_metrics["json_parse_rate"] >= 0.90,
        "action_validation_rate_at_least_90_percent": after_metrics["action_validation_rate"] >= 0.90,
        "action_type_accuracy_improved": (
            after_metrics["action_type_accuracy"] > before_metrics["action_type_accuracy"]
        ),
        "adapter_reload_generation_consistent": reload_consistent,
    }
    report = {
        "status": "pass" if all(gates.values()) else "gate_failed",
        "model_path": str(MODEL_PATH),
        "local_files_only": True,
        "device": str(device),
        "model_info": model_info.to_dict(),
        "module_audit": module_audit.to_dict(),
        "parameter_stats": parameter_stats,
        "training": {
            "fixed_steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "loss_curve": losses,
            "initial_window_mean": initial_mean,
            "final_window_mean": final_mean,
        },
        "before_generation": before_metrics,
        "after_generation": after_metrics,
        "reload_generation": evaluate_sft_generations(reloaded_outputs, samples).to_dict(),
        "gates": gates,
        "formal_sft_allowed": all(gates.values()) and torch.cuda.is_available(),
        "frozen_test_used": False,
        "qrels_used_as_model_input": False,
    }
    (OUTPUT_DIR / "training_metadata.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(gates, ensure_ascii=False, indent=2))
    print("TINY QWEN LORA OVERFIT TEST PASS" if all(gates.values()) else "TINY QWEN LORA OVERFIT GATE FAILED")
    if not torch.cuda.is_available():
        print("FORMAL SFT TRAINING BLOCKED: CUDA unavailable")
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
