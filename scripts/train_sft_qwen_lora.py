"""使用 Qwen3-1.7B 与 LoRA 训练正式 assistant-only Action SFT。"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent.parser import parse_action
from src.evaluation.sft_generation_eval import evaluate_sft_generations
from src.models.qwen_loader import load_qwen_causal_lm, load_qwen_tokenizer
from src.training.lora_utils import (
    DEFAULT_LORA_TARGETS,
    inject_lora,
    inspect_lora_target_modules,
    validate_lora_trainability,
)
from src.training.sft_collator import SFTDataCollator
from src.training.sft_dataset import SFTDataset
from src.training.sft_tokenization import apply_qwen_chat_template


ACTION_TYPES = ("KeywordSearch", "SemanticSearch", "HybridSearch", "Stop")
DEFAULT_TRAIN_FILE = "data/sft/scifact/train_v2.jsonl"
DEFAULT_DEV_FILE = "data/sft/scifact/dev_v2.jsonl"


def build_parser() -> argparse.ArgumentParser:
    """创建正式训练 CLI，不绑定 Windows 或 Colab 模型路径。"""

    parser = argparse.ArgumentParser(
        description="Qwen3-1.7B assistant-only LoRA SFT 正式训练入口"
    )
    parser.add_argument("--model-path", required=True, help="本地 Qwen3-1.7B 目录")
    parser.add_argument("--train-file", default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--dev-file", default=DEFAULT_DEV_FILE)
    parser.add_argument("--output-dir", default="checkpoints/qwen3-1.7b/sft_formal")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-seq-length", type=int, default=2304)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="限制全程 optimizer steps；仅用于正式训练前短跑",
    )
    parser.add_argument(
        "--eval-max-samples",
        type=int,
        default=None,
        help="仅供短跑限制 dev generation 样本数；正式训练应省略",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    """在加载模型前拒绝无效训练参数。"""

    positive = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "max_seq_length": args.max_seq_length,
    }
    if any(value <= 0 for value in positive.values()):
        raise ValueError(f"训练参数必须为正数: {positive}")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("max_steps 必须为正整数")
    if args.eval_max_samples is not None and args.eval_max_samples <= 0:
        raise ValueError("eval_max_samples 必须为正整数")


def validate_formal_data_paths(
    train_file: str | Path,
    dev_file: str | Path,
) -> tuple[Path, Path]:
    """只接受正式 v2 train/dev，硬拒绝 frozen test 或角色互换。"""

    train_path = Path(train_file).expanduser().resolve()
    dev_path = Path(dev_file).expanduser().resolve()
    if train_path.name != "train_v2.jsonl":
        raise PermissionError("formal train 只允许 train_v2.jsonl，禁止 frozen test")
    if dev_path.name != "dev_v2.jsonl":
        raise PermissionError("formal dev 只允许 dev_v2.jsonl，禁止 frozen test")
    if train_path == dev_path:
        raise ValueError("train 和 dev 必须严格分开")
    if not train_path.is_file() or not dev_path.is_file():
        raise FileNotFoundError("train_v2.jsonl 或 dev_v2.jsonl 不存在")
    return train_path, dev_path


def validate_split_isolation(train_samples: Sequence[dict], dev_samples: Sequence[dict]) -> None:
    """检查 task_id 零重叠，避免训练阶段污染 dev。"""

    train_ids = {str(sample.get("task_id")) for sample in train_samples}
    dev_ids = {str(sample.get("task_id")) for sample in dev_samples}
    overlap = train_ids & dev_ids
    if overlap:
        raise ValueError(f"train/dev task overlap: {sorted(overlap)[:10]}")


def select_training_dtype(cuda_available: bool) -> torch.dtype:
    """Tesla T4/CUDA 固定 FP16，只有 CPU fallback 使用 FP32。"""

    return torch.float16 if cuda_available else torch.float32


def configure_model_for_sft(model: Any) -> dict[str, bool]:
    """启用梯度检查点、关闭 cache，并按能力启用 input grads。"""

    model.config.use_cache = False #**推理生成专用 KV‑Cache**。每一步把 key/value 缓存，加速 auto‑regressive 生成。
    model.gradient_checkpointing_enable()#开启梯度检查点，节约显存
    input_grads_enabled = False
    enable_input_grads = getattr(model, "enable_input_require_grads", None)
    #让**embedding 输出的 input_embeds 开启`.requires_grad=True`**，保证梯度可以正常回传，修复梯度检查点下梯度断流问题。

    if callable(enable_input_grads):
        enable_input_grads()
        input_grads_enabled = True
    return {
        "use_cache_disabled": model.config.use_cache is False,
        "gradient_checkpointing_enabled": True,
        "input_require_grads_enabled": input_grads_enabled,
    }


def checkpoint_score(metrics: dict[str, Any]) -> tuple[float, float, float]:
    """按 validation→action type→exact match 的优先级生成比较键。"""

    return (
        float(metrics["action_validation_rate"]),
        float(metrics["action_type_accuracy"]),
        float(metrics["exact_target_match_rate"]),
    )


def is_better_checkpoint(
    candidate: dict[str, Any],
    best: dict[str, Any] | None,
) -> bool:
    """判断 candidate 是否严格优于当前 best。"""

    return best is None or checkpoint_score(candidate) > checkpoint_score(best)


def generation_metrics_by_action(
    generated_texts: Sequence[str],
    samples: Sequence[dict],
) -> dict[str, dict[str, Any]]:
    """复用统一 evaluator，分别统计四类正式 Action。"""

    result: dict[str, dict[str, Any]] = {}
    for action_type in ACTION_TYPES:
        indexes = [
            index
            for index, sample in enumerate(samples)
            if parse_action(sample["messages"][2]["content"]).action.value == action_type
        ]
        if not indexes:
            result[action_type] = {"num_samples": 0}
            continue
        subset_outputs = [generated_texts[index] for index in indexes]
        subset_samples = [samples[index] for index in indexes]
        result[action_type] = evaluate_sft_generations(
            subset_outputs, subset_samples
        ).to_dict()
    return result


def generate_actions(
    model: Any,
    tokenizer: Any,
    samples: Sequence[dict],
    device: torch.device,
) -> list[str]:
    """逐条贪心生成，只解码 prompt 后的新 token。"""

    model.eval()
    generated_texts: list[str] = []
    for sample in samples:
        prompt_ids = apply_qwen_chat_template(
            tokenizer,
            sample["messages"][:-1],
            tokenize=True,
            add_generation_prompt=True,
        )
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)#padding位置掩码
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_ids = output_ids[0, input_ids.shape[1] :].tolist()
        generated_texts.append(
            tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        )
    return generated_texts


def evaluate_dev_loss(
    model: Any,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """只使用 dev labels 计算 assistant-only 平均 loss。"""

    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss
            loss_value = loss.detach().item()
            if not np.isfinite(loss_value):
                raise RuntimeError("dev loss 非有限值")
            losses.append(loss_value)
    if not losses:
        raise ValueError("dev loader 为空")
    return float(np.mean(losses))


def save_adapter_checkpoint(model: Any, tokenizer: Any, path: Path) -> None:
    """只保存 LoRA adapter 与必要 tokenizer/config 文件。"""

    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


def serialize_training_report(report: dict[str, Any]) -> str:
    """把只含 JSON 类型的训练报告稳定序列化。"""

    return json.dumps(report, ensure_ascii=False, indent=2)


def write_training_report(report: dict[str, Any], output_dir: Path) -> Path:
    """以 UTF-8 JSON 保存可复现实验报告。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "training_report.json"
    path.write_text(serialize_training_report(report), encoding="utf-8")
    return path


def _epoch_evaluation(
    model: Any,
    tokenizer: Any,
    dev_loader: DataLoader,
    dev_samples: list[dict],
    device: torch.device,
) -> dict[str, Any]:
    """统一计算 dev loss、全局生成指标和四类 Action 指标。"""

    dev_loss = evaluate_dev_loss(model, dev_loader, device)
    generated = generate_actions(model, tokenizer, dev_samples, device)#model生成的文本
    overall = evaluate_sft_generations(generated, dev_samples).to_dict()
    return {
        "dev_loss": dev_loss,
        **overall,
        "by_action": generation_metrics_by_action(generated, dev_samples),
    }


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    """执行正式 LoRA 训练；max_steps 可将其限制为短跑。"""

    validate_args(args)
    train_path, dev_path = validate_formal_data_paths(args.train_file, args.dev_file)#拿scifact的json格式之后的数据
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
    tokenizer, _ = load_qwen_tokenizer(args.model_path)#加载分词模型
    train_dataset = SFTDataset(
        train_path, tokenizer, max_seq_length=args.max_seq_length
    )#数据格式转化，提取关键格式
    dev_dataset = SFTDataset(dev_path, tokenizer, max_seq_length=args.max_seq_length)
    validate_split_isolation(train_dataset.samples, dev_dataset.samples)#检查数据是否泄露
    evaluated_dev_samples = list(dev_dataset.samples)
    if args.eval_max_samples is not None:
        evaluated_dev_samples = evaluated_dev_samples[: args.eval_max_samples]
    collator = SFTDataCollator(tokenizer.pad_token_id)#统一长度，padding
    data_generator = torch.Generator().manual_seed(args.seed)
    training_loader_dataset: Any = train_dataset
    training_shuffle = True
    if args.max_steps is not None:
        # 短跑只取达到 optimizer-step 上限所需的文件头样本，避免随机抽到
        # 极长样本而把工程 dry-run 误变成长时间 CPU 作业。
        required_samples = (
            args.max_steps
            * args.gradient_accumulation_steps
            * args.batch_size
        )
        training_loader_dataset = torch.utils.data.Subset(
            train_dataset, range(min(required_samples, len(train_dataset)))
        )#截取数据集
        training_shuffle = False
    train_loader = DataLoader(#用来训练和更新参数
        training_loader_dataset,
        batch_size=args.batch_size,
        shuffle=training_shuffle,
        collate_fn=collator,
        generator=data_generator,
    )#加载数据，确定数据输入的方式
    evaluated_dev_dataset = dev_dataset
    if args.eval_max_samples is not None:
        evaluated_dev_dataset = torch.utils.data.Subset(
            dev_dataset, range(len(evaluated_dev_samples))
        )
    dev_loader = DataLoader(#验证数据
        evaluated_dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = select_training_dtype(device.type == "cuda")
    model, model_info = load_qwen_causal_lm(args.model_path, dtype=dtype)#加载基模
    model.to(device)
    model_flags = configure_model_for_sft(model) #开启梯度检查点与关闭kv-cache，
    #KV Cache 是推理加速用的，训练时基本没必要，而且会和 gradient checkpointing 冲突。
    module_audit = inspect_lora_target_modules(model)#训练前检查“我要改的这些参数，模型里到底存不存在”。
    if module_audit.missing_targets:
        raise RuntimeError(f"LoRA targets missing: {module_audit.missing_targets}")
    model = inject_lora(model, rank=16, alpha=32, dropout=0.05)#设定可训练参数
    parameter_stats = validate_lora_trainability(model)#验证开启可训练参数的梯度
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,#把开启梯度更新交给优化器，作用是更新参数（不计算loss）
        weight_decay=0.01,
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    history: list[dict[str, Any]] = []
    optimizer_steps = 0
    best_metrics: dict[str, Any] | None = None
    best_checkpoint: str | None = None
    stop_training = False
    optimizer.zero_grad(set_to_none=True)#清空之前的累计梯度
    for epoch in range(1, args.epochs + 1):
        model.train()#把模型切换到训练模式
        epoch_losses: list[float] = []
        accumulated = 0
        for batch_index, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}#把张量搬运到 GPU 设备上
            loss = model(**batch).loss#计算损失函数，采用crossentropy
            loss_value = loss.detach().item()#把这个张量**从计算图剥离**，不再跟踪梯度。不再参与反向传播。
            if not np.isfinite(loss_value) or loss_value <= 0:
                raise RuntimeError("train loss 不是有限正数")
            (loss / args.gradient_accumulation_steps).backward()#根据计算图计算梯度，不更新参数
            epoch_losses.append(loss_value)
            accumulated += 1
            is_last_batch = batch_index == len(train_loader)
            if accumulated == args.gradient_accumulation_steps or is_last_batch:#损失累计
                optimizer.step()#更新参数
                optimizer.zero_grad(set_to_none=True)#继续清除梯度
                optimizer_steps += 1
                accumulated = 0
                if args.max_steps is not None and optimizer_steps >= args.max_steps:
                    stop_training = True
                    break
        epoch_metrics = _epoch_evaluation(
            model,
            tokenizer,
            dev_loader,
            evaluated_dev_samples,
            device,
        )
        epoch_record = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)),
            "optimizer_steps_total": optimizer_steps,
            **epoch_metrics,
        }
        history.append(epoch_record)
        checkpoint = output_dir / f"checkpoint-epoch-{epoch}"
        save_adapter_checkpoint(model, tokenizer, checkpoint)
        if is_better_checkpoint(epoch_metrics, best_metrics):
            best_metrics = epoch_metrics
            best_path = output_dir / "best_adapter"
            save_adapter_checkpoint(model, tokenizer, best_path)
            best_checkpoint = str(best_path)
        report = {
            "model": str(Path(args.model_path).expanduser()),
            "model_info": model_info.to_dict(),
            "dtype": str(dtype).replace("torch.", ""),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "train_samples": len(train_dataset),
            "short_run_train_samples": len(training_loader_dataset),
            "dev_samples": len(dev_dataset),
            "evaluated_dev_samples": len(evaluated_dev_samples),
            "max_seq_length": args.max_seq_length,
            "lora": {
                "r": 16,
                "alpha": 32,
                "dropout": 0.05,
                "targets": list(DEFAULT_LORA_TARGETS),
                "parameter_stats": parameter_stats,
            },
            "epochs_requested": args.epochs,
            "epochs_completed": len(history),
            "optimizer_steps": optimizer_steps,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "history": history,
            "train_losses": [item["train_loss"] for item in history],
            "dev_metrics": [
                {
                    key: value
                    for key, value in item.items()
                    if key not in {"train_loss"}
                }
                for item in history
            ],
            "best_checkpoint": best_checkpoint,
            "best_checkpoint_score": (
                list(checkpoint_score(best_metrics)) if best_metrics else None
            ),
            "peak_vram_bytes": (
                torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            ),
            "seed": args.seed,
            "model_flags": model_flags,
            "frozen_test_read": False,
            "train_file": str(train_path),
            "dev_file": str(dev_path),
            "max_steps": args.max_steps,
        }
        write_training_report(report, output_dir)
        print(
            f"epoch={epoch} train_loss={epoch_record['train_loss']:.6f} "
            f"dev_loss={epoch_metrics['dev_loss']:.6f} "
            f"validation={epoch_metrics['action_validation_rate']:.6f} "
            f"action_type={epoch_metrics['action_type_accuracy']:.6f} "
            f"exact={epoch_metrics['exact_target_match_rate']:.6f}"
        )
        if stop_training:
            break
    return report


def main() -> int:
    """解析 CLI 并启动受路径守卫保护的正式训练。"""

    args = build_parser().parse_args()
    report = run_training(args)
    print(f"TRAINING REPORT: {Path(args.output_dir) / 'training_report.json'}")
    print(f"FROZEN TEST READ: {report['frozen_test_read']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
