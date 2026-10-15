"""Qwen3-1.7B SFT adapter → LoRA PPO 正式入口；本脚本默认不自动启动训练。"""

from __future__ import annotations

import argparse
import copy
import os
import random
import sys
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmarks.runtime import build_retrievers
from src.environment.capabilities import EnvironmentCapabilities
from src.environment.config import EnvironmentConfig
from src.environment.literature_env import LiteratureEnvironment
from src.evaluation.tasks import load_tasks_jsonl
from src.models.qwen_loader import (
    load_qwen_causal_lm,
    load_qwen_tokenizer,
)
from src.retrieval.corpus import PaperCorpus
from src.retrieval.hybrid_retriever import HybridConfig, HybridRetriever
from src.training.lora_utils import validate_lora_trainability
from src.training.ppo_policy import QwenPPOPolicy
from src.training.ppo_rollout import PPORolloutCollector
from src.training.ppo_trainer import PPOTrainer, write_ppo_training_report
from src.training.ppo_types import PPOConfig
from src.training.ppo_value import ScalarValueHead, value_parameter_report


def build_parser() -> argparse.ArgumentParser:
    """构建显式 CLI；当前本地 gate 应使用 smoke_ppo_pipeline.py。"""

    parser = argparse.ArgumentParser(description="Qwen3 LoRA PPO training")
    parser.add_argument("--config", default="configs/ppo_qwen3_1_7b.yaml")
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="本地短跑必须设置为 1；A100 正式训练可省略",
    )
    parser.add_argument("--base-model-path", default=None)
    parser.add_argument("--sft-adapter-path", default=None)
    parser.add_argument("--train-tasks", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _require_rl_train(path: Path) -> None:
    """硬拒绝 frozen test、dev 和任意非 RL train 输入。"""

    if path.name != "rl_train_tasks.jsonl":
        raise PermissionError(
            "PPO 正式训练只允许 rl_train_tasks.jsonl，禁止 frozen test/dev"
        )
    if not path.is_file():
        raise FileNotFoundError(path)


def _apply_cli_overrides(
    config: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    """应用正式机器显式路径覆盖，不创建模型或 checkpoint。"""

    selected = copy.deepcopy(config)
    overrides = (
        ("base_model_path", getattr(args, "base_model_path", None), selected["model"]),
        (
            "sft_adapter_path",
            getattr(args, "sft_adapter_path", None),
            selected["model"],
        ),
        ("train_tasks", getattr(args, "train_tasks", None), selected["data"]),
        ("output_dir", getattr(args, "output_dir", None), selected["training"]),
    )
    for key, value, section in overrides:
        if value is not None:
            section[key] = value
    return selected


def _preflight(config: dict[str, Any]) -> dict[str, str]:
    """在加载 GPU 模型前验证数据、路径和 strict PPO 数学配置。"""

    errors: list[str] = []
    model = config.get("model", {})
    data = config.get("data", {})
    training = config.get("training", {})
    generation = config.get("generation", {})
    base_raw = model.get("base_model_path")
    adapter_raw = model.get("sft_adapter_path")
    task_raw = data.get("train_tasks")
    output_raw = training.get("output_dir")
    if not isinstance(base_raw, str) or not base_raw.strip():
        errors.append("model.base_model_path must be explicitly provided")
    if not isinstance(adapter_raw, str) or not adapter_raw.strip():
        errors.append("model.sft_adapter_path must be explicitly provided")
    if not isinstance(task_raw, str) or not task_raw.strip():
        errors.append("data.train_tasks must be explicitly provided")
    if not isinstance(output_raw, str) or not output_raw.strip():
        errors.append("training.output_dir must be explicitly provided")
    base_path = _resolve(base_raw or "__missing_base_model__")
    adapter_path = _resolve(adapter_raw or "__missing_sft_adapter__")
    task_path = _resolve(task_raw or "__missing_rl_train_tasks__")
    output_path = _resolve(output_raw or "__missing_ppo_output__")

    if not base_path.is_dir():
        errors.append(f"Base model path does not exist: {base_path}")
    if not adapter_path.is_dir():
        errors.append(
            f"SFT best_adapter does not exist; run formal SFT first: {adapter_path}"
        )
    try:
        _require_rl_train(task_path)
    except (PermissionError, FileNotFoundError) as exc:
        errors.append(str(exc))
    if output_path.exists() and not output_path.is_dir():
        errors.append(f"PPO output path is not a directory: {output_path}")
    output_parent = output_path
    while not output_parent.exists() and output_parent != output_parent.parent:
        output_parent = output_parent.parent
    if not output_parent.is_dir() or not os.access(output_parent, os.W_OK):
        errors.append(f"PPO output parent is not writable: {output_parent}")

    expected_generation = {
        "do_sample": True,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
    }
    for key, expected in expected_generation.items():
        if generation.get(key) != expected:
            errors.append(
                f"generation.{key} must be {expected!r}, got {generation.get(key)!r}"
            )
    if int(generation.get("max_new_tokens", 0)) <= 0:
        errors.append("generation.max_new_tokens must be positive")
    if training.get("action_logprob_reduction") != "sum":
        errors.append("training.action_logprob_reduction must be 'sum'")
    use_reference = bool(training.get("use_reference_policy", True))
    model_reference = model.get("use_reference_policy")
    if model_reference is not None and bool(model_reference) != use_reference:
        errors.append("model/training use_reference_policy settings disagree")
    if use_reference and not adapter_path.is_dir():
        errors.append("reference policy requires the same valid SFT best_adapter")

    if errors:
        message = "PPO PREFLIGHT FAILED\n- " + "\n- ".join(errors)
        print(message, flush=True)
        raise RuntimeError(message)
    print("PPO PREFLIGHT PASS", flush=True)
    return {
        "base_model_path": str(base_path),
        "sft_adapter_path": str(adapter_path),
        "train_tasks": str(task_path),
        "output_dir": str(output_path),
    }


def _load_policy(
    config: dict[str, Any], device: torch.device
) -> tuple[QwenPPOPolicy, dict[str, Any]]:
    """加载两个独立 SFT adapter 分支：trainable policy 与 frozen reference。
    设置tokenizer将prompt转成id，然后设置reference、policy、value model的架构
    """

    from peft import PeftModel

    model_config = config["model"]
    base_path = _resolve(model_config["base_model_path"])
    adapter_path = _resolve(model_config["sft_adapter_path"])
    if not base_path.is_dir():
        raise FileNotFoundError(f"Base model path does not exist: {base_path}")
    if not adapter_path.is_dir():
        raise FileNotFoundError(
            f"SFT best_adapter does not exist; run formal SFT first: {adapter_path}"
        )
    tokenizer, model_info = load_qwen_tokenizer(base_path)
    dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    policy_base, _ = load_qwen_causal_lm(base_path, dtype=dtype)#构建本地可配置的模型
    policy_model = PeftModel.from_pretrained(#设置可训练的参数（sft原来的与value新的）
        policy_base, adapter_path, is_trainable=True
    )
    policy_model.to(device)
    lora_stats = validate_lora_trainability(policy_model)#验证梯度

    ppo_config = PPOConfig.from_mapping(config["training"])#训练配置参数
    reference_model = None
    if ppo_config.use_reference_policy:
        reference_base, _ = load_qwen_causal_lm(base_path, dtype=dtype)#构建reference模型
        reference_model = PeftModel.from_pretrained(
            reference_base, adapter_path, is_trainable=False
        )
        reference_model.to(device).eval()
        for parameter in reference_model.parameters():
            parameter.requires_grad_(False)#原地设置参数不需要计算梯度。
    hidden_size = model_info.hidden_size
    if hidden_size is None:
        raise ValueError("Qwen config 缺少 hidden_size")
    value_head = ScalarValueHead(hidden_size).to(device=device, dtype=torch.float32)#设置最后的value头，将模型最后的token转成标量
    generation = config.get("generation", {})
    policy = QwenPPOPolicy(#组合模型，设置训练流程，输入prompt得到prompt/action然后计算概率，并且计算预计收益与reference模型的概率
        policy_model,
        tokenizer,
        value_head,
        reference_model=reference_model,
        max_new_tokens=int(generation.get("max_new_tokens", 128)),
        do_sample=bool(generation.get("do_sample", True)),
        temperature=float(generation.get("temperature", 1.0)),
        top_p=float(generation.get("top_p", 1.0)),
        top_k=int(generation.get("top_k", 0)),
    )
    return policy, {
        "model_info": model_info.to_dict(),
        "lora": lora_stats,
        "parameters": value_parameter_report(policy_model, value_head),
        "reference_frozen": bool(
            reference_model is None
            or all(
                not parameter.requires_grad
                for parameter in reference_model.parameters()
            )
        ),
        "dtype": str(dtype).replace("torch.", ""),
    }


def _build_environment(config: dict[str, Any]) -> LiteratureEnvironment:
    """复用现有 Retrieval/Environment，不引入 PaSa 环境实现。"""


    """
    _build_environment() 的作用不是训练模型，而是为 PPO 创建“训练场”：把 SciFact 论文库、
    BM25/Dense/Hybrid 检索器、Action能力、预算和 Reward 规则封装进 LiteratureEnvironment，
    PPO Policy 在这个环境里执行 Action、获得 State 转移和 Reward，再用这些 trajectory 做 PPO 更新。
    """

    processed = ROOT / "data" / "benchmarks" / "scifact" / "processed"
    corpus = PaperCorpus.from_jsonl(processed / "papers.jsonl")
    retrieval_mapping = yaml.safe_load(
        _resolve(config["retrieval_config"]).read_text(encoding="utf-8")
    )
    retrievers, _ = build_retrievers(
        corpus,
        retrieval_mapping,
        processed / "indexes" / "dense",
        requested={"bm25", "dense"},
    )
    rrf = retrieval_mapping["rrf"]
    hybrid = HybridRetriever(
        retrievers["bm25"],
        retrievers["dense"],
        HybridConfig(
            bm25_top_n=rrf.get("bm25_top_n", 50),
            dense_top_n=rrf.get("dense_top_n", 50),
            rrf_k=rrf.get("k", 60),
            final_top_k=rrf.get("final_top_k", 20),
        ),
    )
    environment_config = EnvironmentConfig.from_mapping(
        yaml.safe_load(
            _resolve(config["environment_config"]).read_text(encoding="utf-8")
        )
    )
    return LiteratureEnvironment(
        corpus,
        retrievers["bm25"],
        retrievers["dense"],
        hybrid_retriever=hybrid,
        config=environment_config,
        capabilities=EnvironmentCapabilities.scifact(),
    )




def run(args: argparse.Namespace) -> dict[str, Any]:
    """按 rollout task batch 训练并保存 adapter/value/optimizer/report。"""

    config_path = _resolve(args.config)#传入参数
    config = _apply_cli_overrides(
        yaml.safe_load(config_path.read_text(encoding="utf-8")), args
    )#取出路径
    preflight = _preflight(config)
    ppo_config = PPOConfig.from_mapping(config["training"])#从yaml文件读取配置
    if args.max_batches is not None and args.max_batches <= 0:
        raise ValueError("max_batches 必须为正整数")
    task_path = _resolve(config["data"]["train_tasks"])
    _require_rl_train(task_path)
    tasks = load_tasks_jsonl(task_path)
    random.seed(ppo_config.seed)
    np.random.seed(ppo_config.seed)
    torch.manual_seed(ppo_config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, model_report = _load_policy(config, device)#输出改进后的policy模型
    environment = _build_environment(config)#初始化LiteratureEnvironment
    trainer = PPOTrainer(#初始化训练、评估函数
        policy,
        PPORolloutCollector(environment, policy, ppo_config),
        ppo_config,
        pad_token_id=policy.tokenizer.pad_token_id,
    )
    output_dir = _resolve(config["training"]["output_dir"])
    batch_reports = []
    for batch_index, start in enumerate(
        range(0, len(tasks), ppo_config.rollout_tasks_per_batch), start=1
    ):
        if args.max_batches is not None and batch_index > args.max_batches:
            break
        task_batch = tasks[start : start + ppo_config.rollout_tasks_per_batch]
        report = trainer.train_batch(task_batch)#更新审计指标
        batch_reports.append(report)
        trainer.save_checkpoint(output_dir, report)
        print(
            f"batch={batch_index} episodes={report['num_episodes']} "
            f"reward={report['mean_total_reward']:.6f} kl={report['approx_kl']:.6f}",
            flush=True,
        )
    if not batch_reports:
        raise RuntimeError("没有执行任何 PPO batch")
    best_dir = output_dir / "best_adapter"
    best_dir.mkdir(parents=True, exist_ok=True)
    policy.policy_model.save_pretrained(best_dir)
    policy.tokenizer.save_pretrained(best_dir)
    torch.save(policy.value_head.state_dict(), best_dir / "value_head.pt")
    final_report = {
        "config": ppo_config.to_dict(),
        "model": model_report,
        "preflight": preflight,
        "num_batches": len(batch_reports),
        "num_episodes": sum(item["num_episodes"] for item in batch_reports),
        "num_rollouts": sum(item["num_rollouts"] for item in batch_reports),
        "optimizer_steps": trainer.update_steps,
        "mean_environment_reward": mean(
            item["mean_environment_reward"] for item in batch_reports
        ),
        "mean_total_reward": mean(item["mean_total_reward"] for item in batch_reports),
        "environment_reward_mean": mean(
            item["environment_reward_mean"] for item in batch_reports
        ),
        "ppo_reward_after_kl_mean": mean(
            item["ppo_reward_after_kl_mean"] for item in batch_reports
        ),
        "mean_response_tokens": mean(
            item["mean_response_tokens"] for item in batch_reports
        ),
        "mean_action_logprob": mean(
            item["mean_action_logprob"] for item in batch_reports
        ),
        "mean_reference_action_logprob": mean(
            item["mean_reference_action_logprob"]
            for item in batch_reports
            if item["mean_reference_action_logprob"] is not None
        )
        if any(
            item["mean_reference_action_logprob"] is not None for item in batch_reports
        )
        else None,
        "reference_sample_log_ratio_mean": mean(
            item["reference_sample_log_ratio_mean"] for item in batch_reports
        ),
        "success_rate": mean(item["success_rate"] for item in batch_reports),
        "mean_steps": mean(item["mean_steps"] for item in batch_reports),
        "mean_searches": mean(item["mean_searches"] for item in batch_reports),
        "policy_loss": mean(item["policy_loss"] for item in batch_reports),
        "value_loss": mean(item["value_loss"] for item in batch_reports),
        "approx_kl": mean(item["approx_kl"] for item in batch_reports),
        "ppo_update_approx_kl": mean(
            item["ppo_update_approx_kl"] for item in batch_reports
        ),
        "ppo_log_ratio_mean": mean(
            item["ppo_log_ratio_mean"] for item in batch_reports
        ),
        "ppo_ratio_mean": mean(item["ppo_ratio_mean"] for item in batch_reports),
        "clip_fraction": mean(item["clip_fraction"] for item in batch_reports),
        "entropy": mean(item["entropy"] for item in batch_reports),
        "explained_variance": mean(
            item["explained_variance"] for item in batch_reports
        ),
        "gradient_norm": mean(item["gradient_norm"] for item in batch_reports),
        "invalid_action_rate": mean(
            item["invalid_action_rate"] for item in batch_reports
        ),
        "batch_reports": batch_reports,
        "best_adapter": str(best_dir),
        "seed": ppo_config.seed,
        "frozen_test_read": False,
    }
    write_ppo_training_report(final_report, output_dir)
    return final_report


def main() -> int:
    args = build_parser().parse_args()
    report = run(args)
    print(f"PPO TRAINING COMPLETE: episodes={report['num_episodes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
