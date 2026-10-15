"""独立 PPO orchestration：collect、GAE、minibatch update、checkpoint/report。"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any

import torch

from src.evaluation.tasks import RetrievalTask
from src.training.ppo_buffer import PPORolloutBuffer
from src.training.ppo_loss import clipped_ppo_loss
from src.training.ppo_policy import PPOPolicyAdapter
from src.training.ppo_rollout import PPORolloutCollector
from src.training.ppo_types import PPOConfig, PPORolloutEpisode, PPOUpdateMetrics


def explained_variance(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """计算 1-var(target-prediction)/var(target)，常数 target 返回 0。"""

    target_variance = torch.var(targets, unbiased=False)
    if target_variance <= 1e-12:
        return 0.0
    return float(
        (
            1.0 - torch.var(targets - predictions, unbiased=False) / target_variance
        ).item()
    )


class PPOTrainer:
    """不依赖 SFT label 的 PPO trainer；KL 仅通过 rollout reward shaping 使用。"""

    def __init__(
        self,
        policy: PPOPolicyAdapter,
        collector: PPORolloutCollector,#计算GAE
        config: PPOConfig,
        *,
        pad_token_id: int = 0,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        self.policy = policy
        self.collector = collector
        self.config = config
        parameters = [
            parameter for parameter in policy.parameters() if parameter.requires_grad
        ]
        if not parameters:
            raise ValueError("PPO policy/value head 没有可训练参数")
        self.optimizer = optimizer or torch.optim.AdamW(
            parameters, lr=config.learning_rate
        )
        self.pad_token_id = pad_token_id
        self.generator = torch.Generator().manual_seed(config.seed)
        self.update_steps = 0

    @property
    def device(self) -> torch.device:
        """返回首个可训练参数设备。"""

        return next(
            parameter
            for parameter in self.policy.parameters()
            if parameter.requires_grad
        ).device

    def collect(self, tasks: Sequence[RetrievalTask]) -> list[PPORolloutEpisode]:
        """采集一个 task batch；不会读取任何 SFT expert action。"""

        if not tasks:
            raise ValueError("PPO rollout tasks 不能为空")
        return [self.collector.collect(task) for task in tasks]

    def update(self, episodes: Sequence[PPORolloutEpisode]) -> list[PPOUpdateMetrics]:
        """冻结 rollout old stats，执行多 epoch clipped PPO update。"""

        buffer = PPORolloutBuffer(self.config, self.pad_token_id)
        for episode in episodes:
            buffer.add_episode(episode)
        buffer.finalize()
        metrics: list[PPOUpdateMetrics] = []
        # PPO probability/value forward 始终使用 deterministic eval mode；
        # eval 不关闭 autograd，LoRA 与 FP32 value head 仍正常获得梯度。
        self.policy.eval()
        invariant = getattr(self.policy, "assert_reference_invariant", None)
        if invariant is not None:
            invariant()
        for _ in range(self.config.ppo_epochs):
            for batch in buffer.minibatches(generator=self.generator):
                batch = batch.to(self.device)
                new_logprobs, entropies, new_values = self.policy.evaluate_actions(#找到value的真实模型
                    batch.input_ids,
                    batch.attention_mask,
                    batch.response_mask,
                    batch.prompt_lengths,
                )
                loss = clipped_ppo_loss(#计算总损失函数
                    new_logprobs,
                    batch.old_logprobs,
                    batch.advantages,#徐advantage
                    new_values,#需要valuemodel
                    batch.old_values,
                    batch.returns,
                    entropies,
                    clip_range=self.config.clip_range,
                    value_clip_range=self.config.value_clip_range,
                    policy_coef=self.config.policy_coef,
                    value_coef=self.config.value_coef,
                    entropy_coef=self.config.entropy_coef,
                    log_ratio_clamp=self.config.log_ratio_clamp,
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.total_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [
                        parameter
                        for parameter in self.policy.parameters()
                        if parameter.requires_grad
                    ],
                    self.config.max_grad_norm,
                )
                if not torch.isfinite(torch.as_tensor(grad_norm)):
                    raise RuntimeError("PPO gradient norm 含 NaN/Inf")
                self.optimizer.step()
                if invariant is not None:
                    invariant()
                self.update_steps += 1
                metrics.append(
                    PPOUpdateMetrics(
                        total_loss=float(loss.total_loss.detach().item()),
                        policy_loss=float(loss.policy_loss.detach().item()),
                        value_loss=float(loss.value_loss.detach().item()),
                        entropy=float(loss.entropy.detach().item()),
                        approx_kl=float(loss.approx_kl.detach().item()),
                        ppo_log_ratio_mean=float(loss.log_ratio_mean.detach().item()),
                        ppo_ratio_mean=float(loss.ratio_mean.detach().item()),
                        clip_fraction=float(loss.clip_fraction.detach().item()),
                        gradient_norm=float(torch.as_tensor(grad_norm).item()),
                    )
                )
        return metrics

    def train_batch(self, tasks: Sequence[RetrievalTask]) -> dict[str, Any]:
        """完成一次 collect→GAE→PPO update 并返回审计指标。"""

        episodes = self.collect(tasks)#传入计算的各项结果tasks的reward，计算advantage、GAE
        updates = self.update(episodes)
        steps = [step for episode in episodes for step in episode.steps]
        update_dicts = [asdict(item) for item in updates]
        returns = torch.tensor([step.return_ for step in steps])
        values = torch.tensor([step.value for step in steps])
        reference_logprobs = [
            step.reference_logprob
            for step in steps
            if step.reference_logprob is not None
        ]
        reference_log_ratios = [
            step.reference_action_log_ratio
            for step in steps
            if step.reference_logprob is not None
        ]
        environment_reward_mean = mean(step.environment_reward for step in steps)
        ppo_reward_after_kl_mean = mean(step.total_ppo_reward for step in steps)
        return {
            "num_rollouts": len(steps),
            "num_episodes": len(episodes),
            "mean_environment_reward": environment_reward_mean,
            "mean_total_reward": ppo_reward_after_kl_mean,
            "environment_reward_mean": environment_reward_mean,
            "ppo_reward_after_kl_mean": ppo_reward_after_kl_mean,
            "mean_response_tokens": mean(sum(step.response_mask) for step in steps),
            "mean_action_logprob": mean(step.old_logprob for step in steps),
            "mean_reference_action_logprob": (
                mean(reference_logprobs) if reference_logprobs else None
            ),
            "reference_sample_log_ratio_mean": (
                mean(reference_log_ratios) if reference_log_ratios else 0.0
            ),
            "success_rate": mean(
                float(
                    bool(
                        episode.steps[-1]
                        .observation.get("events", {})
                        .get("success", False)
                    )
                )
                for episode in episodes
            ),
            "mean_steps": mean(len(episode.steps) for episode in episodes),
            "mean_searches": mean(
                episode.steps[-1].next_state["search_count"] for episode in episodes
            ),
            "invalid_action_rate": mean(float(not step.valid_action) for step in steps),
            "policy_loss": mean(item.policy_loss for item in updates),
            "value_loss": mean(item.value_loss for item in updates),
            "approx_kl": mean(item.approx_kl for item in updates),
            "ppo_update_approx_kl": mean(item.approx_kl for item in updates),
            "ppo_log_ratio_mean": mean(item.ppo_log_ratio_mean for item in updates),
            "ppo_ratio_mean": mean(item.ppo_ratio_mean for item in updates),
            "clip_fraction": mean(item.clip_fraction for item in updates),
            "entropy": mean(item.entropy for item in updates),
            "gradient_norm": mean(item.gradient_norm for item in updates),
            "explained_variance": explained_variance(values, returns),
            "optimizer_steps": self.update_steps,
            "updates": update_dicts,
            "frozen_test_read": False,
        }

    def save_checkpoint(self, output_dir: str | Path, report: dict[str, Any]) -> Path:
        """保存 adapter、value head、optimizer、配置和训练进度。"""

        target = Path(output_dir) / f"checkpoint-step-{self.update_steps}"
        target.mkdir(parents=True, exist_ok=True)
        if hasattr(self.policy, "save_checkpoint"):
            self.policy.save_checkpoint(target)  # type: ignore[attr-defined]
        torch.save(self.optimizer.state_dict(), target / "optimizer.pt")
        (target / "ppo_config.json").write_text(
            json.dumps(self.config.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (target / "training_progress.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return target


def write_ppo_training_report(report: dict[str, Any], output_dir: str | Path) -> Path:
    """写出正式 PPO 汇总报告并强制声明 frozen test 未读取。"""

    if report.get("frozen_test_read") is not False:
        raise ValueError("PPO report 必须显式记录 frozen_test_read=false")
    path = Path(output_dir) / "ppo_training_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path
