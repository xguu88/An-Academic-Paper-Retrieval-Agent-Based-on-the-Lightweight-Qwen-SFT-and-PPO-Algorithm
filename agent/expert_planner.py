"""在 RL train/dev truth 上离线选择高质量轨迹，但不把 truth 写入 prompt。"""

from dataclasses import asdict, dataclass
from typing import Any

from src.agent.baselines import (
    HybridOnlyPolicy,
    KeywordOnlyPolicy,
    KeywordThenSemanticPolicy,
    RuleBasedPolicy,
    SemanticOnlyPolicy,
    SemanticThenKeywordPolicy,
)
from src.agent.trajectory import AgentTrajectory
from src.environment.literature_env import LiteratureEnvironment
from src.evaluation.metrics import hit_at_k, mrr, recall_at_k
from src.evaluation.tasks import RetrievalTask


@dataclass(frozen=True, slots=True)
class ExpertPlannerConfig:
    """限制 alternative 数量和近似质量容差，避免重复样本爆炸。"""

    top_k: int = 10
    max_alternative_trajectories: int = 1
    recall_tolerance: float = 0.05
    mrr_tolerance: float = 0.05


@dataclass(frozen=True, slots=True)
class ExpertTrajectoryScore:
    """按检索质量优先、成本靠后的字典序保存 expert 分数。"""

    success: bool
    final_recall: float
    final_mrr: float
    final_hit: float
    total_reward: float
    steps: int
    searches: int

    def ranking_key(self) -> tuple[float, ...]:
        """返回 success→Recall→MRR→Hit@1→reward→少步骤→少搜索。"""

        return (
            float(self.success),
            self.final_recall,
            self.final_mrr,
            self.final_hit,
            self.total_reward,
            -float(self.steps),
            -float(self.searches),
        )

    def to_dict(self) -> dict[str, Any]:
        """转换为 JSON 兼容字典。"""

        return asdict(self)


@dataclass(slots=True)
class ExpertTrajectory:
    """关联策略名、稳定 trajectory_id、轨迹和检索质量分数。"""

    trajectory_id: str
    source_policy: str
    trajectory: AgentTrajectory
    score: ExpertTrajectoryScore


@dataclass(slots=True)
class ExpertPlan:
    """保存 primary、少量 alternative 以及需要改写的 hard-task 标记。"""

    task_id: str
    primary: ExpertTrajectory | None
    alternatives: list[ExpertTrajectory]
    needs_rewrite: bool
    attempted_scores: dict[str, ExpertTrajectoryScore]


class ExpertPlanner:
    """尝试五种短策略并按明确字典序选择 expert。"""

    def __init__(
        self,
        environment: LiteratureEnvironment,
        config: ExpertPlannerConfig | None = None,
    ) -> None:
        """绑定离线 Environment；不加载模型或外部 teacher。"""

        self.environment = environment
        self.config = config or ExpertPlannerConfig()
        self.policies: list[RuleBasedPolicy] = [
            KeywordOnlyPolicy(),
            SemanticOnlyPolicy(),
            HybridOnlyPolicy(),
            KeywordThenSemanticPolicy(),
            SemanticThenKeywordPolicy(),
        ]

    def _score(
        self,
        task: RetrievalTask,
        trajectory: AgentTrajectory,
    ) -> ExpertTrajectoryScore:
        """使用内部 truth 评价最终候选；truth 不写回 trajectory。"""

        final_state = trajectory.steps[-1].state_after
        ids = final_state.candidate_paper_ids
        relevant = set(task.relevant_paper_ids)
        return ExpertTrajectoryScore(
            success=bool(trajectory.success),
            final_recall=recall_at_k(ids, relevant, self.config.top_k),
            final_mrr=mrr(ids[: self.config.top_k], relevant),
            final_hit=hit_at_k(ids, relevant, 1),
            total_reward=float(trajectory.total_reward or 0.0),
            steps=len(trajectory.steps),
            searches=final_state.search_count,
        )

    def _near_quality(
        self,
        primary: ExpertTrajectoryScore,
        candidate: ExpertTrajectoryScore,
    ) -> bool:
        """alternative 必须成功状态相同、Hit@1 相同且 Recall/MRR 近似。"""

        return (
            candidate.success == primary.success
            and candidate.final_hit == primary.final_hit
            and primary.final_recall - candidate.final_recall <= self.config.recall_tolerance
            and primary.final_mrr - candidate.final_mrr <= self.config.mrr_tolerance
        )

    def plan(self, task: RetrievalTask) -> ExpertPlan:
        """运行五种合法策略；全部 Top-10 失败时标记 needs_rewrite 并不产 positive。,对五种策略进行打分排序，取出成功且打分策略最高的作为方案"""

        attempted: list[ExpertTrajectory] = []
        for policy in self.policies:
            trajectory = policy.run(self.environment, task, top_k=self.config.top_k)
            attempted.append(
                ExpertTrajectory(
                    trajectory_id=f"{task.task_id}:{policy.name}",
                    source_policy=policy.name,
                    trajectory=trajectory,
                    score=self._score(task, trajectory),
                )
            )
        scores = {item.source_policy: item.score for item in attempted}
        successful = [item for item in attempted if item.score.success]
        if not successful:
            return ExpertPlan(task.task_id, None, [], True, scores)
        ordered = sorted(successful, key=lambda item: item.score.ranking_key(), reverse=True)
        primary = ordered[0]
        alternatives = [
            item
            for item in ordered[1:]
            if item.source_policy != primary.source_policy
            and self._near_quality(primary.score, item.score)
        ][: self.config.max_alternative_trajectories]
        return ExpertPlan(task.task_id, primary, alternatives, False, scores)

