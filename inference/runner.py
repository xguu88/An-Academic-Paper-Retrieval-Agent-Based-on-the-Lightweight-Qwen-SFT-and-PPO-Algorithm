"""Policy→Environment 推理 runner；完全不导入任何 Trainer。"""

from __future__ import annotations

from time import perf_counter
from uuid import uuid4

from src.agent.prompt_serializer import PromptSerializerConfig, serialize_agent_prompt
from src.agent.prompts import build_system_prompt
from src.environment.literature_env import LiteratureEnvironment
from src.evaluation.tasks import RetrievalTask
from src.inference.policy import AgentPolicy
from src.inference.trace import InferenceResult, InferenceTraceStep


class InferenceRunner:
    """执行在线或 benchmark task；在线结果不声明 ground-truth success。"""

    def __init__(
        self,
        environment: LiteratureEnvironment,
        policy: AgentPolicy,
        *,
        model_name: str,
        max_steps: int = 6,
        prompt_config: PromptSerializerConfig | None = None,
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps 必须大于 0")
        self.environment = environment
        self.policy = policy
        self.model_name = model_name
        self.max_steps = max_steps
        self.prompt_config = prompt_config or PromptSerializerConfig()

    def run_query(self, query: str) -> InferenceResult:
        """运行无 qrels 在线查询，benchmark_success 固定为 None。"""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query 必须是非空字符串")
        task = RetrievalTask(f"online-{uuid4().hex}", query.strip(), [])
        return self._run(task, expose_benchmark_success=False)

    def run_task(self, task: RetrievalTask) -> InferenceResult:
        """显式 benchmark 路径可返回 Environment truth success。"""

        return self._run(task, expose_benchmark_success=True)

    def _run(
        self, task: RetrievalTask, *, expose_benchmark_success: bool
    ) -> InferenceResult:
        started = perf_counter()
        state, _ = self.environment.reset(task)
        history: list[dict[str, object]] = []
        trace: list[InferenceTraceStep] = []
        total_reward = 0.0
        invalid_count = 0
        for index in range(self.max_steps):
            user_prompt = serialize_agent_prompt(
                state,
                history,
                self.environment.capabilities,
                self.environment.corpus,
                self.prompt_config,
            )
            messages = [
                {
                    "role": "system",
                    "content": build_system_prompt(self.environment.capabilities),
                },
                {"role": "user", "content": user_prompt},
            ]
            decision = self.policy.generate_action(state, messages)
            next_state, observation, reward, done, _ = self.environment.step(
                decision.action
            )
            observation_dict = observation.to_dict()
            if not expose_benchmark_success:
                observation_dict = self._sanitize_online_observation(observation_dict)
            valid = bool(decision.parsed_valid and observation.valid_action)
            invalid_count += int(not valid)
            total_reward += float(reward)
            trace.append(
                InferenceTraceStep(
                    step=index,
                    state_summary={
                        "original_query": state.original_query,
                        "current_query": state.current_query,
                        "candidate_count": len(state.candidate_paper_ids),
                    },
                    action=decision.action.to_dict(),
                    raw_action_text=decision.raw_text,
                    observation=observation_dict,
                    reward=float(reward),
                    remaining_budget=dict(observation.remaining_budget),
                    valid_action=valid,
                )
            )
            history.append(observation_dict)
            state = next_state
            if done:
                break
        candidates = []
        for rank, paper_id in enumerate(state.candidate_paper_ids, start=1):
            paper = self.environment.corpus.get(paper_id)
            if paper is not None:
                candidates.append(
                    {
                        "paper_id": paper.paper_id,
                        "title": paper.title,
                        "abstract": paper.abstract or "",
                        "rank": rank,
                    }
                )
        benchmark_success = None
        if expose_benchmark_success and self.environment.trajectory is not None:
            benchmark_success = bool(self.environment.trajectory.success)
        return InferenceResult(
            query=task.question,
            model=self.model_name,
            terminated=bool(state.terminated),
            trajectory=trace,
            final_candidates=candidates,
            metrics={
                "steps": len(trace),
                "searches": state.search_count,
                "reads": state.read_count,
                "total_environment_reward": total_reward,
                "invalid_action_count": invalid_count,
            },
            benchmark_success=benchmark_success,
            latency_ms=(perf_counter() - started) * 1000,
        )

    @staticmethod
    def _sanitize_online_observation(
        observation: dict[str, object],
    ) -> dict[str, object]:
        """在线 trace 删除 Environment 内部 truth success 及其文字暗示。"""

        copied = dict(observation)
        events = (
            dict(copied.get("events", {}))
            if isinstance(copied.get("events"), dict)
            else {}
        )
        events.pop("success", None)
        copied["events"] = events
        if copied.get("action_type") == "Stop":
            copied["message"] = "agent stopped"
        return copied
