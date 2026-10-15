"""Inference loop for local Qwen plus the ground-truth-free online environment."""

from __future__ import annotations

from time import perf_counter

from src.environment.online_literature_env import OnlineLiteratureEnvironment
from src.inference.online_prompt import (
    ONLINE_SYSTEM_PROMPT,
    invalid_output_retry_messages,
    serialize_online_agent_prompt,
)
from src.inference.policy import AgentPolicy
from src.inference.query_plan import QueryPlanner
from src.inference.result_organizer import organize_results
from src.inference.trace import InferenceResult, InferenceTraceStep


class OnlineInferenceRunner:
    def __init__(
        self,
        environment: OnlineLiteratureEnvironment,
        policy: AgentPolicy,
        *,
        model_name: str = "base",
        max_steps: int = 4,
        invalid_retries: int = 1,
        query_planner: QueryPlanner | None = None,
    ) -> None:
        self.environment = environment
        self.policy = policy
        self.model_name = model_name
        self.max_steps = min(max_steps, environment.max_steps)
        self.invalid_retries = min(max(invalid_retries, 0), 1)
        self.query_planner = query_planner or QueryPlanner()

    def run_query(self, query: str) -> InferenceResult:
        started = perf_counter()
        state, _ = self.environment.reset(query)#处理query格式转成AgentState
        query_plan_started = perf_counter()
        plan_result = self.query_planner.build(query, self.policy)#利用llm处理query格式转成action结果
        query_plan_latency_ms = (perf_counter() - query_plan_started) * 1000
        state.metadata["query_plan"] = plan_result.plan.to_dict()#融合结果与所有状态
        # The environment owns the mutable state; mirror visible planning metadata there.
        assert self.environment.state is not None
        self.environment.state.metadata["query_plan"] = plan_result.plan.to_dict()
        trace: list[InferenceTraceStep] = []
        latest: dict[str, object] | None = None
        invalid_count = 0
        retry_count = 0
        input_tokens = plan_result.input_tokens
        output_tokens = plan_result.output_tokens
        policy_generation_latency_ms = 0.0
        for index in range(self.max_steps):
            state.metadata["online_costs"] = {
                "openalex_requests": self.environment.openalex_requests,
                "serper_requests": self.environment.serper_requests,
                "citation_requests": self.environment.citation_requests,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "latency_ms_so_far": (perf_counter() - started) * 1000,
            }
            if self.environment.state is not None:
                self.environment.state.metadata["online_costs"] = state.metadata["online_costs"]
            user_prompt = serialize_online_agent_prompt(#输出整合信息后的提示词
                state, self.environment.final_candidates(), latest
            )
            messages = [
                {"role": "system", "content": ONLINE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            outputs: list[str] = []
            policy_started = perf_counter()
            decision = self.policy.generate_action(state, messages)#根据llm做出决策
            policy_generation_latency_ms += (perf_counter() - policy_started) * 1000
            if decision.input_tokens is not None:
                input_tokens = (input_tokens or 0) + decision.input_tokens
            if decision.output_tokens is not None:
                output_tokens = (output_tokens or 0) + decision.output_tokens
            outputs.append(decision.raw_text)
            first_error = decision.error
            if not decision.parsed_valid and self.invalid_retries:
                invalid_count += 1
                retry_count += 1
                policy_started = perf_counter()
                try:
                    decision = self.policy.generate_action(
                        state, invalid_output_retry_messages(messages, decision.raw_text)
                    )
                finally:
                    policy_generation_latency_ms += (perf_counter() - policy_started) * 1000
                if decision.input_tokens is not None:
                    input_tokens = (input_tokens or 0) + decision.input_tokens
                if decision.output_tokens is not None:
                    output_tokens = (output_tokens or 0) + decision.output_tokens
                outputs.append(decision.raw_text)
            if not decision.parsed_valid:
                invalid_count += 1
                state, observation, done = self.environment.abort_invalid_model_output()#执行policy的action输出最新state
                latest = observation.to_dict()
                trace.append(
                    InferenceTraceStep(
                        step=index + 1,
                        state_summary={
                            "original_query": state.original_query,
                            "candidate_count": len(self.environment.candidate_memory),
                        },
                        action={"action": "InvalidAction"},
                        raw_action_text=decision.raw_text,
                        observation=latest,
                        reward=0.0,
                        remaining_budget=observation.remaining_budget,
                        valid_action=False,
                        raw_model_outputs=outputs,
                        parse_error=decision.error or first_error,
                        retry_count=len(outputs) - 1,
                    )
                )
                if done:
                    break
            else:
                state_before = state
                state, observation, _, done, _ = self.environment.step(decision.action)
                latest = observation.to_dict()
                trace.append(
                    InferenceTraceStep(
                        step=index + 1,
                        state_summary={
                            "original_query": state_before.original_query,
                            "current_query": state_before.current_query,
                            "candidate_count": len(self.environment.candidate_memory),
                        },
                        action=decision.action.to_dict(),
                        raw_action_text=decision.raw_text,
                        observation=latest,
                        reward=0.0,
                        remaining_budget=observation.remaining_budget,
                        valid_action=observation.valid_action,
                        raw_model_outputs=outputs,
                        parse_error=first_error,
                        retry_count=len(outputs) - 1,
                    )
                )
                if done:
                    break
        candidates = [item.to_dict() for item in self.environment.final_candidates()]#academic search
        paper_count = sum(item["candidate_type"] == "paper" for item in candidates)
        resource_count = len(candidates) - paper_count
        relevance_counts = {
            level: sum(item.get("relevance_level") == level for item in candidates)
            for level in ("highly_relevant", "relevant", "partially_relevant")
        }
        total_api_requests = self.environment.serper_requests + self.environment.openalex_requests
        organizer_started = perf_counter()
        result_groups = organize_results(candidates)
        organizer_latency_ms = (perf_counter() - organizer_started) * 1000
        total_latency_ms = (perf_counter() - started) * 1000
        latency_breakdown = {
            "query_plan_latency_ms": query_plan_latency_ms,
            "policy_generation_latency_ms": policy_generation_latency_ms,
            "openalex_latency_ms": self.environment.openalex_latency_ms,
            "serper_latency_ms": self.environment.serper_latency_ms,
            "rerank_latency_ms": self.environment.rerank_latency_ms,
            "organizer_latency_ms": organizer_latency_ms,
            "total_latency_ms": total_latency_ms,
        }
        return InferenceResult(
            query=query.strip(),
            model=self.model_name,
            mode="online",
            terminated=state.terminated,
            trajectory=trace,
            final_candidates=candidates,
            metrics={
                "steps": len(trace),
                "searches": state.search_count,
                "serper_requests": self.environment.serper_requests,
                "openalex_requests": self.environment.openalex_requests,
                "citation_requests": self.environment.citation_requests,
                "semantic_scholar_requests": 0,
                "search_api_requests_total": total_api_requests,
                "candidate_count": len(candidates),
                "latency_ms": total_latency_ms,
                **latency_breakdown,
                "invalid_action_count": invalid_count,
                "invalid_retry_count": retry_count,
                "searches_before_stop": state.search_count,
                "unique_queries": len(state.seen_queries),
                "duplicate_query_count": self.environment.duplicate_query_count,
                "paper_candidate_count": paper_count,
                "related_resource_count": resource_count,
                "highly_relevant_count": relevance_counts["highly_relevant"],
                "relevant_count": relevance_counts["relevant"],
                "partially_relevant_count": relevance_counts["partially_relevant"],
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": (
                    input_tokens + output_tokens
                    if input_tokens is not None and output_tokens is not None else None
                ),
            },
            benchmark_success=None,
            latency_ms=total_latency_ms,
            query_plan=plan_result.plan.to_dict(),
            query_plan_diagnostics=plan_result.public_diagnostics(),
            result_groups=result_groups,
            cost_metrics={
                "model_input_tokens": input_tokens,
                "model_output_tokens": output_tokens,
                "serper_requests": self.environment.serper_requests,
                "openalex_requests": self.environment.openalex_requests,
                "citation_requests": self.environment.citation_requests,
                "search_api_requests_total": total_api_requests,
                "total_tokens": (
                    input_tokens + output_tokens
                    if input_tokens is not None and output_tokens is not None else None
                ),
                **latency_breakdown,
            },
        )
