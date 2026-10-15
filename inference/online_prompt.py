"""Prompt serialization for online academic search and citation expansion."""

from __future__ import annotations

import json
from collections.abc import Sequence

from src.agent.state import AgentState
from src.retrieval.online.models import OnlinePaperCandidate


ONLINE_SYSTEM_PROMPT = """你是一个学术论文搜索 Agent。
目标是通过搜索 query 找到与用户研究问题高度相关的真实学术论文。
候选论文只能来自工具返回的在线搜索结果；禁止凭模型记忆编造论文。
Before choosing Stop, internally evaluate whether the currently retrieved evidence is sufficient for the user's original research question.
Check all of the following without printing private reasoning:
1. Relevance: Are the current papers directly relevant to the original question?
2. Academic quality: Are there enough results from credible academic sources?
3. Coverage: Do the candidates cover the main concepts in the original question?
4. Diversity and duplicates: Are results meaningfully different rather than duplicate pages or copies of the same paper?
5. Search improvement potential: Would a new, more specific or complementary query retrieve substantially better evidence?
6. Search cost: Additional searches consume API calls, tokens, latency, and budget; search again only when expected information gain is meaningful.
Choose Stop only when the current evidence is sufficiently useful. Otherwise output KeywordSearch with an improved query that has not already been used.
There is no required minimum number of searches: strong, diverse, directly relevant academic evidence may justify Stop after one search.
当前只允许三个已有 AgentAction：
- {"action":"KeywordSearch","query":"string","top_k":"optional positive integer"}
- {"action":"ExpandCitation","paper_id":"candidate id","direction":"references or citations"}
- {"action":"Stop"}
只输出一个严格 JSON AgentAction，不得输出解释、Markdown、推理过程或 self-check 字段。"""


def _visible_latest_observation(
    observation: dict[str, object] | None,
) -> dict[str, object] | None:
    """Keep tool outcome summaries while excluding full raw result payloads."""

    if observation is None:
        return None
    allowed = (
        "action_type",
        "query_used",
        "results_count",
        "candidates_added",
        "message",
        "remaining_budget",
        "terminated",
        "valid_action",
        "error",
    )
    return {key: observation[key] for key in allowed if key in observation}


def serialize_online_agent_prompt(
    state: AgentState,
    candidates: Sequence[OnlinePaperCandidate],
    latest_observation: dict[str, object] | None,
) -> str:
    plan = state.metadata.get("query_plan")
    plan_dict = plan if isinstance(plan, dict) else {}
    planned_subqueries = [
        item for item in plan_dict.get("subqueries", []) if isinstance(item, str)
    ]
    executed = {item.casefold().strip() for item in state.seen_queries}
    pending_subqueries = [
        item for item in planned_subqueries if item.casefold().strip() not in executed
    ]
    explicit_constraints = {
        key: plan_dict.get(key, [])
        for key in (
            "core_concepts", "methods", "datasets", "domains",
            "time_constraints", "venue_constraints", "other_constraints",
        )
    }
    relevance_distribution = {
        level: sum(item.relevance_level == level for item in candidates)
        for level in ("highly_relevant", "relevant", "partially_relevant")
    }
    relevance_distribution["unscored"] = sum(
        item.relevance_level is None for item in candidates
    )
    evidence_summary = {
        "candidate_count": len(candidates),
        "paper_count": sum(item.candidate_type == "paper" for item in candidates),
        "distinct_domains": len({item.domain for item in candidates}),
        "relevance_distribution": relevance_distribution,
        "explicit_query_constraints": explicit_constraints,
        "planned_subqueries": planned_subqueries,
        "executed_queries": state.seen_queries,
        "planned_subqueries_not_yet_executed": pending_subqueries,
    }
    costs = state.metadata.get("online_costs")
    costs_dict = costs if isinstance(costs, dict) else {}
    max_searches = int(state.metadata.get("max_searches", 3))
    max_serper = int(state.metadata.get("max_serper_requests", 0))
    max_openalex = int(state.metadata.get("max_openalex_requests", 0))
    visible = [
        {
            "id": item.id,
            "title": item.title,
            "domain": item.domain,
            "candidate_type": item.candidate_type,
            "snippet": item.snippet[:350],
            "query_used": item.query_used,
            "recommended_rank": item.recommended_rank,
            "search_rank": item.search_rank,
            "authors": item.authors[:5],
            "publication_year": item.publication_year,
            "venue": item.venue,
            "citation_count": item.citation_count,
            "relevance_level": item.relevance_level,
            "relevance_score": item.relevance_score,
            "openalex_id": item.external_ids.get("openalex_id"),
        }
        for item in candidates[:8]
    ]
    return "\n".join(
        [
            f"Original research question: {state.original_query}",
            f"Current query: {state.current_query}",
            f"Step: {state.step}/{state.max_steps}",
            "Structured query plan: "
            + json.dumps(plan, ensure_ascii=False),
            "Coverage and evidence summary: "
            + json.dumps(evidence_summary, ensure_ascii=False),
            "Current online costs: "
            + json.dumps(costs, ensure_ascii=False),
            "Remaining budget: "
            + json.dumps(
                {
                    "steps": state.remaining_steps,
                    "searches_used": state.search_count,
                    "maximum_searches": max_searches,
                    "searches_remaining": max(0, max_searches - state.search_count),
                    "maximum_serper_requests": max_serper,
                    "serper_requests_remaining": max(
                        0, max_serper - int(costs_dict.get("serper_requests", 0) or 0)
                    ),
                    "maximum_openalex_requests": max_openalex,
                    "openalex_requests_remaining": max(
                        0, max_openalex - int(costs_dict.get("openalex_requests", 0) or 0)
                    ),
                },
                ensure_ascii=False,
            ),
            "Search queries already used: " + json.dumps(state.seen_queries, ensure_ascii=False),
            "Previous actions: "
            + json.dumps(
                [action.to_dict() for action in state.action_history],
                ensure_ascii=False,
            ),
            "Cumulative candidates from actual search results: "
            + json.dumps(visible, ensure_ascii=False),
            "Latest tool observation: "
            + json.dumps(_visible_latest_observation(latest_observation), ensure_ascii=False),
            "Internally assess whether the evidence covers every explicit QueryPlan constraint. "
            "Treat planned_subqueries_not_yet_executed as opportunities, not mandatory steps. "
            "If a complementary query could add meaningful information within the remaining budget, "
            "choose KeywordSearch; otherwise choose Stop.",
        ]
    )


def invalid_output_retry_messages(
    messages: Sequence[dict[str, str]], raw_output: str
) -> list[dict[str, str]]:
    return [
        *messages,
        {"role": "assistant", "content": raw_output},
        {
            "role": "user",
            "content": (
                "Your previous output was invalid. Return ONLY one valid JSON action. "
                "Use KeywordSearch with a non-empty query, ExpandCitation for an "
                "OpenAlex candidate, or Stop."
            ),
        },
    ]
