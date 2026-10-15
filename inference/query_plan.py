"""Strict, concise query understanding for online inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
from typing import Any

from src.inference.policy import QwenAgentPolicy


LOGGER = logging.getLogger(__name__)
QUERY_PLAN_MAX_NEW_TOKENS = 384


QUERY_PLAN_SYSTEM_PROMPT = """You analyze complex academic literature questions.
Return ONLY one strict JSON object with exactly these keys:
original_query, research_intent, core_concepts, methods, datasets, domains,
time_constraints, venue_constraints, other_constraints, subqueries.
original_query and research_intent are strings; all other fields are arrays of short strings.
Copy the user's full question into original_query and write a non-empty concise research_intent.
Extract every explicit concept and constraint in the user's question. For a composite question,
provide concise complementary subqueries that cover its distinct aspects. Use an empty array only
when that category is genuinely absent. Do not wrap the JSON in Markdown.
Do not provide reasoning."""

QUERY_PLAN_EXAMPLE_USER = (
    "Find papers on graph neural networks for molecular property prediction "
    "using the QM9 dataset, published after 2020."
)
QUERY_PLAN_EXAMPLE_ASSISTANT = json.dumps(
    {
        "original_query": QUERY_PLAN_EXAMPLE_USER,
        "research_intent": "Find recent graph neural network research on QM9 molecular property prediction",
        "core_concepts": ["graph neural networks", "molecular property prediction"],
        "methods": ["graph neural networks"],
        "datasets": ["QM9"],
        "domains": ["molecular machine learning"],
        "time_constraints": ["after 2020"],
        "venue_constraints": [],
        "other_constraints": [],
        "subqueries": [
            "graph neural networks QM9 molecular property prediction after 2020",
            "recent GNN molecular property prediction QM9 benchmark",
        ],
    },
    ensure_ascii=False,
    separators=(",", ":"),
)


@dataclass(frozen=True, slots=True)
class QueryPlan:
    original_query: str
    research_intent: str
    core_concepts: list[str]
    methods: list[str]
    datasets: list[str]
    domains: list[str]
    time_constraints: list[str]
    venue_constraints: list[str]
    other_constraints: list[str]
    subqueries: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class QueryPlanResult:
    plan: QueryPlan
    generated: bool
    parse_error: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    raw_output: str | None = None
    extraction_mode: str | None = None

    def public_diagnostics(self) -> dict[str, Any]:
        """Bounded production-safe diagnostics; raw model text stays server-side."""

        return {
            "planner_valid": self.generated,
            "planner_fallback_used": not self.generated,
            "planner_parse_error": self.parse_error,
            "planner_raw_output_chars": len(self.raw_output or ""),
            "planner_json_extraction": self.extraction_mode,
            "planner_input_tokens": self.input_tokens,
            "planner_output_tokens": self.output_tokens,
        }


def fallback_query_plan(query: str, error: str | None = None) -> QueryPlanResult:
    normalized = query.strip()
    return QueryPlanResult(
        QueryPlan(normalized, normalized, [], [], [], [], [], [], [], [normalized]),
        False, error,
    )


def _direct_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped:
        raise ValueError("query plan output is empty")
    try:
        direct = json.loads(stripped)
    except json.JSONDecodeError:
        direct = None
    if isinstance(direct, dict):
        return direct
    return None


def _embedded_json_objects(text: str) -> list[dict[str, Any]]:
    """Return complete embedded objects without exposing surrounding model prose."""

    stripped = text.strip()
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    spans: set[tuple[int, int]] = set()
    for index, character in enumerate(stripped):
        if character != "{":
            continue
        try:
            candidate, end = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            span = (index, index + end)
            if span not in spans:
                spans.add(span)
                objects.append(candidate)
    return objects


def _validate_query_plan_payload(payload: dict[str, Any], original_query: str) -> QueryPlan:
    expected = {
        "original_query", "research_intent", "core_concepts", "methods", "datasets",
        "domains", "time_constraints", "venue_constraints", "other_constraints", "subqueries",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("query plan schema mismatch")
    original = payload["original_query"]
    intent = payload["research_intent"]
    if not isinstance(original, str) or not original.strip() or not isinstance(intent, str) or not intent.strip():
        raise ValueError("original_query and research_intent must be non-empty")
    arrays: dict[str, list[str]] = {}
    for key in expected - {"original_query", "research_intent"}:
        value = payload[key]
        if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
            raise ValueError(f"{key} must be a string array")
        arrays[key] = [item.strip() for item in value[:8]]
    subqueries = arrays["subqueries"] or [original_query.strip()]
    return QueryPlan(
        original.strip(), intent.strip(), arrays["core_concepts"], arrays["methods"],
        arrays["datasets"], arrays["domains"], arrays["time_constraints"],
        arrays["venue_constraints"], arrays["other_constraints"], subqueries,
    )


def parse_query_plan_with_mode(text: str, original_query: str) -> tuple[QueryPlan, str]:
    direct = _direct_json_object(text)
    if direct is not None:
        return _validate_query_plan_payload(direct, original_query), "direct"
    candidates = _embedded_json_objects(text)
    if not candidates:
        raise ValueError("query plan contains no complete JSON object")
    errors: list[str] = []
    for payload in candidates:
        try:
            return _validate_query_plan_payload(payload, original_query), "embedded"
        except ValueError as exc:
            errors.append(str(exc))
    raise ValueError("no embedded QueryPlan passed schema validation: " + errors[-1])


def parse_query_plan(text: str, original_query: str) -> QueryPlan:
    plan, _ = parse_query_plan_with_mode(text, original_query)
    return plan


class QueryPlanner:
    def build(self, query: str, policy: object) -> QueryPlanResult:
        if not isinstance(policy, QwenAgentPolicy):
            return fallback_query_plan(query, "policy does not support planning generation")
        generated = policy.generate_text(
            [
                {"role": "system", "content": QUERY_PLAN_SYSTEM_PROMPT},
                {"role": "user", "content": QUERY_PLAN_EXAMPLE_USER},
                {"role": "assistant", "content": QUERY_PLAN_EXAMPLE_ASSISTANT},
                {
                    "role": "user",
                    "content": (
                        "Build a new QueryPlan for the exact question below. "
                        "original_query must copy it verbatim. Populate every applicable "
                        "concept and constraint; do not return an empty schema template.\n"
                        f"<question>{query.strip()}</question>"
                    ),
                },
            ],
            max_new_tokens=QUERY_PLAN_MAX_NEW_TOKENS,
        )
        try:
            plan, extraction_mode = parse_query_plan_with_mode(generated.text, query)
        except ValueError as exc:
            fallback = fallback_query_plan(query, str(exc))
            LOGGER.warning(
                "QueryPlanner fallback: error=%s output_chars=%d",
                exc, len(generated.text),
            )
            return QueryPlanResult(
                fallback.plan, False, str(exc), generated.input_tokens,
                generated.output_tokens, generated.text, "failed",
            )
        return QueryPlanResult(
            plan, True, None, generated.input_tokens, generated.output_tokens,
            generated.text, extraction_mode,
        )
