"""在线/benchmark inference 的稳定 JSON trace 类型。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class InferenceTraceStep:
    """前端展示的一次 Agent decision transition。"""

    step: int
    state_summary: dict[str, Any]
    action: dict[str, Any]
    raw_action_text: str
    observation: dict[str, Any]
    reward: float
    remaining_budget: dict[str, int]
    valid_action: bool
    raw_model_outputs: list[str] | None = None
    parse_error: str | None = None
    retry_count: int = 0


@dataclass(slots=True)
class InferenceResult:
    """区分 online metrics 与可选 benchmark success。"""

    query: str
    model: str
    terminated: bool
    trajectory: list[InferenceTraceStep]
    final_candidates: list[dict[str, Any]]
    metrics: dict[str, int | float | None]
    mode: str = "benchmark"
    benchmark_success: bool | None = None
    latency_ms: float = 0.0
    query_plan: dict[str, Any] | None = None
    query_plan_diagnostics: dict[str, Any] | None = None
    result_groups: list[dict[str, Any]] | None = None
    cost_metrics: dict[str, int | float | None] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
