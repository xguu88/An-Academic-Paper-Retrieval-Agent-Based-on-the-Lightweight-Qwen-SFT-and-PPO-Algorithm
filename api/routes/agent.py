"""Agent online inference 路由。"""

from fastapi import APIRouter, HTTPException, Request

from src.api.schemas import AgentRunRequest, AgentRunResponse
from src.inference.runner import InferenceRunner
from src.inference.online_runner import OnlineInferenceRunner

router = APIRouter(prefix="/api/agent", tags=["agent"])


@router.post("/run", response_model=AgentRunResponse)
def run_agent(payload: AgentRunRequest, request: Request) -> AgentRunResponse:
    services = request.app.state.services
    try:
        policy = services.registry.get(payload.model)
        environment = services.environment_factory(payload.max_steps)
        runner_type = (
            OnlineInferenceRunner
            if services.retrieval_mode == "online"
            else InferenceRunner
        )
        result = runner_type(
            environment, policy, model_name=payload.model, max_steps=payload.max_steps
        ).run_query(payload.query)
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    data = result.to_dict()
    return AgentRunResponse(
        query=data["query"],
        model=data["model"],
        mode=services.retrieval_mode,
        success=None,
        terminated=data["terminated"],
        trajectory=data["trajectory"],
        final_candidates=data["final_candidates"],
        metrics=data["metrics"],
        latency_ms=data["latency_ms"],
        query_plan=data.get("query_plan"),
        query_plan_diagnostics=data.get("query_plan_diagnostics"),
        result_groups=data.get("result_groups"),
        cost_metrics=data.get("cost_metrics"),
    )
