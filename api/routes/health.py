"""健康检查路由。"""

from fastapi import APIRouter, Request

from src.api.schemas import HealthResponse

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health", response_model=HealthResponse, response_model_exclude_none=True)
def health(request: Request) -> HealthResponse:
    services = request.app.state.services
    if services.retrieval_mode == "mock":
        return HealthResponse(model_mode=services.model_mode)
    return HealthResponse(
        model_mode=services.model_mode,
        retrieval_mode=services.retrieval_mode,
        base_model_loaded=services.base_model_loaded,
        serper_configured=services.serper_configured,
        academic_backends=services.academic_backends,
    )
