"""模型 registry 状态路由。"""

from fastapi import APIRouter, Request

router = APIRouter(prefix="/api", tags=["models"])


@router.get("/models")
def models(request: Request) -> dict[str, object]:
    services = request.app.state.services
    mode = services.retrieval_mode if services.retrieval_mode == "online" else services.model_mode
    return {"mode": mode, "models": services.registry.list_models()}
