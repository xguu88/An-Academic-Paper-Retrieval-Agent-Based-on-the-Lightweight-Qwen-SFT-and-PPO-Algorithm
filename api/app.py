"""FastAPI app factory；默认 AGENT_MODEL_MODE=mock。"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.dependencies import (
    AppServices,
    build_local_services,
    build_mock_services,
    build_online_services,
)
from src.api.routes.agent import router as agent_router
from src.api.routes.health import router as health_router
from src.api.routes.models import router as models_router


def create_app(services: AppServices | None = None) -> FastAPI:
    """创建可测试 app；真实 checkpoint services 可由部署入口注入。"""

    mode = os.getenv("AGENT_MODEL_MODE", "mock").strip().lower()
    retrieval_mode = os.getenv("RETRIEVAL_MODE", "mock").strip().lower()
    if services is None:
        root = Path(__file__).resolve().parents[2]
        if mode == "mock" and retrieval_mode == "mock":
            services = build_mock_services()
        elif mode == "base" and retrieval_mode == "online":
            services = build_online_services(root)
        elif retrieval_mode == "scifact":
            services = build_local_services(root)
        else:
            raise RuntimeError(
                f"unsupported mode combination: AGENT_MODEL_MODE={mode}, "
                f"RETRIEVAL_MODE={retrieval_mode}"
            )
    app = FastAPI(title="Literature Retrieval Agent API", version="0.1.0")
    app.state.services = services
    app.add_middleware(#浏览器有同源策略：前端页面地址和后端接口地址不一样时，浏览器会拦截 AJAX 请求。中间件就是用来放开这个限制。
        CORSMiddleware,#就是**加在 FastAPI 后端的一层中间件**，在 HTTP 响应头里添加 CORS 相关头部，告诉浏览器：**允许这个前端跨域访问我的接口**。
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
    )
    app.include_router(health_router)#健康检查路由，一般是`/health`，用来监控服务是否存活。
    app.include_router(models_router)#模型相关接口，比如模型列表、推理。
    app.include_router(agent_router)#Agent 业务接口，Agent 的调用、会话等。
    return app


app = create_app()
