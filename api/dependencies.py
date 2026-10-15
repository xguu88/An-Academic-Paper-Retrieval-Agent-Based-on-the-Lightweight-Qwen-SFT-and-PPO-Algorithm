"""API services 与完全离线 mock environment factory。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
import os
from pathlib import Path
from typing import Any

import yaml

from src.environment.capabilities import EnvironmentCapabilities
from src.environment.config import EnvironmentConfig
from src.environment.literature_env import LiteratureEnvironment
from src.inference.model_registry import (
    ModelRegistry,
    load_base_qwen_policy,
    load_qwen_adapter_policy,
    load_qwen_registry,
)
from src.inference.policy import FakeAgentPolicy
from src.retrieval.base import Retriever
from src.retrieval.corpus import PaperCorpus
from src.retrieval.hybrid_retriever import HybridConfig, HybridRetriever
from src.retrieval.paper import PaperDocument
from src.retrieval.types import RetrievalResult, RetrievalSource


@dataclass(slots=True)
class AppServices:
    """FastAPI routes 的显式依赖容器。"""

    registry: ModelRegistry
    environment_factory: Callable[[int], Any]
    model_mode: str
    retrieval_mode: str = "mock"
    base_model_loaded: bool = False
    serper_configured: bool = False
    academic_backends: dict[str, bool] | None = None


class _MockRetriever(Retriever):
    """无需第三方索引依赖的稳定开发检索器。"""

    def __init__(self, corpus: PaperCorpus, source: RetrievalSource) -> None:
        self.corpus = corpus
        self.source = source

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        if not query.strip() or top_k <= 0:
            raise ValueError("mock retrieve query/top_k 非法")
        tokens = set(query.lower().split())
        papers = list(self.corpus)
        ordered = sorted(
            papers,
            key=lambda paper: (
                -sum(
                    token in f"{paper.title} {paper.abstract or ''}".lower()
                    for token in tokens
                ),
                paper.paper_id,
            ),
        )
        return [
            RetrievalResult(
                paper.paper_id,
                paper.title,
                score=float(len(ordered) - index),
                rank=index + 1,
                source=self.source,
            )
            for index, paper in enumerate(ordered[:top_k])
        ]


def build_mock_environment(max_steps: int = 6) -> LiteratureEnvironment:
    """构建真实 Environment + 轻量 BM25 的离线开发环境。"""

    corpus = PaperCorpus(
        [
            PaperDocument(
                "mock-transformer",
                "Attention Is All You Need",
                "The Transformer uses attention mechanisms for sequence modeling.",
                year=2017,
            ),
            PaperDocument(
                "mock-retrieval",
                "Dense Passage Retrieval for Open-Domain Question Answering",
                "Dense dual encoders retrieve passages using learned vector representations.",
                year=2020,
            ),
            PaperDocument(
                "mock-rl",
                "Proximal Policy Optimization Algorithms",
                "PPO optimizes a clipped surrogate objective for stable policy updates.",
                year=2017,
            ),
        ]
    )
    keyword = _MockRetriever(corpus, RetrievalSource.BM25)
    semantic = _MockRetriever(corpus, RetrievalSource.DENSE)
    hybrid = HybridRetriever(
        keyword,
        semantic,
        HybridConfig(bm25_top_n=3, dense_top_n=3, final_top_k=3),
    )
    return LiteratureEnvironment(
        corpus,
        keyword,
        semantic,
        hybrid_retriever=hybrid,
        config=EnvironmentConfig(
            max_steps=max_steps, max_searches=max(2, max_steps), max_reads=3
        ),
        capabilities=EnvironmentCapabilities.scifact(),
    )


def build_mock_services() -> AppServices:
    """让 base/sft/ppo 共用 API 契约但不伪造 checkpoint 效果。"""

    registry = ModelRegistry()
    for name in ModelRegistry.ALLOWED:
        registry.register(name, FakeAgentPolicy())
    return AppServices(registry, build_mock_environment, "mock", "mock")


def build_local_services(project_root: str | Path) -> AppServices:
    """加载本地 checkpoint 与 SciFact corpus；仍不读取任何 qrels/task split。"""

    from src.benchmarks.runtime import build_retrievers

    root = Path(project_root).expanduser().resolve()
    inference_config = yaml.safe_load(
        (root / "configs" / "inference.yaml").read_text(encoding="utf-8")
    )
    retrieval_config = yaml.safe_load(
        (root / inference_config["retrieval_config"]).read_text(encoding="utf-8")
    )
    environment_config = EnvironmentConfig.from_mapping(
        yaml.safe_load(
            (root / inference_config["environment_config"]).read_text(encoding="utf-8")
        )
    )
    processed = root / "data" / "benchmarks" / "scifact" / "processed"
    corpus = PaperCorpus.from_jsonl(processed / "papers.jsonl")
    retrievers, _ = build_retrievers(
        corpus,
        retrieval_config,
        processed / "indexes" / "dense",
        requested={"bm25", "dense"},
    )
    rrf = retrieval_config["rrf"]
    hybrid = HybridRetriever(
        retrievers["bm25"],
        retrievers["dense"],
        HybridConfig(
            bm25_top_n=rrf.get("bm25_top_n", 50),
            dense_top_n=rrf.get("dense_top_n", 50),
            rrf_k=rrf.get("k", 60),
            final_top_k=rrf.get("final_top_k", 20),
        ),
    )

    def environment_factory(max_steps: int) -> LiteratureEnvironment:
        return LiteratureEnvironment(
            corpus,
            retrievers["bm25"],
            retrievers["dense"],
            hybrid_retriever=hybrid,
            config=replace(environment_config, max_steps=max_steps),
            capabilities=EnvironmentCapabilities.scifact(),
        )

    return AppServices(
        load_qwen_registry(root / "configs" / "inference.yaml"),
        environment_factory,
        "local",
        "scifact",
        True,
        False,
    )


def build_online_services(project_root: str | Path) -> AppServices:
    """Load local Base Qwen, OpenAlex-first search, Serper, and local reranker."""

    from src.environment.online_literature_env import OnlineLiteratureEnvironment
    from src.retrieval.online.academic_search import AcademicSearch
    from src.retrieval.online.openalex_client import OpenAlexClient
    from src.retrieval.online.paper_reranker import OnlinePaperReranker
    from src.retrieval.online.serper_client import SerperClient
    from src.retrieval.sentence_transformer_cross_encoder import SentenceTransformerCrossEncoderScorer

    root = Path(project_root).expanduser().resolve()
    api_key = os.getenv("SERPER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("SERPER_API_KEY is required for online retrieval")
    raw_model_path = os.getenv("BASE_MODEL_PATH", "").strip()
    if not raw_model_path:
        raise FileNotFoundError("Base Qwen model path does not exist")
    model_path = Path(raw_model_path).expanduser()
    if not model_path.is_dir():
        raise FileNotFoundError("Base Qwen model path does not exist")
    config = yaml.safe_load(
        (root / "configs" / "inference.yaml").read_text(encoding="utf-8")
    )
    retrieval_config = yaml.safe_load(
        (root / "configs" / "retrieval.yaml").read_text(encoding="utf-8")
    )
    reranker_config = retrieval_config["reranker"]
    reranker_path = Path(reranker_config["model_name_or_path"]).expanduser()
    scorer = SentenceTransformerCrossEncoderScorer(
        reranker_path,
        device=str(reranker_config.get("device", "cpu")),
        batch_size=int(reranker_config.get("batch_size", 16)),
        allow_remote=False,
    )
    reranker = OnlinePaperReranker(scorer)
    registry = ModelRegistry()
    registry.register(
        "base", load_base_qwen_policy(model_path, generation=config.get("generation", {}))
    )
    for name in ("sft", "ppo"):
        entry = config.get("models", {}).get(name, {})
        adapter = entry.get("adapter_path")
        resolved_adapter = (root / adapter).resolve() if adapter else None
        if resolved_adapter is not None and resolved_adapter.is_dir():
            registry.register(
                name,
                load_qwen_adapter_policy(
                    model_path,
                    resolved_adapter,
                    generation=config.get("generation", {}),
                ),
            )
        else:
            registry.mark_unavailable(name, "checkpoint not available")

    def environment_factory(max_steps: int) -> OnlineLiteratureEnvironment:
        client = SerperClient(api_key, timeout_seconds=10.0, network_retries=1)
        openalex = OpenAlexClient(
            api_key=os.getenv("OPENALEX_API_KEY", "").strip() or None,
            timeout_seconds=10.0,
            network_retries=1,
            max_requests=6,
        )
        return OnlineLiteratureEnvironment(
            AcademicSearch(client, openalex_client=openalex),
            max_steps=min(max_steps, 20), max_searches=3, reranker=reranker,
        )

    return AppServices(
        registry=registry,
        environment_factory=environment_factory,
        model_mode="base",
        retrieval_mode="online",
        base_model_loaded=True,
        serper_configured=True,
        academic_backends={"openalex": True, "semantic_scholar": False, "serper": True},
    )
