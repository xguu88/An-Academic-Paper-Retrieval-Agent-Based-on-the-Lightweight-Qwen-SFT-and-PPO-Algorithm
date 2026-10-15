"""与 Agent 解耦的语料、检索接口、结果和内存引用图公共接口。"""

from src.retrieval.base import Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.embedding import TextEmbedder
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.pair_scorer import PairScorer
from src.retrieval.reranker import Reranker
from src.retrieval.cross_encoder_reranker import CrossEncoderReranker
from src.retrieval.reranked_retriever import RerankedRetriever
from src.retrieval.sentence_transformer_cross_encoder import (
    SentenceTransformerCrossEncoderScorer,
)
from src.retrieval.citation_graph import CitationGraph, CitationGraphStats, build_citation_graph
from src.retrieval.corpus import PaperCorpus
from src.retrieval.paper import PaperDocument
from src.retrieval.types import RetrievalResult, RetrievalSource
from src.retrieval.validation import CorpusValidationError

__all__ = [
    "PaperDocument",
    "PaperCorpus",
    "CorpusValidationError",
    "Retriever",
    "TextEmbedder",
    "DenseRetriever",
    "HybridRetriever",
    "PairScorer",
    "Reranker",
    "CrossEncoderReranker",
    "RerankedRetriever",
    "SentenceTransformerCrossEncoderScorer",
    "RetrievalResult",
    "RetrievalSource",
    "CitationGraph",
    "CitationGraphStats",
    "build_citation_graph",
]
