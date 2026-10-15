"""Server-side online academic retrieval, isolated from benchmark retrievers."""

from src.retrieval.online.academic_search import AcademicSearch
from src.retrieval.online.models import OnlinePaperCandidate
from src.retrieval.online.serper_client import SerperClient

__all__ = ["AcademicSearch", "OnlinePaperCandidate", "SerperClient"]
