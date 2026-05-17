from functools import lru_cache
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.core.config import get_settings


@lru_cache
def get_embedding_model() -> HuggingFaceEmbeddings:
    settings = get_settings()
    return HuggingFaceEmbeddings(
        model_name=settings.EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
