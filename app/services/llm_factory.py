from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from app.core.config import get_settings


def get_llm() -> BaseChatModel:
    settings = get_settings()
    provider = settings.LLM_PROVIDER.lower()

    if provider == "groq":
        if not settings.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is missing in .env")
        return ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model_name=settings.GROQ_MODEL,
            temperature=0.2,
        )

    elif provider == "anthropic":
        if not settings.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY is missing in .env")
        return ChatAnthropic(
            api_key=settings.ANTHROPIC_API_KEY,
            model_name=settings.ANTHROPIC_MODEL,
            temperature=0.2,
        )

    else:
        raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. Choose 'groq' or 'anthropic'.")
