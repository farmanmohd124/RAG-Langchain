from fastapi import APIRouter            # APIRouter lets us define routes in separate files
from app.api.models import HealthResponse  # the response shape we defined in Step 9
from app.core.config import get_settings   # our centralized config from Step 2

# APIRouter is like a mini FastAPI app — we define routes here
# and register them in server.py with a prefix like /api
router = APIRouter()


@router.get(
    "/health",                       # the URL path: GET /health
    response_model=HealthResponse,   # FastAPI validates the response matches this shape
    tags=["Health"],                 # groups this endpoint under "Health" in the /docs UI
    summary="Check if the API is running",  # short description shown in /docs
)
def health_check():
    # load settings to report what is currently configured
    settings = get_settings()

    # return a HealthResponse object — FastAPI automatically converts it to JSON
    return HealthResponse(
        status="ok",                          # server is alive
        llm_provider=settings.LLM_PROVIDER,   # e.g. "groq" or "anthropic"
        embed_model=settings.EMBED_MODEL,     # e.g. "all-MiniLM-L6-v2"
    )
