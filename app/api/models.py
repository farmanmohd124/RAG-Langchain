from pydantic import BaseModel, Field   # BaseModel is the base class, Field adds extra config per field
from typing import Optional             # Optional means the field can be None (not required)


# ── Request Models (what the client SENDS to the API) ──────────────────────────

class ChatRequest(BaseModel):
    # the user's question — required, cannot be empty
    query: str = Field(..., min_length=1, description="The user's question or message")

    # session_id groups messages into one conversation
    # Optional with a default means the client can omit it and we fall back to "default"
    session_id: Optional[str] = Field(default="default", description="Unique session ID for conversation memory")


class ClearSessionRequest(BaseModel):
    # used when the user wants to start a fresh conversation
    session_id: str = Field(..., description="The session ID to clear")


# ── Response Models (what the API SENDS back to the client) ────────────────────

class ChatResponse(BaseModel):
    # the agent's answer text
    answer: str = Field(..., description="The agent's response to the query")

    # echo back the session_id so the client knows which session this belongs to
    session_id: str = Field(..., description="The session ID for this conversation")

    # whether the request was successful
    success: bool = Field(default=True, description="Whether the request succeeded")


class UploadResponse(BaseModel):
    # confirmation message after a file is uploaded and indexed
    message: str = Field(..., description="Status message about the upload")

    # how many chunks were added to the vector store from this file
    chunks_added: int = Field(..., description="Number of text chunks added to the knowledge base")

    # the name of the uploaded file
    filename: str = Field(..., description="Name of the uploaded file")

    success: bool = Field(default=True, description="Whether the upload succeeded")


class HealthResponse(BaseModel):
    # simple status field — "ok" means the server is running
    status: str = Field(default="ok", description="Server status")

    # which LLM provider is currently active (groq or anthropic)
    llm_provider: str = Field(..., description="Currently active LLM provider")

    # which embedding model is loaded
    embed_model: str = Field(..., description="Embedding model in use")


class ErrorResponse(BaseModel):
    # returned when something goes wrong — gives the client a clear error message
    success: bool = Field(default=False)
    error: str = Field(..., description="Description of what went wrong")
