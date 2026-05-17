from fastapi import APIRouter, HTTPException          # APIRouter for routes, HTTPException for errors
from app.api.models import (                          # import all models we need
    ChatRequest,
    ChatResponse,
    ClearSessionRequest,
    ErrorResponse,
)
from app.agent.agent import RAGAgent                  # our agent brain from Step 8

router = APIRouter()

# single shared agent instance — holds conversation memory across requests
# creating it here (module level) means it lives as long as the server runs
_agent = RAGAgent()


@router.post(
    "/chat",                          # URL: POST /chat
    response_model=ChatResponse,      # successful response must match this shape
    tags=["Chat"],                    # groups under "Chat" in /docs UI
    summary="Send a message to the RAG agent",
)
def chat(request: ChatRequest):       # FastAPI automatically parses the JSON body into ChatRequest
    try:
        # pass the user's query and session_id to the agent
        # the agent will search tools, reason, and return a structured dict
        result = _agent.chat(
            query=request.query,
            session_id=request.session_id,
        )

        # wrap the result in a ChatResponse and return it
        # FastAPI converts this Pydantic model to JSON automatically
        return ChatResponse(
            answer=result["answer"],
            session_id=result["session_id"],
            success=True,
        )

    except Exception as e:
        # if the agent crashes (LLM timeout, tool error, etc.)
        # return a clean 500 error instead of exposing the raw traceback
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/chat/clear",                           # URL: POST /chat/clear
    tags=["Chat"],
    summary="Clear conversation history for a session",
)
def clear_session(request: ClearSessionRequest):
    # delete this session's message history from the agent's memory
    _agent.clear_session(request.session_id)

    # return a simple dict — FastAPI converts it to JSON {"message": "..."}
    return {"message": f"Session '{request.session_id}' cleared successfully."}
