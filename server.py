import uvicorn                                        # the HTTP server that runs FastAPI
from fastapi import FastAPI                           # the web framework
from fastapi.middleware.cors import CORSMiddleware    # allows frontend to call the API from a different port
from app.api.routes import health, documents, chat    # our three route files
from app.core.config import get_settings              # centralized config


settings = get_settings()   # load settings once at startup

# create the FastAPI app instance
# title and description appear in the auto-generated /docs page
app = FastAPI(
    title="YTRAG - RAG Agent API",
    description="A production-ready RAG agent with document search and web fallback.",
    version="1.0.0",
)

# CORS = Cross-Origin Resource Sharing
# browsers block requests from one port (3000) to another (8000) by default
# this middleware tells the browser "it's okay, allow these origins"
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],         # allow all origins — restrict to specific URLs in production
    allow_credentials=True,      # allow cookies and auth headers
    allow_methods=["*"],         # allow GET, POST, PUT, DELETE, etc.
    allow_headers=["*"],         # allow all request headers
)

# register each router with a prefix
# prefix="/api" means all routes become /api/health, /api/chat, /api/documents/upload
app.include_router(health.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
app.include_router(documents.router, prefix="/api")


@app.on_event("startup")
def on_startup():
    # this function runs once when the server starts — before it accepts any requests
    # we pre-load the vector store so the first user request isn't slow
    print("[INFO] Server starting up...")
    try:
        from app.services.vectorstore import VectorStore
        store = VectorStore()
        store.load()                  # load FAISS index into memory at startup
        print("[INFO] Vector store pre-loaded successfully")
    except FileNotFoundError:
        # if no vector store exists yet, that's fine — it will be built on first upload
        print("[WARN] No vector store found. Upload documents to build one.")


# root route — a quick sanity check when someone visits http://localhost:8000
@app.get("/")
def root():
    return {
        "message": "YTRAG RAG Agent API is running.",
        "docs": "http://localhost:8000/docs",       # link to auto-generated API docs
        "health": "http://localhost:8000/api/health",
    }


# entry point when running: python server.py
if __name__ == "__main__":
    uvicorn.run(
        "server:app",               # "filename:FastAPI_instance_name"
        host=settings.HOST,         # 0.0.0.0 means accessible from any network interface
        port=settings.PORT,         # 8000 by default
        reload=True,                # auto-restart server when you save a file (dev mode only)
    )
