# FastAPI service exposing the RAG pipeline over HTTP (used by the UI/app.js and any
# other client). Query flow per request: embed question -> vector search in Qdrant ->
# feed retrieved chunks + question to Gemini -> return the answer plus the chunks used,
# so the UI can show "reference excerpts" alongside the answer.
import os
import sys
import urllib.request
from fastapi import FastAPI, HTTPException, status
from dotenv import load_dotenv

# Ensure root directory is in the path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

# Load environment variables
load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, '.env'))

from src.embedding_service.embedder import GeminiEmbedder
from src.embedding_service.qdrant_manager import QdrantManager
from src.generation_service.gemini_rag_llm import GeminiRAG
from src.generation_service.validation import QueryRequest, QueryResponse

from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI App
app = FastAPI(
    title="Financial RAG Generation Service",
    description="REST API for answering questions about financial PDFs using Qdrant vector DB and Gemini LLM.",
    version="1.0.0"
)

# Enable CORS for frontend web integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global clients
qdrant_manager = None
embedder = None
rag_llm = None

def init_qdrant_manager(collection_name):
    # Probe for a real Qdrant server (e.g. the Docker container from docker-compose.yml)
    # first; if none responds within 1s, fall back to the embedded local-file client
    # pointed at ./qdrant_storage. This lets the exact same code run in Docker (server
    # mode) or on a bare laptop with no Docker running (local mode).
    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    use_server = False
    try:
        with urllib.request.urlopen(f"{qdrant_url.rstrip('/')}/healthz", timeout=1.0) as response:
            if response.status == 200:
                use_server = True
    except Exception:
        pass

    if use_server:
        print(f"Connected to Qdrant Docker server at {qdrant_url}")
        return QdrantManager(url=qdrant_url, collection_name=collection_name)
    else:
        qdrant_path = os.path.join(PROJECT_ROOT, "qdrant_storage")
        print(f"Qdrant Docker server not detected. Using local path: {qdrant_path}")
        return QdrantManager(path=qdrant_path, collection_name=collection_name)

@app.on_event("startup")
def startup_event():
    # Clients are created once at process startup (not per-request) since each holds a
    # connection/HTTP client that's expensive to set up repeatedly. Stored as module-level
    # globals so query_rag() and health_check() below can reach them.
    global qdrant_manager, embedder, rag_llm
    collection_name = os.environ.get("COLLECTION_NAME", "financial_documents")
    print("Initializing clients for generation service...")
    try:
        qdrant_manager = init_qdrant_manager(collection_name)
        embedder = GeminiEmbedder(model_name="gemini-embedding-2")
        rag_llm = GeminiRAG()
        print("All clients successfully initialized.")
    except Exception as e:
        print(f"ERROR: Initialization failed during startup: {e}")
        # Note: We do not fail hard immediately to allow debugging/health checks to run,
        # but queries will fail.



@app.options("/query")
def options_query():
    # Explicit handler for CORS preflight requests (browsers send OPTIONS before a
    # cross-origin POST); CORSMiddleware above handles the actual headers.
    return {}

@app.post("/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
def query_rag(request: QueryRequest):
    global qdrant_manager, embedder, rag_llm

    if not qdrant_manager or not embedder or not rag_llm:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service clients are not initialized. Check API keys and DB connection."
        )

    # 1. Embed query
    try:
        query_embedding = embedder.embed_text(request.question)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate query embedding: {e}"
        )

    # 2. Retrieve matched text chunks from Qdrant
    try:
        top_chunks = qdrant_manager.search_text_chunks(query_embedding, limit=request.limit)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search Qdrant vector DB: {e}"
        )

    if not top_chunks:
        return QueryResponse(
            question=request.question,
            answer="No matching document context was found in the database.",
            pages_retrieved=0,
            chunks=[]
        )

    # 3. Generate answer via Gemini LLM
    try:
        answer = rag_llm.answer_question(request.question, top_chunks)
        return QueryResponse(
            question=request.question,
            answer=answer,
            pages_retrieved=len(top_chunks),
            chunks=top_chunks
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate answer from Gemini LLM: {e}"
        )

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    global qdrant_manager, embedder, rag_llm
    
    qdrant_ok = qdrant_manager is not None
    embedder_ok = embedder is not None
    llm_ok = rag_llm is not None
    
    status_str = "healthy" if (qdrant_ok and embedder_ok and llm_ok) else "degraded"
    
    return {
        "status": status_str,
        "details": {
            "qdrant_connected": qdrant_ok,
            "embedder_initialized": embedder_ok,
            "llm_initialized": llm_ok
        }
    }

if __name__ == "__main__":
    import uvicorn
    # Use reload=False when running directly, or "src.generation_service.app:app" with reload=True for development
    uvicorn.run(app, host="0.0.0.0", port=8000)
