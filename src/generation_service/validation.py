# Pydantic request/response schemas for the /query endpoint (see app.py). FastAPI uses
# these both to validate incoming requests and to auto-generate the OpenAPI docs at
# /docs, so the Field descriptions below double as API documentation.
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    question: str = Field(..., description="The query question for the financial documents.")
    # ge/le bound how many chunks a client can request per query, capping LLM context size
    # (and cost/latency) even if a caller passes an unreasonable value.
    limit: int = Field(default=3, ge=1, le=10, description="The maximum number of matching pages to retrieve.")

class QueryResponse(BaseModel):
    question: str
    answer: str
    pages_retrieved: int
    chunks: list[dict] = Field(default=[], description="List of retrieved text chunks with page metadata.")
