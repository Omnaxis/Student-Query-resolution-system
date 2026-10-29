"""
main.py — FastAPI Application Entry Point
==========================================
Student Query Resolution System — Microservice 2 (Python Logic Engine)

Exposes a single POST /process-query endpoint that orchestrates:
  1. LLM-based query classification
  2. Rule-based expert system evaluation
  3. A* pathfinding for escalation routing

Run locally:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from logic import resolve_query

# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Student Query Resolution — Logic Engine",
    description=(
        "AI & DSA-powered microservice that classifies student queries, "
        "applies rule-based routing, and uses A* search with a Min-Heap "
        "for optimal department escalation."
    ),
    version="1.0.0",
)

# CORS — allow the API Gateway and frontend origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    """Incoming query payload forwarded by the Node.js API Gateway."""
    studentId: str = Field(..., description="Unique student identifier")
    queryText: str = Field(..., description="Raw natural-language query text")


class ClassificationSchema(BaseModel):
    QueryID: str | None = None
    StudentID: str | None = None
    Type: str
    Subject: str
    Urgency: str
    Handler: str


class RoutingSchema(BaseModel):
    resolved: bool
    department: str
    resolution: str
    ruleFired: str


class EscalationSchema(BaseModel):
    path: list[str]
    totalCost: float
    nodesExplored: int
    explanation: str


class QueryResponse(BaseModel):
    """Full response returned to the API Gateway."""
    queryId: str
    studentId: str
    classification: ClassificationSchema
    routing: RoutingSchema
    escalation: EscalationSchema | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def health_check():
    """Health-check endpoint for uptime monitoring."""
    return {
        "service": "Student Query Resolution — Logic Engine",
        "status": "healthy",
        "version": "1.0.0",
    }


@app.post("/process-query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    POST /process-query

    Accepts a student query, classifies it using an LLM (or mock),
    evaluates routing rules, and optionally runs A* escalation.

    Request Body:
        { "studentId": "STU001", "queryText": "How do I pay my tuition fee?" }

    Returns:
        Full resolution response including classification, routing decision,
        and (if applicable) the A* escalation path with algorithm trace.
    """
    if not request.queryText.strip():
        raise HTTPException(status_code=400, detail="queryText must not be empty.")

    try:
        result = await resolve_query(
            student_id=request.studentId,
            query_text=request.queryText,
        )
        return result
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Internal processing error: {str(exc)}",
        )


# ---------------------------------------------------------------------------
# Standalone Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

