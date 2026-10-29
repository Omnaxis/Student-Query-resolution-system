"""
main.py — FastAPI Application Entry Point
==========================================
Student Query Resolution System — Microservice 2 (Python Logic Engine)

Exposes endpoints for:
  • POST /process-query     — Classify, route, and resolve a student query
  • GET  /history/{id}      — Retrieve a student's query history
  • GET  /query/{query_id}  — Retrieve a specific query result
  • GET  /analytics         — Aggregate query statistics
  • GET  /                  — Health check

Integrates with Neon Postgres for persistent query storage.

Run locally:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from database import connect_db, disconnect_db, get_query_by_id, get_student_history, get_recent_queries, get_analytics, is_connected
from logic import resolve_query


# ---------------------------------------------------------------------------
# Lifespan — Startup / Shutdown Hooks
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application lifecycle:
      • Startup:  Connect to Neon Postgres and create tables.
      • Shutdown: Close the database connection pool.
    """
    # --- Startup ---
    await connect_db()
    yield
    # --- Shutdown ---
    await disconnect_db()


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Student Query Resolution — Logic Engine",
    description=(
        "AI & DSA-powered microservice that classifies student queries, "
        "applies rule-based routing, and uses A* search with a Min-Heap "
        "for optimal department escalation. Backed by Neon Postgres."
    ),
    version="1.0.0",
    lifespan=lifespan,
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
        "database": "connected" if is_connected() else "not configured",
    }


@app.post("/process-query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    POST /process-query

    Accepts a student query, classifies it using an LLM (or mock),
    evaluates routing rules, and optionally runs A* escalation.
    Persists the result to Neon Postgres.

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


@app.get("/history/{student_id}")
async def student_history(
    student_id: str,
    limit: int = Query(default=20, ge=1, le=100, description="Max results"),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
):
    """
    GET /history/{student_id}

    Retrieve the query history for a specific student, most recent first.
    Supports pagination via limit and offset query params.
    """
    if not is_connected():
        raise HTTPException(
            status_code=503,
            detail="Database not configured. Set DATABASE_URL to enable history.",
        )

    history = await get_student_history(student_id, limit=limit, offset=offset)
    return {
        "studentId": student_id,
        "count": len(history),
        "queries": history,
    }


@app.get("/query/{query_id}")
async def get_query(query_id: str):
    """
    GET /query/{query_id}

    Retrieve a specific query result by its unique query_id.
    """
    if not is_connected():
        raise HTTPException(
            status_code=503,
            detail="Database not configured. Set DATABASE_URL to enable lookups.",
        )

    result = await get_query_by_id(query_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Query '{query_id}' not found.")
    return result


@app.get("/queries/recent")
async def recent_queries(
    limit: int = Query(default=50, ge=1, le=200, description="Max results"),
):
    """
    GET /queries/recent

    Retrieve the most recent queries across all students.
    """
    if not is_connected():
        raise HTTPException(
            status_code=503,
            detail="Database not configured. Set DATABASE_URL to enable history.",
        )

    queries = await get_recent_queries(limit=limit)
    return {"count": len(queries), "queries": queries}


@app.get("/analytics")
async def analytics():
    """
    GET /analytics

    Aggregate query statistics: totals, breakdowns by type/urgency/department,
    escalation rate, and most active students.
    """
    if not is_connected():
        raise HTTPException(
            status_code=503,
            detail="Database not configured. Set DATABASE_URL to enable analytics.",
        )

    return await get_analytics()


# ---------------------------------------------------------------------------
# Standalone Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
