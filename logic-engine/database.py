"""
database.py — Neon PostgreSQL Database Layer
=============================================
Student Query Resolution System — Persistent Storage

Uses asyncpg for high-performance async Postgres connections to Neon's
serverless Postgres. Manages connection pooling, table creation, and
all CRUD operations for query history.

Connection string is read from the DATABASE_URL environment variable,
which Neon provides in the format:
    postgresql://user:password@ep-xxx.region.aws.neon.tech/dbname?sslmode=require
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import asyncpg
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATABASE_URL: str = os.getenv("DATABASE_URL", "")

# Connection pool — initialised at application startup
_pool: asyncpg.Pool | None = None


# ---------------------------------------------------------------------------
# Connection Management
# ---------------------------------------------------------------------------

async def connect_db() -> None:
    """
    Create the asyncpg connection pool to Neon Postgres.

    Called once during FastAPI startup. The pool maintains multiple
    connections for concurrent request handling.

    Neon requires SSL, which asyncpg handles automatically when
    `sslmode=require` is in the connection string.
    """
    global _pool

    if not DATABASE_URL:
        print("[DATABASE] No DATABASE_URL set — running without persistence.")
        return

    try:
        _pool = await asyncpg.create_pool(
            DATABASE_URL,
            min_size=2,       # Minimum connections in pool
            max_size=10,      # Maximum connections in pool
            command_timeout=30,
            statement_cache_size=0,  # Neon recommendation for serverless
        )
        print("[DATABASE] Connected to Neon Postgres successfully.")
        await _create_tables()
    except Exception as exc:
        print(f"[DATABASE] Connection failed: {exc}")
        _pool = None


async def disconnect_db() -> None:
    """Close the connection pool during FastAPI shutdown."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        print("[DATABASE] Connection pool closed.")


def is_connected() -> bool:
    """Check if the database pool is available."""
    return _pool is not None


# ---------------------------------------------------------------------------
# Schema — Table Creation
# ---------------------------------------------------------------------------

async def _create_tables() -> None:
    """
    Create the database tables if they don't already exist.

    Tables:
      • students       — Student directory (optional lookup)
      • queries        — Every query processed by the system
      • escalations    — A* routing results linked to queries
    """
    if not _pool:
        return

    async with _pool.acquire() as conn:
        await conn.execute("""
            -- Students table: tracks unique students who have submitted queries
            CREATE TABLE IF NOT EXISTS students (
                student_id    TEXT PRIMARY KEY,
                first_seen    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                query_count   INTEGER NOT NULL DEFAULT 0
            );
        """)

        await conn.execute("""
            -- Queries table: stores every query and its resolution
            CREATE TABLE IF NOT EXISTS queries (
                id              SERIAL PRIMARY KEY,
                query_id        TEXT UNIQUE NOT NULL,
                student_id      TEXT NOT NULL REFERENCES students(student_id),
                query_text      TEXT NOT NULL,

                -- NLP Classification fields
                class_type      TEXT,
                class_subject   TEXT,
                class_urgency   TEXT,
                class_handler   TEXT,

                -- Routing decision
                resolved        BOOLEAN NOT NULL DEFAULT FALSE,
                department      TEXT,
                resolution      TEXT,
                rule_fired      TEXT,

                -- Metadata
                created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
        """)

        await conn.execute("""
            -- Escalations table: A* pathfinding results
            CREATE TABLE IF NOT EXISTS escalations (
                id              SERIAL PRIMARY KEY,
                query_id        TEXT UNIQUE NOT NULL REFERENCES queries(query_id),
                path            TEXT[] NOT NULL,            -- Array of department names
                total_cost      REAL NOT NULL,
                nodes_explored  INTEGER NOT NULL,
                explanation     TEXT,
                created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
        """)

        # Index for fast student history lookups
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_queries_student_id
            ON queries(student_id);
        """)

        # Index for time-range queries
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_queries_created_at
            ON queries(created_at DESC);
        """)

        print("[DATABASE] Tables and indexes verified/created.")


# ---------------------------------------------------------------------------
# CRUD Operations
# ---------------------------------------------------------------------------

async def save_query(
    query_id: str,
    student_id: str,
    query_text: str,
    classification: dict[str, str],
    routing: dict[str, Any],
    escalation: dict[str, Any] | None = None,
) -> bool:
    """
    Persist a resolved query and its results to the database.

    This is called at the end of the resolution pipeline in logic.py.
    Uses a transaction to ensure atomicity across the students, queries,
    and escalations tables.

    Returns True on success, False if DB is unavailable or on error.
    """
    if not _pool:
        return False

    try:
        async with _pool.acquire() as conn:
            async with conn.transaction():
                # Upsert the student record and increment query count
                await conn.execute("""
                    INSERT INTO students (student_id, query_count)
                    VALUES ($1, 1)
                    ON CONFLICT (student_id)
                    DO UPDATE SET query_count = students.query_count + 1;
                """, student_id)

                # Insert the query record
                await conn.execute("""
                    INSERT INTO queries (
                        query_id, student_id, query_text,
                        class_type, class_subject, class_urgency, class_handler,
                        resolved, department, resolution, rule_fired
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11);
                """,
                    query_id,
                    student_id,
                    query_text,
                    classification.get("Type"),
                    classification.get("Subject"),
                    classification.get("Urgency"),
                    classification.get("Handler"),
                    routing.get("resolved", False),
                    routing.get("department"),
                    routing.get("resolution"),
                    routing.get("ruleFired"),
                )

                # Insert escalation if A* was triggered
                if escalation and escalation.get("path"):
                    await conn.execute("""
                        INSERT INTO escalations (
                            query_id, path, total_cost, nodes_explored, explanation
                        ) VALUES ($1, $2, $3, $4, $5);
                    """,
                        query_id,
                        escalation["path"],
                        float(escalation["totalCost"]),
                        escalation["nodesExplored"],
                        escalation.get("explanation"),
                    )

        return True
    except Exception as exc:
        print(f"[DATABASE] Error saving query {query_id}: {exc}")
        return False


async def get_query_by_id(query_id: str) -> dict[str, Any] | None:
    """Retrieve a single query by its query_id."""
    if not _pool:
        return None

    async with _pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT q.*, e.path, e.total_cost, e.nodes_explored, e.explanation AS esc_explanation
            FROM queries q
            LEFT JOIN escalations e ON q.query_id = e.query_id
            WHERE q.query_id = $1;
        """, query_id)

    if not row:
        return None

    return _row_to_dict(row)


async def get_student_history(
    student_id: str,
    limit: int = 20,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """
    Retrieve the query history for a specific student.

    Returns most recent queries first, with pagination support.
    """
    if not _pool:
        return []

    async with _pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT q.*, e.path, e.total_cost, e.nodes_explored, e.explanation AS esc_explanation
            FROM queries q
            LEFT JOIN escalations e ON q.query_id = e.query_id
            WHERE q.student_id = $1
            ORDER BY q.created_at DESC
            LIMIT $2 OFFSET $3;
        """, student_id, limit, offset)

    return [_row_to_dict(row) for row in rows]


async def get_recent_queries(limit: int = 50) -> list[dict[str, Any]]:
    """Retrieve the most recent queries across all students."""
    if not _pool:
        return []

    async with _pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT q.*, e.path, e.total_cost, e.nodes_explored, e.explanation AS esc_explanation
            FROM queries q
            LEFT JOIN escalations e ON q.query_id = e.query_id
            ORDER BY q.created_at DESC
            LIMIT $1;
        """, limit)

    return [_row_to_dict(row) for row in rows]


async def get_analytics() -> dict[str, Any]:
    """
    Aggregate analytics: total queries, breakdown by type/urgency/department,
    escalation rate, and busiest students.
    """
    if not _pool:
        return {"available": False}

    async with _pool.acquire() as conn:
        total = await conn.fetchval("SELECT COUNT(*) FROM queries;")

        type_breakdown = await conn.fetch("""
            SELECT class_type AS type, COUNT(*) AS count
            FROM queries
            GROUP BY class_type
            ORDER BY count DESC;
        """)

        urgency_breakdown = await conn.fetch("""
            SELECT class_urgency AS urgency, COUNT(*) AS count
            FROM queries
            GROUP BY class_urgency
            ORDER BY count DESC;
        """)

        dept_breakdown = await conn.fetch("""
            SELECT department, COUNT(*) AS count
            FROM queries
            GROUP BY department
            ORDER BY count DESC;
        """)

        escalation_count = await conn.fetchval(
            "SELECT COUNT(*) FROM escalations;"
        )

        top_students = await conn.fetch("""
            SELECT student_id, query_count
            FROM students
            ORDER BY query_count DESC
            LIMIT 10;
        """)

    return {
        "available": True,
        "totalQueries": total,
        "escalationCount": escalation_count,
        "escalationRate": round(escalation_count / max(total, 1) * 100, 1),
        "byType": [dict(r) for r in type_breakdown],
        "byUrgency": [dict(r) for r in urgency_breakdown],
        "byDepartment": [dict(r) for r in dept_breakdown],
        "topStudents": [dict(r) for r in top_students],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row_to_dict(row: asyncpg.Record) -> dict[str, Any]:
    """Convert an asyncpg Record (with joined escalation) to a clean dict."""
    d = dict(row)

    # Format timestamps as ISO strings
    for key in ("created_at", "first_seen"):
        if key in d and isinstance(d[key], datetime):
            d[key] = d[key].isoformat()

    # Nest escalation fields if present
    if d.get("path"):
        d["escalation"] = {
            "path": list(d.pop("path")),
            "totalCost": d.pop("total_cost", 0),
            "nodesExplored": d.pop("nodes_explored", 0),
            "explanation": d.pop("esc_explanation", ""),
        }
    else:
        d.pop("path", None)
        d.pop("total_cost", None)
        d.pop("nodes_explored", None)
        d.pop("esc_explanation", None)
        d["escalation"] = None

    return d

