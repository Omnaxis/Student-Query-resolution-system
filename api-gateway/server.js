/**
 * server.js — Node.js Express API Gateway
 * ==========================================
 * Student Query Resolution System — Microservice 1
 *
 * This service acts as the single entry point for the frontend client.
 * It exposes POST /api/resolve, validates the incoming payload, and
 * proxies the request to the Python FastAPI Logic Engine for processing.
 *
 * Run locally:
 *   npm start          (production)
 *   npm run dev        (development)
 *
 * Environment Variables (via .env):
 *   PORT                — Server port (default: 3000)
 *   PYTHON_SERVICE_URL  — Base URL of the Python Logic Engine
 */

const express = require("express");
const cors = require("cors");
const axios = require("axios");

// Load environment variables from .env file
require("dotenv").config();

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const PORT = process.env.PORT || 3000;
const PYTHON_SERVICE_URL =
process.env.PYTHON_SERVICE_URL || "http://localhost:8000";

// ---------------------------------------------------------------------------
// Express Application Setup
// ---------------------------------------------------------------------------

const app = express();

// Middleware
app.use(cors()); // Allow cross-origin requests from the frontend
app.use(express.json()); // Parse JSON request bodies

// ---------------------------------------------------------------------------
// Health Check
// ---------------------------------------------------------------------------

app.get("/", (_req, res) => {
  res.json({
    service: "Student Query Resolution — API Gateway",
    status: "healthy",
    version: "1.0.0",
    pythonServiceUrl: PYTHON_SERVICE_URL,
  });
});

// ---------------------------------------------------------------------------
// POST /api/resolve — Main Query Resolution Endpoint
// ---------------------------------------------------------------------------

/**
 * Accepts a student query from the frontend, forwards it to the Python
 * FastAPI Logic Engine, and returns the structured resolution response.
 *
 * Request Body:
 *   {
 *     "studentId": "STU001",
 *     "queryText": "How do I pay my tuition fee?"
 *   }
 *
 * Response:
 *   The full JSON response from the Logic Engine, including:
 *     - classification (LLM-extracted frame)
 *     - routing (rule-based decision)
 *     - escalation (A* pathfinding result, if applicable)
 */
app.post("/api/resolve", async (req, res) => {
  try {
    const { studentId, queryText } = req.body;

    // --- Input Validation ---
    if (!studentId || !queryText) {
      return res.status(400).json({
        error: "Bad Request",
        message: "Both 'studentId' and 'queryText' are required.",
      });
    }

    if (typeof queryText !== "string" || queryText.trim().length === 0) {
      return res.status(400).json({
        error: "Bad Request",
        message: "'queryText' must be a non-empty string.",
      });
    }

    console.log(
      `[GATEWAY] Received query from ${studentId}: "${queryText.substring(0, 80)}..."`
    );

    // --- Forward to Python Logic Engine ---
    const response = await axios.post(
      `${PYTHON_SERVICE_URL}/process-query`,
      { studentId, queryText },
      {
        headers: { "Content-Type": "application/json" },
        timeout: 30000, // 30-second timeout for LLM calls
      }
    );

    console.log(
      `[GATEWAY] Query ${response.data.queryId} resolved → ${response.data.routing.department}`
    );

    // --- Return the Logic Engine's response to the client ---
    return res.json(response.data);
  } catch (error) {
    // Distinguish between Logic Engine errors and network errors
    if (error.response) {
      // The Python service returned an error status
      console.error(
        `[GATEWAY] Logic Engine error: ${error.response.status}`,
        error.response.data
      );
      return res.status(error.response.status).json({
        error: "Logic Engine Error",
        message:
          error.response.data?.detail ||
          "The Logic Engine returned an error.",
      });
    } else if (error.request) {
      // The request was made but no response was received
      console.error(
        "[GATEWAY] Logic Engine unreachable:",
        error.message
      );
      return res.status(503).json({
        error: "Service Unavailable",
        message:
          "The Logic Engine is currently unreachable. Please try again later.",
      });
    } else {
      // Something else went wrong
      console.error("[GATEWAY] Unexpected error:", error.message);
      return res.status(500).json({
        error: "Internal Server Error",
        message: "An unexpected error occurred.",
      });
    }
  }
});

// ---------------------------------------------------------------------------
// Generic proxy helper for GET requests to the Python Logic Engine
// ---------------------------------------------------------------------------

/**
 * Forwards a GET request to the Python Logic Engine and returns its response.
 * Used for history, analytics, and query-lookup endpoints.
 */
async function proxyGet(pythonPath, res) {
  try {
    const response = await axios.get(`${PYTHON_SERVICE_URL}${pythonPath}`, {
      timeout: 15000,
    });
    return res.json(response.data);
  } catch (error) {
    if (error.response) {
      return res.status(error.response.status).json({
        error: "Logic Engine Error",
        message: error.response.data?.detail || "Error from Logic Engine.",
      });
    } else if (error.request) {
      return res.status(503).json({
        error: "Service Unavailable",
        message: "The Logic Engine is currently unreachable.",
      });
    } else {
      return res.status(500).json({
        error: "Internal Server Error",
        message: "An unexpected error occurred.",
      });
    }
  }
}

// ---------------------------------------------------------------------------
// GET /api/history/:studentId — Student Query History
// ---------------------------------------------------------------------------

app.get("/api/history/:studentId", async (req, res) => {
  const { studentId } = req.params;
  const { limit = 20, offset = 0 } = req.query;
  console.log(`[GATEWAY] History request for ${studentId}`);
  return proxyGet(`/history/${studentId}?limit=${limit}&offset=${offset}`, res);
});

// ---------------------------------------------------------------------------
// GET /api/query/:queryId — Single Query Lookup
// ---------------------------------------------------------------------------

app.get("/api/query/:queryId", async (req, res) => {
  const { queryId } = req.params;
  console.log(`[GATEWAY] Query lookup: ${queryId}`);
  return proxyGet(`/query/${queryId}`, res);
});

// ---------------------------------------------------------------------------
// GET /api/queries/recent — Recent Queries
// ---------------------------------------------------------------------------

app.get("/api/queries/recent", async (req, res) => {
  const { limit = 50 } = req.query;
  console.log("[GATEWAY] Recent queries request");
  return proxyGet(`/queries/recent?limit=${limit}`, res);
});

// ---------------------------------------------------------------------------
// GET /api/analytics — Aggregate Analytics
// ---------------------------------------------------------------------------

app.get("/api/analytics", async (req, res) => {
  console.log("[GATEWAY] Analytics request");
  return proxyGet("/analytics", res);
});

// ---------------------------------------------------------------------------
// Start Server
// ---------------------------------------------------------------------------

app.listen(PORT, () => {
  console.log(`\n🚀 API Gateway running on port ${PORT}`);
  console.log(`   Python Service URL: ${PYTHON_SERVICE_URL}`);
  console.log(`   Endpoint: POST http://localhost:${PORT}/api/resolve\n`);
});

