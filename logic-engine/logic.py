"""
logic.py — Core AI & DSA Logic Engine
======================================
Student Query Resolution System — Microservice 2 (Logic Layer)

This module implements three core sub-systems:
  A) LLM-Based Natural Language Classification (frame extraction)
  B) Rule-Based Expert System (semantic-net routing)
  C) A* Pathfinding with Min-Heap for optimal department escalation

=== Algorithmic Complexity Notes ===
  • Min-Heap push/pop: O(log n) per operation.
  • A* Search overall: O(b^d) worst-case where b = branching factor, d = depth.
    With an admissible heuristic and the Min-Heap open list the practical
    performance is far better — each node extraction is O(log n) instead of
    the O(n) scan a naive list would require.
  • Rule engine evaluation: O(R) where R = number of rules (linear scan).
"""

from __future__ import annotations

import heapq
import json
import os
import re
import uuid
from typing import Any, Callable

import httpx
from dotenv import load_dotenv

from database import save_query, is_connected

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Gemini / OpenAI API key loaded from environment
LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini")  # "gemini" | "openai"

# ---------------------------------------------------------------------------
# Module A: LLM-Based Natural Language Classification
# ---------------------------------------------------------------------------

# The system prompt instructs the LLM to return a strict JSON frame.
_SYSTEM_PROMPT = """You are a university query classifier. Given a student's 
query, extract EXACTLY the following JSON (no markdown, no extra keys):
{
  "Type": "<one of: Fee, Exam, Registration, Library, Scholarship, Hostel, General>",
  "Subject": "<brief subject description, max 10 words>",
  "Urgency": "<one of: Low, Medium, High, Critical>",
  "Handler": "<suggested first-contact department>"
}
Return ONLY valid JSON. No explanation."""


async def classify_query_llm(query_text: str) -> dict[str, str]:
    """
    Sends the raw student query to an LLM and parses the structured
    JSON frame that the model returns.

    Falls back to a deterministic mock classifier when no API key is
    configured — useful for local development and testing.
    """
    if not LLM_API_KEY:
        return _mock_classify(query_text)

    try:
        if LLM_PROVIDER == "gemini":
            return await _classify_gemini(query_text)
        else:
            return await _classify_openai(query_text)
    except Exception as exc:
        print(f"[LLM ERROR] {exc} — falling back to mock classifier")
        return _mock_classify(query_text)


async def _classify_gemini(query_text: str) -> dict[str, str]:
    """Call Google Gemini API (v1beta generateContent)."""
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-2.0-flash:generateContent"
    )
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": f"{_SYSTEM_PROMPT}\n\nStudent query: {query_text}"}
                ]
            }
        ]
    }
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            url,
            json=payload,
            params={"key": LLM_API_KEY},
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        # Strip markdown fences if present
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*", "", text)
        return json.loads(text.strip())


async def _classify_openai(query_text: str) -> dict[str, str]:
    """Call OpenAI ChatCompletion API."""
    url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": query_text},
        ],
        "temperature": 0.0,
    }
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            url,
            json=payload,
            headers={
                "Authorization": f"Bearer {LLM_API_KEY}",
                "Content-Type": "application/json",
            },
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*", "", text)
        return json.loads(text.strip())


def _mock_classify(query_text: str) -> dict[str, str]:
    """
    Deterministic keyword-based classifier used when no LLM API key is
    present.  Provides a reasonable default for demonstration purposes.
    """
    q = query_text.lower()
    if any(kw in q for kw in ["fee", "payment", "tuition", "refund", "dues"]):
        return {
            "Type": "Fee",
            "Subject": "Fee / payment inquiry",
            "Urgency": "Medium",
            "Handler": "Accounts",
        }
    if any(kw in q for kw in ["exam", "grade", "result", "marks", "gpa"]):
        return {
            "Type": "Exam",
            "Subject": "Examination inquiry",
            "Urgency": "High",
            "Handler": "Examination Cell",
        }
    if any(kw in q for kw in ["register", "enroll", "course", "drop", "add"]):
        return {
            "Type": "Registration",
            "Subject": "Course registration inquiry",
            "Urgency": "Medium",
            "Handler": "Registrar",
        }
    if any(kw in q for kw in ["library", "book", "borrow", "fine", "overdue"]):
        return {
            "Type": "Library",
            "Subject": "Library services inquiry",
            "Urgency": "Low",
            "Handler": "Library",
        }
    if any(kw in q for kw in ["scholarship", "financial aid", "grant", "merit"]):
        return {
            "Type": "Scholarship",
            "Subject": "Scholarship inquiry",
            "Urgency": "High",
            "Handler": "Financial Aid Office",
        }
    if any(kw in q for kw in ["hostel", "dorm", "room", "accommodation"]):
        return {
            "Type": "Hostel",
            "Subject": "Hostel accommodation inquiry",
            "Urgency": "Medium",
            "Handler": "Hostel Administration",
        }
    # Default / General
    return {
        "Type": "General",
        "Subject": "General inquiry",
        "Urgency": "Low",
        "Handler": "Student Affairs",
    }


# ---------------------------------------------------------------------------
# Module B: Rule-Based Expert System (Semantic Net)
# ---------------------------------------------------------------------------

# Semantic net modelled as a dict-of-dicts.
# Each query Type maps to a department and an immediate resolution message.
SEMANTIC_NET: dict[str, dict[str, str]] = {
    "Fee": {
        "department": "Accounts",
        "resolution": (
            "Your fee-related query has been routed to the Accounts department. "
            "Standard processing time is 2-3 business days."
        ),
    },
    "Exam": {
        "department": "Examination Cell",
        "resolution": (
            "Your examination query has been forwarded to the Examination Cell. "
            "Please check the portal for real-time updates."
        ),
    },
    "Registration": {
        "department": "Registrar",
        "resolution": (
            "Your registration query has been sent to the Registrar's Office. "
            "Ensure your documents are uploaded on the student portal."
        ),
    },
    "Library": {
        "department": "Library",
        "resolution": (
            "Your library query is being handled by the Library desk. "
            "Visit the library portal for self-service options."
        ),
    },
    "Scholarship": {
        "department": "Financial Aid Office",
        "resolution": (
            "Your scholarship query has been routed to the Financial Aid Office. "
            "Check eligibility criteria on the scholarship portal."
        ),
    },
    "Hostel": {
        "department": "Hostel Administration",
        "resolution": (
            "Your hostel query has been forwarded to Hostel Administration. "
            "Room allotment updates are published weekly."
        ),
    },
}

# Propositional logic rules — evaluated in order.
# Each rule is a (condition_fn, action) tuple.
RULES: list[tuple[Callable, str]] = [
    (
        lambda f: f.get("Type") == "Fee" and f.get("Urgency") in ("High", "Critical"),
        "ESCALATE: Urgent fee issue — route to Senior Accounts Officer.",
    ),
    (
        lambda f: f.get("Type") == "Exam" and f.get("Urgency") == "Critical",
        "ESCALATE: Critical exam issue — route to Controller of Examinations.",
    ),
    (
        lambda f: f.get("Type") in SEMANTIC_NET,
        "RESOLVE: Standard rule match — direct department routing applied.",
    ),
]


def evaluate_rules(frame: dict[str, str]) -> dict[str, Any]:
    """
    Fire propositional logic rules against the classified frame.

    Returns:
        dict with keys:
          - resolved (bool): True if a standard rule handled the query.
          - department (str): Target department.
          - resolution (str): Human-readable resolution text.
          - rule_fired (str): Description of the rule that matched.
          - needs_escalation (bool): True when advanced routing is required.
    """
    for condition, action in RULES:
        if condition(frame):
            query_type = frame.get("Type", "General")
            net_entry = SEMANTIC_NET.get(query_type)

            if action.startswith("ESCALATE"):
                return {
                    "resolved": False,
                    "department": net_entry["department"] if net_entry else "Student Affairs",
                    "resolution": action,
                    "rule_fired": action,
                    "needs_escalation": True,
                }
            else:
                return {
                    "resolved": True,
                    "department": net_entry["department"],
                    "resolution": net_entry["resolution"],
                    "rule_fired": action,
                    "needs_escalation": False,
                }

    # No rule matched — escalate through A* routing
    return {
        "resolved": False,
        "department": "Student Affairs",
        "resolution": "No standard rule matched. Initiating advanced A* routing.",
        "rule_fired": "NONE",
        "needs_escalation": True,
    }


# ---------------------------------------------------------------------------
# Module C: A* Search with Min-Heap — Optimal Department Escalation
# ---------------------------------------------------------------------------

"""
=== Data Structure: Min-Heap (Priority Queue) ===

A Min-Heap is a complete binary tree where each parent node's key is ≤ its
children's keys.  This gives us:

  • insert (heappush):  O(log n)  — sift-up from leaf to root
  • extract-min (heappop): O(log n)  — replace root, sift-down
  • peek-min:  O(1)

We use Python's `heapq` module which implements a binary min-heap over a
plain list.  Each element in the heap is a tuple:

    (f_cost, unique_counter, node_name)

The unique_counter breaks ties so that nodes with identical f-costs are
extracted in FIFO order, preventing comparison errors on the node_name
string when f-costs collide.

=== A* Algorithm ===

A* combines Dijkstra's uniform-cost search with a heuristic estimate:

    f(n) = g(n) + h(n)

  • g(n) = actual cost from start to node n
  • h(n) = heuristic estimate of cost from n to goal
  • f(n) = estimated total cost of the cheapest path through n

The algorithm is **optimal** when h(n) is admissible (never overestimates)
and **efficient** because the Min-Heap ensures we always expand the most
promising node first in O(log n) time.

Time Complexity:  O(b^d) worst-case, where b = branching factor, d = depth.
Space Complexity: O(b^d) for the open and closed lists.
With a good heuristic, practical performance is much better.
"""

# Mocked institutional department graph.
# Edges are (neighbour, cost) pairs.  Cost represents administrative
# overhead / processing time units.
DEPARTMENT_GRAPH: dict[str, list[tuple[str, int]]] = {
    "Student Affairs":       [("Registrar", 2), ("Accounts", 4), ("Library", 3)],
    "Registrar":             [("Student Affairs", 2), ("Examination Cell", 3), ("Dean Office", 5)],
    "Accounts":              [("Student Affairs", 4), ("Financial Aid Office", 2), ("Dean Office", 6)],
    "Library":               [("Student Affairs", 3), ("Registrar", 4)],
    "Examination Cell":      [("Registrar", 3), ("Dean Office", 4)],
    "Financial Aid Office":  [("Accounts", 2), ("Dean Office", 3)],
    "Dean Office":           [("Registrar", 5), ("Accounts", 6), ("Examination Cell", 4),
                              ("Financial Aid Office", 3), ("Hostel Administration", 2)],
    "Hostel Administration": [("Dean Office", 2), ("Student Affairs", 5)],
}

# Heuristic table — straight-line (admissible) estimates to the Dean Office,
# the highest-authority escalation point.
_HEURISTICS_TO_DEAN: dict[str, int] = {
    "Student Affairs":       6,
    "Registrar":             4,
    "Accounts":              5,
    "Library":               7,
    "Examination Cell":      3,
    "Financial Aid Office":  2,
    "Dean Office":           0,
    "Hostel Administration": 2,
}


class MinHeapOpenList:
    """
    Custom Min-Heap wrapper around Python's heapq for the A* open list.

    Each entry is a tuple (f_cost, counter, node_name) so that:
      1. Nodes are ordered by ascending f_cost.
      2. Ties are broken by insertion order (counter) — FIFO.
      3. We avoid comparing node_name strings directly.

    === Complexity ===
      push : O(log n) — heapq.heappush sifts up
      pop  : O(log n) — heapq.heappop sifts down
      peek : O(1)     — heap[0]
      len  : O(1)
    """

    def __init__(self) -> None:
        self._heap: list[tuple[float, int, str]] = []
        self._counter: int = 0  # monotonically increasing tie-breaker

    def push(self, f_cost: float, node: str) -> None:
        """Insert a node with its f-cost into the min-heap.  O(log n)."""
        heapq.heappush(self._heap, (f_cost, self._counter, node))
        self._counter += 1

    def pop(self) -> tuple[float, str]:
        """
        Extract and return the node with the smallest f-cost.  O(log n).

        Returns:
            (f_cost, node_name)
        """
        f_cost, _counter, node = heapq.heappop(self._heap)
        return f_cost, node

    def peek(self) -> tuple[float, str]:
        """Return the smallest-f-cost node without removing it.  O(1)."""
        f_cost, _counter, node = self._heap[0]
        return f_cost, node

    def __len__(self) -> int:
        return len(self._heap)

    def __bool__(self) -> bool:
        return bool(self._heap)

    def __repr__(self) -> str:
        return f"MinHeapOpenList(size={len(self._heap)})"


def astar_search(
    start: str,
    goal: str,
    graph: dict[str, list[tuple[str, int]]] | None = None,
    heuristics: dict[str, int] | None = None,
) -> dict[str, Any]:
    """
    A* Pathfinding through the university department graph.

    Parameters:
        start      — Starting department node.
        goal       — Target department node (escalation endpoint).
        graph      — Adjacency list; defaults to DEPARTMENT_GRAPH.
        heuristics — h(n) values; defaults to _HEURISTICS_TO_DEAN.

    Returns:
        dict with:
          path       — Ordered list of departments from start to goal.
          total_cost — The optimal g(goal) cost.
          explored   — Number of nodes expanded.
          explanation — Step-by-step trace of the algorithm for transparency.

    === Algorithm Walkthrough ===
    1. Initialise the open list (Min-Heap) with the start node, f = h(start).
    2. While the open list is not empty:
       a. Pop the node n with the lowest f(n) from the Min-Heap.  O(log n)
       b. If n == goal → reconstruct and return the path.
       c. For each neighbour m of n:
          - Compute tentative g(m) = g(n) + cost(n, m).
          - If g(m) < previously recorded g(m):
            • Update g(m) and set parent(m) = n.
            • Push m into the Min-Heap with f(m) = g(m) + h(m).   O(log n)
    3. If open list empties without reaching goal → no path exists.
    """
    if graph is None:
        graph = DEPARTMENT_GRAPH
    if heuristics is None:
        heuristics = _HEURISTICS_TO_DEAN

    # Validate inputs
    if start not in graph:
        return {"path": [start], "total_cost": 0, "explored": 0,
                "explanation": f"Start node '{start}' not in graph."}
    if goal not in graph:
        return {"path": [start], "total_cost": 0, "explored": 0,
                "explanation": f"Goal node '{goal}' not in graph."}

    # --- A* Initialisation ---
    open_list = MinHeapOpenList()                 # Min-Heap open list
    g_costs: dict[str, float] = {start: 0}        # g(n): actual cost from start
    parents: dict[str, str | None] = {start: None}  # For path reconstruction
    closed_set: set[str] = set()                   # Already-expanded nodes
    explanation_steps: list[str] = []              # Algorithm trace

    # Push start node: f(start) = g(start) + h(start) = 0 + h(start)
    h_start = heuristics.get(start, 0)
    open_list.push(h_start, start)
    explanation_steps.append(
        f"INIT: Push '{start}' onto Min-Heap with f={h_start} "
        f"(g=0, h={h_start})."
    )

    explored_count = 0

    # --- A* Main Loop ---
    while open_list:
        # Step 2a: Extract node with lowest f(n) — O(log n) via Min-Heap
        current_f, current = open_list.pop()
        explanation_steps.append(
            f"POP : '{current}' from Min-Heap (f={current_f}). "
            f"Heap size now {len(open_list)}."
        )

        # Skip if already expanded (duplicate entries in heap)
        if current in closed_set:
            explanation_steps.append(f"SKIP: '{current}' already in closed set.")
            continue

        closed_set.add(current)
        explored_count += 1

        # Step 2b: Goal check
        if current == goal:
            path = _reconstruct_path(parents, goal)
            explanation_steps.append(
                f"GOAL: '{goal}' reached! Optimal cost = {g_costs[goal]}."
            )
            return {
                "path": path,
                "total_cost": g_costs[goal],
                "explored": explored_count,
                "explanation": "\n".join(explanation_steps),
            }

        # Step 2c: Expand neighbours
        for neighbour, edge_cost in graph.get(current, []):
            if neighbour in closed_set:
                continue

            tentative_g = g_costs[current] + edge_cost

            # Only update if we found a cheaper path to neighbour
            if tentative_g < g_costs.get(neighbour, float("inf")):
                g_costs[neighbour] = tentative_g
                parents[neighbour] = current
                h_n = heuristics.get(neighbour, 0)
                f_n = tentative_g + h_n

                # Push into Min-Heap — O(log n)
                open_list.push(f_n, neighbour)
                explanation_steps.append(
                    f"PUSH: '{neighbour}' onto Min-Heap with "
                    f"f={f_n} (g={tentative_g}, h={h_n}). "
                    f"Parent='{current}'."
                )

    # Step 3: No path found
    explanation_steps.append("FAIL: Open list exhausted — no path to goal.")
    return {
        "path": [],
        "total_cost": float("inf"),
        "explored": explored_count,
        "explanation": "\n".join(explanation_steps),
    }


def _reconstruct_path(parents: dict[str, str | None], goal: str) -> list[str]:
    """Walk the parent pointers from goal back to start. O(d) where d = depth."""
    path = []
    node: str | None = goal
    while node is not None:
        path.append(node)
        node = parents.get(node)
    path.reverse()
    return path


# ---------------------------------------------------------------------------
# Public API: Full Resolution Pipeline
# ---------------------------------------------------------------------------

async def resolve_query(student_id: str, query_text: str) -> dict[str, Any]:
    """
    End-to-end resolution pipeline:
      1. Classify the query via LLM → structured JSON frame.
      2. Evaluate the frame against the rule-based expert system.
      3. If escalation is needed, run A* search for optimal routing.
      4. Return a comprehensive response with routing + explanation.
    """
    query_id = str(uuid.uuid4())[:8]

    # --- Step 1: NLP Classification ---
    frame = await classify_query_llm(query_text)
    frame["QueryID"] = query_id
    frame["StudentID"] = student_id

    # --- Step 2: Rule-Based Evaluation ---
    rule_result = evaluate_rules(frame)

    # --- Step 3: A* Escalation (if needed) ---
    astar_result = None
    if rule_result["needs_escalation"]:
        start_dept = rule_result["department"]
        goal_dept = "Dean Office"  # Highest escalation authority
        astar_result = astar_search(start_dept, goal_dept)

    # --- Step 4: Compose Response ---
    response = {
        "queryId": query_id,
        "studentId": student_id,
        "classification": frame,
        "routing": {
            "resolved": rule_result["resolved"],
            "department": rule_result["department"],
            "resolution": rule_result["resolution"],
            "ruleFired": rule_result["rule_fired"],
        },
        "escalation": None,
    }

    if astar_result:
        response["escalation"] = {
            "path": astar_result["path"],
            "totalCost": astar_result["total_cost"],
            "nodesExplored": astar_result["explored"],
            "explanation": astar_result["explanation"],
        }

    # --- Step 5: Persist to Neon Postgres (non-blocking) ---
    if is_connected():
        await save_query(
            query_id=query_id,
            student_id=student_id,
            query_text=query_text,
            classification=frame,
            routing=response["routing"],
            escalation=response.get("escalation"),
        )

    return response

