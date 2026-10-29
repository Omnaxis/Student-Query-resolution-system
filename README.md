# Student Query Resolution System

A microservice-based automated query resolution platform that classifies, routes, and resolves student administrative and academic inquiries using AI, rule-based expert systems, and A* pathfinding with Min-Heap optimisation.

> Built for **CCSE0301 Data Structures and Algorithms & Artificial Intelligence** coursework.

---

## Architecture

```
┌─────────────────┐       ┌───────────────────┐       ┌─────────────────────────┐
│   Frontend      │       │  API Gateway      │       │  Logic Engine           │
│   (Vercel)      │──────▶│  Node.js/Express  │──────▶│  Python/FastAPI         │
│   HTML/CSS/JS   │       │  Port 3000        │       │  Port 8000              │
│                 │◀──────│                   │◀──────│                         │
└─────────────────┘       └───────────────────┘       │  ┌─ NLP Classifier      │
                                                      │  ├─ Rule Engine         │
                                                      │  └─ A* + Min-Heap      │
                                                      └─────────────────────────┘
```

## Core DSA Concepts

### Min-Heap (Priority Queue)
- Used as the **open list** in the A* algorithm
- **O(log n)** insert and extract-min operations
- Ensures the most promising node is always expanded first

### A* Search Algorithm
- Finds the **optimal department escalation path** through a mocked institutional graph
- Uses `f(n) = g(n) + h(n)` with admissible heuristics
- **Time Complexity:** O(b^d) worst-case; much better in practice with good heuristics

### Rule-Based Expert System
- Propositional logic rules fire against classified query frames
- Semantic net maps query types to university departments

---

## Project Structure

```
Student-Query-Resolution-System/
├── api-gateway/                # Microservice 1: Node.js Express
│   ├── server.js               # Express server & proxy logic
│   ├── package.json
│   ├── .env.example
│   └── .gitignore
├── logic-engine/               # Microservice 2: Python FastAPI
│   ├── main.py                 # FastAPI application entry point
│   ├── logic.py                # NLP + Rule Engine + A* Search
│   ├── requirements.txt
│   ├── .env.example
│   └── .gitignore
├── frontend/                   # Frontend: Vercel deployment
│   ├── index.html              # Single-file chat interface
│   └── vercel.json             # Vercel routing configuration
├── render.yaml                 # Render deployment blueprint
├── .gitignore
└── README.md
```

---

## Local Development

### Prerequisites
- Node.js ≥ 18
- Python ≥ 3.10
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/Omnaxis/Student-Query-resolution-system.git
cd Student-Query-resolution-system
```

### 2. Start the Python Logic Engine

```bash
cd logic-engine
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # Edit .env to add your LLM_API_KEY (optional)
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Start the Node.js API Gateway

```bash
cd api-gateway
npm install
cp .env.example .env            # Verify PYTHON_SERVICE_URL=http://localhost:8000
npm start
```

### 4. Open the Frontend

Open `frontend/index.html` in your browser, or serve it with:

```bash
cd frontend
npx serve .
```

---

## Deployment

### Backend on Render

Both services can be deployed as **Web Services** on [Render](https://render.com). A `render.yaml` Blueprint is included for one-click deploy.

**Steps:**
1. Push the repo to GitHub.
2. Connect your Render account to the GitHub repo.
3. Create a **Blueprint** from `render.yaml`, or manually create two Web Services:

| Service        | Root Directory   | Build Command                       | Start Command                                        |
|----------------|------------------|-------------------------------------|------------------------------------------------------|
| API Gateway    | `api-gateway`    | `npm install`                       | `node server.js`                                     |
| Logic Engine   | `logic-engine`   | `pip install -r requirements.txt`   | `uvicorn main:app --host 0.0.0.0 --port $PORT`       |

4. Set environment variables on each service:

   **API Gateway:**
   - `PYTHON_SERVICE_URL` = `https://<your-logic-engine>.onrender.com`

   **Logic Engine:**
   - `LLM_API_KEY` = your Gemini/OpenAI key (optional — mock classifier works without it)
   - `LLM_PROVIDER` = `gemini` or `openai`

### Frontend on Vercel

1. In the Vercel dashboard, import the repo.
2. Set the **Root Directory** to `frontend`.
3. Framework preset: **Other**.
4. After deploy, update the `API_URL` in `index.html`:
   ```js
   const API_URL = "https://<your-api-gateway>.onrender.com";
   ```

---

## Environment Variables

### API Gateway (`api-gateway/.env`)

| Variable             | Default                  | Description                          |
|----------------------|--------------------------|--------------------------------------|
| `PORT`               | `3000`                   | Server port                          |
| `PYTHON_SERVICE_URL` | `http://localhost:8000`  | URL of the Python Logic Engine       |

### Logic Engine (`logic-engine/.env`)

| Variable        | Default   | Description                                         |
|-----------------|-----------|-----------------------------------------------------|
| `LLM_API_KEY`   | *(empty)* | Gemini or OpenAI API key (leave blank for mock mode) |
| `LLM_PROVIDER`  | `gemini`  | `gemini` or `openai`                                |

---

## API Reference

### `POST /api/resolve` (API Gateway → Frontend)

**Request:**
```json
{
  "studentId": "STU001",
  "queryText": "How do I pay my tuition fee?"
}
```

**Response:**
```json
{
  "queryId": "a1b2c3d4",
  "studentId": "STU001",
  "classification": {
    "Type": "Fee",
    "Subject": "Fee / payment inquiry",
    "Urgency": "Medium",
    "Handler": "Accounts"
  },
  "routing": {
    "resolved": true,
    "department": "Accounts",
    "resolution": "Your fee-related query has been routed to the Accounts department...",
    "ruleFired": "RESOLVE: Standard rule match — direct department routing applied."
  },
  "escalation": null
}
```

---

## License

MIT

