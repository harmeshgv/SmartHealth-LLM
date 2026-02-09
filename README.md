# SmartHealth-LLM

SmartHealth-LLM is a multi-agent health assistant backend + frontend stack.

It includes:
- FastAPI backend with agent orchestration
- Specialized agents (`conversation`, `symptom_matcher`, `disease_info`, `reasoning`)
- Local/vector retrieval + optional internet fallback
- Built-in run metrics collection and export
- Excel-based evaluation runner for batch query testing

## Repository Structure

```text
backend/               FastAPI app, agents, tools, prompts, models
frontend/              React client
scripts/               Bootstrap + evaluation scripts
tests/                 Pytest suites
Dockerfile.backend     Backend container
docker-compose.yml     Full stack local docker run
```

## Reproducible Setup (Fresh Clone)

### Option A: One-command bootstrap (recommended)

```bash
./scripts/bootstrap.sh
```

This does:
- create `.venv`
- install backend dependencies
- install frontend dependencies (`npm ci`)
- create `backend/.env` from `backend/.env.example` if missing

### Option B: Make targets

```bash
make setup
```

Useful commands:
- `make dev-backend`
- `make dev-frontend`
- `make test-backend`
- `make test-all`
- `make docker-up`
- `make docker-down`

## Environment Configuration

Copy and edit backend env file:

```bash
cp backend/.env.example backend/.env
```

Important vars:
- `GROQ_API_KEY` (if using Groq adapter)
- `SERPER_API_KEY` (optional, enables live web fallback)
- `OLLAMA_HOST` (default `http://localhost:11434`)

## Run Locally

### Backend

```bash
source .venv/bin/activate
cd backend
uvicorn app.main:app --reload --port 7860
```

Backend URL: `http://localhost:7860`

### Frontend

```bash
cd frontend
npm start
```

Frontend URL: `http://localhost:3000`

Optional frontend API base override:

```bash
REACT_APP_API_URL=http://localhost:7860 npm start
```

## Run with Docker

```bash
cp backend/.env.example backend/.env
# fill required keys in backend/.env

docker compose up --build
```

URLs:
- Frontend: `http://localhost:3000`
- Backend: `http://localhost:7860`

## How To Use (End User Flow)

1. Open frontend at `http://localhost:3000`
2. Start chat with normal text:
   - casual message -> conversation agent path
   - symptom/disease question -> medical agent workflow
3. Backend endpoint used by frontend: `POST /chat/send`
4. Optional debug run with logs: `POST /debug/debug_chat_send`
5. Check run analytics with:
   - `GET /metrics/summary`
   - `GET /metrics/runs`

## Backend API

### Core
- `GET /` -> backend status message
- `GET /health/status` -> `{"status":"ok"}`
- `GET /health/ping`
- `GET /health/live`
- `GET /health/ready`

### Chat
- `POST /chat/send`
- `POST /chat/history`
- `POST /chat/clear`

Example:

```bash
curl -X POST http://localhost:7860/chat/send \
  -H "Content-Type: application/json" \
  -d '{"message":"I have fever and cough","session_id":"demo-1"}'
```

### Debug
- `POST /debug/debug_chat_send`

### Metrics
- `GET /metrics/summary`
- `GET /metrics/runs?limit=50`
- `POST /metrics/save-local`
- `POST /metrics/reset`

Save metrics locally:

```bash
curl -X POST http://localhost:7860/metrics/save-local \
  -H "Content-Type: application/json" \
  -d '{"filepath":"metrics_store/session_metrics.json","limit":500}'
```

## Metrics Captured

Per run:
- routing: intent, planned/executed agents
- tool usage: local DB calls/success, vector DB calls/success, internet calls/success
- memory usage: recall/save counts, context items used
- latency and status
- final output and relevance score

Aggregate summary fields include:
- `local_data_usage_rate`
- `internet_usage_rate`
- `web_fallback_rate`
- `local_hit_success_rate`
- `avg_relevance_score`
- `avg_latency_ms`, `p95_latency_ms`
- `medical_query_rate`, `conversation_query_rate`

## Excel Evaluation Workflow

Script: `scripts/run_excel_eval.py`

### 1) Create template

```bash
source .venv/bin/activate
python scripts/run_excel_eval.py --input eval_queries.xlsx --create-template
```

This creates an Excel file with `queries` column.

### 2) Fill queries

Put one query per row under `queries`.

### 3) Run batch evaluation

```bash
python scripts/run_excel_eval.py \
  --input eval_queries.xlsx \
  --output eval_queries_evaluated.xlsx
```

The output file writes results back in the same row with columns like:
- `run_id`, `status`, `error`
- `intent`, `agents_planned`, `agents_executed`
- `conversation_output`, `symptom_matcher_output`, `disease_info_output`, `reasoning_output`
- `final_output`
- metrics columns (`latency_ms`, relevance, local/internet usage, memory usage, tool errors)

## Tests

Backend tests:

```bash
source .venv/bin/activate
pytest -q tests/backend
```

All tests:

```bash
source .venv/bin/activate
pytest -q
```

## Hosting Options

Free-tier details change over time. The notes below are accurate as checked on **February 9, 2026**.

### 1) Hugging Face Spaces (best truly free option for demos)

Why:
- Free CPU Basic hardware is available.
- Good for public demo sharing.

How:
1. Create a new **Docker Space** on Hugging Face.
2. Connect your GitHub repo (or push repo files directly).
3. Ensure `backend/.env` values are set as Space Secrets (for keys).
4. Build/deploy using `Dockerfile.backend` or full-stack approach you choose.
5. Verify health endpoint after deployment.

Notes:
- Free hardware has limits (CPU/RAM/storage).
- Disk persistence on default free setup is limited/non-persistent for app runtime data.
- Official references:
  - https://huggingface.co/pricing
  - https://huggingface.co/docs/hub/en/spaces-overview

### 2) Render (good free preview, not for heavy production)

Why:
- Free web services are available for testing/hobby preview.

How:
1. Create a Render account and connect GitHub repo.
2. Create a new Web Service from this repo.
3. Set build/start commands for backend:
   - Build: `pip install -r backend/requirements.txt`
   - Start: `cd backend && uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. Add environment variables from `backend/.env.example`.
5. Deploy and test `GET /health/status`.

Notes:
- Free services have usage/feature limits and are not recommended for production.
- Official references:
  - https://render.com/docs/free

### 3) Railway (trial-friendly, but not fully free long-term)

Why:
- Very quick deploy workflow.

How:
1. Create Railway project from GitHub repo.
2. Add backend service with start command:
   - `cd backend && uvicorn app.main:app --host 0.0.0.0 --port $PORT`
3. Add env vars from `backend/.env.example`.
4. Deploy and validate health + chat endpoints.

Notes:
- Current model includes trial credits and then low-cost paid usage.
- Use it for fast testing if free budget is acceptable.
- Official references:
  - https://railway.com/pricing
  - https://docs.railway.com/reference/pricing/free-trial

### 4) AWS / GCP / Azure (not free for real workloads)

Best when you need:
- reliability, scaling control, networking/security compliance.

### 5) Fly.io

Notes:
- Historical free allowances changed; verify current plan terms before choosing.
- Use mainly if you want Fly’s multi-region container model.

## Additional Deployment/Setup Notes

Detailed reproducible setup and hosting notes:
- `docs/REPRODUCIBLE_SETUP_AND_HOSTING.md`

## License

MIT
