# Reproducible Setup and Hosting

## 1) Local Reproducible Setup (Fresh Clone)

```bash
git clone <your-repo-url>
cd SmartHealth-LLM
./scripts/bootstrap.sh
```

What it does:
- Creates `.venv`
- Installs backend dependencies
- Installs frontend dependencies (`npm ci`)
- Creates `backend/.env` from `backend/.env.example` if missing

Then:

```bash
# Terminal 1
source .venv/bin/activate
cd backend
uvicorn app.main:app --reload --port 7860

# Terminal 2
cd frontend
npm start
```

## 2) Docker Reproducible Setup

```bash
cp backend/.env.example backend/.env
# fill API keys in backend/.env
docker compose up --build
```

Services:
- Backend: `http://localhost:7860`
- Frontend: `http://localhost:3000`

## 3) Reproducible Evaluation Run (Excel)

Create query sheet template:

```bash
source .venv/bin/activate
python scripts/run_excel_eval.py --input eval_queries.xlsx --create-template
```

Add queries under the `queries` column, then run:

```bash
python scripts/run_excel_eval.py --input eval_queries.xlsx --output eval_queries_evaluated.xlsx
```

## 4) Hosting Options

### Option A: Hugging Face Spaces (good for demo)
- Best for quick public demo.
- Use Docker Space with current Dockerfiles.
- Pros: easiest visibility, free tiers.
- Cons: cold starts, resource limits.

### Option B: Render
- Deploy backend web service + static frontend.
- Pros: simple CI/CD from GitHub.
- Cons: free tier sleeps, limited CPU/GPU.

### Option C: Railway
- Fast Docker deploy, easy env vars.
- Pros: straightforward setup, managed platform.
- Cons: paid usage scales with runtime.

### Option D: AWS (EC2 + optional ECS)
- Best control for production.
- Pros: scalable, VPC/network/security control.
- Cons: higher DevOps complexity.

### Option E: Fly.io
- Good for containerized API near users.
- Pros: global deployment, Docker-native.
- Cons: learning curve for persistent storage/ops.

## 5) Recommended Path

1. Start with `Render` or `Railway` for backend + frontend split.
2. Keep Docker setup as source of truth.
3. Move to AWS when traffic, compliance, or control needs increase.
