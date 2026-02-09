.PHONY: setup setup-backend setup-frontend dev-backend dev-frontend test-backend test-all docker-up docker-down eval-excel

setup: setup-backend setup-frontend

setup-backend:
	python3 -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip setuptools wheel && pip install -r backend/requirements.txt
	@if [ ! -f backend/.env ]; then cp backend/.env.example backend/.env; fi

setup-frontend:
	cd frontend && npm ci

dev-backend:
	. .venv/bin/activate && cd backend && uvicorn app.main:app --reload --port 7860

dev-frontend:
	cd frontend && npm start

test-backend:
	. .venv/bin/activate && pytest -q tests/backend

test-all:
	. .venv/bin/activate && pytest -q

docker-up:
	docker compose up --build

docker-down:
	docker compose down

eval-excel:
	. .venv/bin/activate && python scripts/run_excel_eval.py --help
