#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
FRONTEND_DIR="$ROOT_DIR/frontend"
VENV_DIR="$ROOT_DIR/.venv"

echo "==> SmartHealth bootstrap starting..."

if [[ ! -d "$VENV_DIR" ]]; then
  echo "==> Creating Python virtualenv at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

echo "==> Installing backend dependencies"
"$VENV_DIR/bin/pip" install --upgrade pip setuptools wheel
"$VENV_DIR/bin/pip" install -r "$BACKEND_DIR/requirements.txt"

if [[ ! -f "$BACKEND_DIR/.env" ]]; then
  echo "==> Creating backend/.env from backend/.env.example"
  cp "$BACKEND_DIR/.env.example" "$BACKEND_DIR/.env"
fi

echo "==> Installing frontend dependencies"
cd "$FRONTEND_DIR"
npm ci

echo "==> Bootstrap complete"
echo "Next steps:"
echo "1) Edit backend/.env and add keys if needed"
echo "2) Start backend:  source .venv/bin/activate && cd backend && uvicorn app.main:app --reload --port 7860"
echo "3) Start frontend: cd frontend && npm start"
