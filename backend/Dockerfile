# ---------- Base Image ----------
FROM python:3.12-slim

# ---------- Environment ----------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# ---------- Working Directory ----------
WORKDIR /app

# ---------- System Dependencies ----------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 && \
    rm -rf /var/lib/apt/lists/*

# ---------- Copy Files ----------
COPY . .

# ---------- Python Dependencies ----------
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
        torch==2.4.1+cpu \
        torchvision==0.19.1+cpu \
        --index-url https://download.pytorch.org/whl/cpu && \
    pip install --prefer-binary -r requirements.fastapi.txt

# ---------- Expose Port ----------
EXPOSE 8000
# ---------- Hugging Face Cache Fix ----------
ENV HF_HOME=/tmp/hf_cache
ENV TRANSFORMERS_CACHE=/tmp/hf_cache

# ---------- Run FastAPI ----------
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]

