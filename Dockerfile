FROM python:3.9-slim

WORKDIR /backend

# Copy backend folder
COPY backend/ /backend

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
