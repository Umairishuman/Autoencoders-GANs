# Build from the repository root:  docker build -f backend/Dockerfile -t restoration-lab-api .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src:/app/backend \
    MODELS_DIR=/app/models \
    SAMPLES_DIR=/app/backend/samples

WORKDIR /app
COPY backend/requirements.txt backend/requirements.txt
RUN pip install -r backend/requirements.txt

# pet_data.py is the SAME module that produced the training/validation/test corruptions
COPY src/pet_data.py src/pet_data.py
COPY backend/app backend/app
COPY backend/samples backend/samples

# The ONNX models are not baked into the image: mount them at /app/models (see docker-compose.yml)
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=90s --retries=5 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/api/health').status==200 else 1)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "/app/backend"]
