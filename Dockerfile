FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# CPU-only PyTorch: the embedder is small, and the CUDA build would add ~4 GB to the image
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements-docker.txt .
RUN pip install -r requirements-docker.txt

COPY src/ src/
COPY data/policies/ data/policies/
COPY data/eval/ data/eval/

# Build the database and the vector index inside the image (also downloads bge-small once)
RUN python src/build_db.py && python src/build_index.py

# Run as a normal user, not root
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

# At runtime: no Hugging Face downloads, and Ollama runs on the host machine (GPU)
ENV HF_HUB_OFFLINE=1 \
    OLLAMA_HOST=http://host.docker.internal:11434

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=5)"
CMD ["uvicorn", "api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]