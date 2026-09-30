FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PRISM_ROOT=/app EMBEDDING_BACKEND=fastembed EMBEDDING_DEVICE=cpu EMBEDDING_LOCAL_FILES_ONLY=true
ENV EMBEDDING_CACHE_DIR=/app/.cache/fastembed HF_HOME=/opt/models/huggingface
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
ENV OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

WORKDIR /app

COPY requirements.lock ./
RUN pip install --no-cache-dir -r requirements.lock
COPY requirements-cpu.lock ./
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.14.0 \
    && pip install --no-cache-dir -r requirements-cpu.lock

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps .

COPY AGENTS.md ./
COPY docs ./docs
COPY tests ./tests
COPY scripts ./scripts
RUN useradd --create-home --uid 1000 prism
COPY --chown=prism:prism submission-assets/data ./data
COPY --chown=prism:prism submission-assets/lancedb ./.cache/lancedb
COPY --chown=prism:prism submission-assets/fastembed ./.cache/fastembed
COPY --chown=prism:prism submission-assets/huggingface /opt/models/huggingface
RUN mkdir -p /app/.cache/telemetry && chown prism:prism /app/.cache/telemetry
USER prism
EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=5s --start-period=300s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/api/health', timeout=4)"

CMD ["prism-rag", "serve", "--host", "0.0.0.0", "--port", "8080"]
