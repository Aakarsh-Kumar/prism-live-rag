FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[dev]"

COPY AGENTS.md ./
COPY docs ./docs
COPY tests ./tests

CMD ["prism-rag", "run-demo", "--domain", "cloud"]
