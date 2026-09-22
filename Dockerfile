FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.lock ./
RUN pip install --no-cache-dir -r requirements.lock

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps -e .

COPY AGENTS.md ./
COPY docs ./docs
COPY tests ./tests

CMD ["prism-rag", "run-demo", "--domain", "cloud"]
