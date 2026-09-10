FROM python:3.11-slim

WORKDIR /app
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY config/ ./config/
COPY src/ ./src/
RUN mkdir -p data/processed data/models data/raw

EXPOSE 8000 8501
CMD ["uvicorn", "twin.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
