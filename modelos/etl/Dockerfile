ARG PY_BASE_IMAGE=python:3.11-slim
FROM ${PY_BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app
ENV PYTHONPATH=/app/backend:/app

COPY web/backend/requirements.txt /tmp/backend-requirements.txt
COPY modelos/etl/requirements.txt /tmp/etl-requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
	pip install --no-cache-dir -r /tmp/backend-requirements.txt -r /tmp/etl-requirements.txt

COPY web/backend /app/backend
COPY modelos /app/modelos
COPY modelos/etl /app/etl
COPY data /app/data

ENV PYTHONPATH=/app/backend
ENV DATA_DIR=/app/data
ENV OUTPUT_DIR=/app/data/out

ENTRYPOINT ["python", "etl/build_laliga_enriched.py"]
