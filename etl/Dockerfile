FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY backend/requirements.txt /tmp/backend-requirements.txt
COPY etl/requirements.txt /tmp/etl-requirements.txt
RUN pip install --no-cache-dir -r /tmp/backend-requirements.txt -r /tmp/etl-requirements.txt

COPY backend /app/backend
COPY etl /app/etl
COPY data /app/data

ENV PYTHONPATH=/app/backend
ENV DATA_DIR=/app/data
ENV OUTPUT_DIR=/app/data/out

ENTRYPOINT ["python", "etl/build_laliga_enriched.py"]
