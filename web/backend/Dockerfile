ARG PY_BASE_IMAGE=rest-service:latest
FROM ${PY_BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app/backend
ENV PYTHONPATH=/app/backend:/app

COPY web/backend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
	pip install --no-cache-dir -r /tmp/requirements.txt

COPY web/backend /app/backend
COPY modelos /app/modelos
COPY data /app/data

ENV DATA_DIR=/app/data
ENV OUTPUT_DIR=/app/data/out
ENV MODEL_DIR=/app/backend/app/models/store

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
