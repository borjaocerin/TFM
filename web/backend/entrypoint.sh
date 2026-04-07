#!/bin/sh
set -e

echo "Starting TFM API..."

# Create output directory
mkdir -p /app/data/out

# Check if fixtures_enriched.csv exists
if [ ! -f /app/data/out/fixtures_enriched.csv ]; then
    echo "WARNING: fixtures_enriched.csv not found. Attempting to generate it..."
    
    # Generate historical enriched data if missing
    if [ ! -f /app/data/out/laliga_enriched_all.csv ]; then
        echo "Generating historical enriched data..."
        python -c "
import sys
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/backend')

from app.services.datasets import ingest_datasets
from app.schemas.datasets import DatasetIngestRequest

try:
    result = ingest_datasets(DatasetIngestRequest(
        historical='data/historical/laliga_merged_matches.csv',
        football_data_dir='data/football-data'
    ))
    print(f'Historical enriched rows: {result[\"rows_total\"]}')
except Exception as e:
    print(f'WARNING: dataset generation failed: {e}')
    " 2>&1 || true
    fi
    
    # Generate enriched fixtures if possible
    if [ -f /app/data/out/laliga_enriched_all.csv ]; then
        echo "Generating fixtures_enriched.csv..."
        python -c "
import sys
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/backend')

from app.services.datasets import build_fixtures_features
from app.schemas.datasets import FixturesFeatureRequest

try:
    result = build_fixtures_features(FixturesFeatureRequest(
        fixtures_csv='data/fixtures/fixtures.csv',
        windows=[5, 10]
    ))
    print(f'Enriched fixtures rows: {result[\"rows_total\"]}')
except Exception as e:
    print(f'WARNING: fixtures generation failed: {e}')
    " 2>&1 || echo "WARNING: Could not generate enriched fixtures"
    else
        echo "WARNING: historical data not available, skipping fixtures enrichment"
    fi
else
    echo "fixtures_enriched.csv already exists"
fi

# Generate predictions using the active best model.
# If no model exists, predict_matches triggers default training automatically.
if [ -f /app/data/out/fixtures_enriched.csv ]; then
    echo "Generating predictions.csv with the active model..."
    python -c "
import sys
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/backend')

from app.schemas.predict import PredictRequest
from app.services.predict import predict_matches

try:
    result = predict_matches(PredictRequest(fixtures_enriched_path='data/out/fixtures_enriched.csv'))
    print(f'Predictions generated: {result[\"rows\"]} rows -> {result[\"output_csv\"]}')
except Exception as e:
    print(f'WARNING: predictions generation failed: {e}')
    " 2>&1 || true
else
    echo "WARNING: fixtures_enriched.csv missing, skipping predictions generation"
fi

echo "Starting server on 0.0.0.0:8000..."
exec uvicorn main:app --host 0.0.0.0 --port 8000
