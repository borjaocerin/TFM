.PHONY: up down etl train predict logs

COMPOSE_FILE=infra/docker-compose.yml

up:
	docker compose -f $(COMPOSE_FILE) up --build

down:
	docker compose -f $(COMPOSE_FILE) down

logs:
	docker compose -f $(COMPOSE_FILE) logs -f

etl:
	docker compose -f $(COMPOSE_FILE) run --rm etl \
		--hist data/historical/laliga_merged_matches.csv \
		--fdata_dir data/football-data \
		--elo data/elo/ELO_RATINGS.csv \
		--fixtures data/fixtures/fixtures.csv \
		--windows 5,10

train:
	curl -X POST "http://localhost:8000/api/v1/model/train" \
	  -H "Content-Type: application/json" \
	  -d "{\"dataset_path\":\"data/out/laliga_enriched_model.csv\",\"use_xgb\":false,\"calibration\":\"platt\"}"

predict:
	curl -X POST "http://localhost:8000/api/v1/predict" \
	  -H "Content-Type: application/json" \
	  -d "{\"fixtures_enriched_path\":\"data/out/fixtures_enriched.csv\"}"
