.PHONY: up down etl train predict logs

up:
	$(MAKE) -C infra up

down:
	$(MAKE) -C infra down

etl:
	$(MAKE) -C infra etl

train:
	$(MAKE) -C infra train

predict:
	$(MAKE) -C infra predict

logs:
	$(MAKE) -C infra logs
