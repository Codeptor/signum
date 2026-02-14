.PHONY: setup infra down ingest train optimize backtest dashboard test lint format bench

setup:
	pip install -e ".[all]"

infra:
	docker-compose -f infra/docker-compose.yml up -d

down:
	docker-compose -f infra/docker-compose.yml down

ingest:
	python -m python.data.ingestion

train:
	python -m python.alpha.train

optimize:
	python -m python.portfolio.optimizer

backtest:
	python -m python.backtest.run

dashboard:
	python -m python.monitoring.dashboard

test:
	pytest tests/ -v

lint:
	ruff check python/ tests/

format:
	ruff format python/ tests/

bench:
	cd rust && cargo bench
