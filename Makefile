.PHONY: install test lint run compose-up
install:
	python -m pip install -e ".[dev]"
test:
	pytest
lint:
	ruff check .
run:
	uvicorn ragforge.api:app --reload
compose-up:
	docker compose up --build

