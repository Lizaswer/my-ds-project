.PHONY: setup check lint types s3-up s3-down ingest data

# --- Из ДЗ №1 ---
setup:
	uv sync --all-groups
	uv run pre-commit install

check: lint types

lint:
	uv run ruff check .
	uv run ruff format --check .

types:
	uv run mypy src/

# --- Из ДЗ №2 (Блок 4) ---

# 1. Поднять хранилище S3
s3-up:
	@if [ ! -f .env ]; then cp .env.example .env; echo "Создан .env из .env.example"; fi
	docker compose up -d

# 2. Остановить хранилище S3
s3-down:
	docker compose down

# 3. Загрузить сырые данные в S3
ingest:
	uv run python -m my_ds_project.ingest

# 4. Запустить весь конвейер обработки

# Переменные по умолчанию
VERSION ?= v1
SEED ?= 42
VAL_SIZE ?= 5000

data:
	uv run python -m my_ds_project.processing.convert
	uv run python -m my_ds_project.processing.split --seed $(SEED) --val-size $(VAL_SIZE)
	uv run python -m my_ds_project.processing.stats
	uv run python -m my_ds_project.processing.upload --version $(VERSION) --seed $(SEED) --val-size $(VAL_SIZE)