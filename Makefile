.PHONY: setup check lint types

setup:

		uv sync --all-groups
		uv run pre-commit install

check: lint types

lint:

		uv run ruff check .
		uv run ruff format --check .

types:

		uv run mypy src/