.PHONY: install dev lint format test check precommit

install:
	uv sync

dev:
	uv sync --extra dev

lint:
	uv run --extra dev ruff check .

format:
	uv run --extra dev ruff format .

test:
	uv run --extra dev pytest

check: lint test

precommit:
	uv run --extra dev pre-commit run --all-files
