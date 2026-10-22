.PHONY: setup test lint format universe coverage app

setup:
	uv sync

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

universe:
	uv run ripplekg universe-check

coverage:
	uv run ripplekg coverage

app:
	uv run streamlit run app/Home.py
