.PHONY: sync test lint typecheck

sync:
	uv sync --group dev --group isaac
	uv run --no-sync python scripts/fix_isaaclab_stubs.py

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uvx ty check .
