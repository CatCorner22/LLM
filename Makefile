.PHONY: install install-dev lint format typecheck test test-unit test-integration coverage serve train clean

PYTHON ?= python3
PIP ?= pip

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev]"

lint:
	ruff check src tests
	ruff format --check src tests

format:
	ruff check --fix src tests
	ruff format src tests

typecheck:
	mypy src/pioneer

test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v -m unit

test-integration:
	pytest tests/integration/ -v -m integration

coverage:
	pytest tests/ --cov=pioneer --cov-report=html --cov-fail-under=50

serve:
	pioneer-serve

train:
	pioneer-train --experiment baseline --epochs 3

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

ci: lint typecheck test
