# Makefile

.PHONY: help install lint format test setup-hooks

help:
	@echo "Commands:"
	@echo "  install       : Install python dependencies and pre-commit hooks"
	@echo "  lint          : Lint and check formatting"
	@echo "  format        : Auto-format python code"
	@echo "  test          : Run all tests"
	@echo "  setup-hooks   : Install pre-commit hooks"

install:
	poetry install
	poetry run pre-commit install

lint:
	poetry run ruff .
	poetry run black --check .

format:
	poetry run ruff --fix .
	poetry run black .

test:
	poetry run pytest --cov=eurosat_vit_analysis --cov-report=term-missing

setup-hooks:
	poetry run pre-commit install
