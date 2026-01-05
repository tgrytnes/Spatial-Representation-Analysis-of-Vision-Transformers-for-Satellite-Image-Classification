# Agent Guidelines

- Run local checks before committing or pushing.
- Preferred: `make lint` and `make test`.
- If `make` is unavailable: `poetry run ruff check .`, `poetry run black --check .`, and `poetry run pytest --cov=eurosat_vit_analysis --cov-fail-under=80`.
