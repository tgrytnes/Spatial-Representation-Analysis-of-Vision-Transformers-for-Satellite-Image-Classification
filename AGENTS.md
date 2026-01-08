# Agent Guidelines

## Workflow
- Keep changes small and focused; confirm scope before coding when unclear.
- Do a brief root cause analysis before fixing errors.
- Use the project environment (`poetry run ...`) for Python commands.

## Stories & Issues
- GitHub issues are the source of truth for significant changes.
- If no issue exists for a non-trivial change, create or request one first.
- Ensure acceptance criteria are clear before implementation.

## Tests & Quality
- Develop in TDD style when changing behavior: write failing tests first.
- Prefer tests for behavior changes; update/add unit and E2E tests as needed.
- Run local checks before committing or pushing.
- Preferred: `make lint` and `make test`.
- If `make` is unavailable: `poetry run ruff check .`, `poetry run black --check .`, and `poetry run pytest --cov=eurosat_vit_analysis --cov-fail-under=80`.

## Branching
- Use a feature branch and PR for non-trivial changes when possible.
