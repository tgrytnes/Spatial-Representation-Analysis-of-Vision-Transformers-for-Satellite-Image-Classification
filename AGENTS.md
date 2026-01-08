# Agent Guidelines

## Workflow
- Keep changes small and focused; confirm scope before coding when unclear.
- Do a brief root cause analysis before fixing errors.
- Use the project environment (`poetry run ...`) for Python commands.

## Stories & Issues
- GitHub Issues are the single source of truth for all stories and issues.
- Do not implement a story or issue unless it exists in GitHub Issues.
- If no issue exists for a non-trivial change, create or request one first.
- Ensure acceptance criteria are clear before implementation.
- When asked to continue or implement a story, use `gh` to search GitHub Issues for the story ID/title before starting work.

## Tests & Quality
- Develop in TDD style when changing behavior: write failing tests first.
- Prefer tests for behavior changes; update/add unit and E2E tests as needed.
- Run local checks before committing or pushing.
- Preferred: `make lint` and `make test`.
- If `make` is unavailable: `poetry run ruff check .`, `poetry run black --check .`, and `poetry run pytest --cov=eurosat_vit_analysis --cov-fail-under=80`.

## Branching
- Create and switch to a feature branch before starting any new feature work.
- Use a PR for non-trivial changes.
