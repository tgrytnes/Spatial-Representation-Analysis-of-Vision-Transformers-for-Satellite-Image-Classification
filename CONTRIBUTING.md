# Contributing Guidelines

This project follows a strict development workflow to ensure stability and code quality. All contributors (human and AI) must adhere to these rules.

## Core Rules

1.  **NO Direct Commits to Main**: The `main` branch is protected. Direct changes are forbidden.
2.  **Feature Branches**: All new work (features, bugfixes, refactoring) must happen on a dedicated branch.
    *   Naming convention: `feature/short-description` or `fix/issue-description`.
3.  **Pull Requests (PR)**: Merging to `main` requires a Pull Request.
4.  **CI Validation**: All PRs must pass the Continuous Integration (CI) pipeline (linting, formatting, tests) before merging.

## Development Workflow

1.  **Create a Branch**:
    ```bash
    git checkout -b feature/my-new-feature
    ```

2.  **Develop & Test Locally**:
    *   Write code and tests.
    *   Run linting: `poetry run ruff check .` and `poetry run black --check .`
    *   Run tests: `poetry run pytest`

3.  **Commit Changes**:
    ```bash
    git add .
    git commit -m "feat: Implement my new feature"
    ```

4.  **Push & Create PR**:
    ```bash
    git push origin feature/my-new-feature
    # Then create a PR via GitHub UI or CLI
    ```

5.  **Merge**:
    *   Only after CI passes (`check` mark).
    *   Squash and merge is preferred to keep history clean.

## Code Style

*   **Formatter**: Black
*   **Linter**: Ruff
*   **Imports**: Sorted via isort (enforced by Ruff). Known third-party modules are explicitly defined in `pyproject.toml`.

## Environment

*   **Dependency Management**: Poetry
*   **Python Version**: 3.11+
