# AGENTS.md

This file defines working rules for agents in this repository.

## Project Baseline

- `pyproject.toml` and `environment.yml` are both project source-of-truth files and must stay aligned.
- Use these files to validate and manage:
  - Python compatibility and packaging/build metadata (for example `requires-python`, package `name`/`version`, `build-system`)
  - lint/format behavior (Ruff config, import sorting, source/include scopes)
  - environment composition and dependency declarations (Conda channels/deps and pip deps, including editable install entries)
- Current baseline values:
  - Python version: `pyproject.toml` uses `~=3.11.0`, `environment.yml` uses `python=3.11`
  - Ruff: line length `99`; run `ruff check` and `ruff format`
- If dependency or tooling configuration changes, update related files and Make targets when needed to keep workflows consistent.

## Workflow

1. Read `pyproject.toml`, `environment.yml`, and `Makefile` before making meaningful changes.
2. Prefer existing Make targets when available.
3. Keep changes scoped to the user request.

## Git Rules

- Follow GitFlow for version control and collaboration:
  - long-lived branches: `main` (or `master`) and `dev`
  - day-to-day work starts from `dev` using feature branches: `feature/<short-topic>`
  - release preparation uses release branches from `dev`: `release/<version>`
  - merge releases into `main`/`master` and back-merge into `dev`
- Never push directly to `main`/`master`.
- Do not commit directly to `dev`; use feature or release branches and PRs.
- Use Conventional Commits for commit messages:
  - format: `<type>(<scope>): <description>`
  - examples: `feat(model): add session feature encoder`, `fix(api): handle empty cart input`
- Do not change CI/CD pipelines without explicit user permission.

## Testing Rules

- Verify changes by running tests before finishing:
  - preferred: `make test`
  - fallback: `python -m pytest tests`
- Add or update tests for behavior changes.
- Place tests under `tests/` with descriptive names:
  - files: `test_<feature>_<behavior>.py`
  - test functions: `test_<unit>_<expected_behavior>`
- Include a short module-level description in each test file explaining what the tests cover.

## Makefile Guidance

Current common commands:

- `make requirements`
- `make lint`
- `make format`
- `make test`
- `make clean`
- `make create_environment`
- `make data`

When needed, add new Make targets sparingly:

- only for repeated workflows
- keep target names explicit and short
- avoid duplicating one-off shell commands

## Dependency Management (uv)

Use `uv` for dependency handling and lockfile operations.

- Add dependency: `uv add <package>`
- Remove dependency: `uv remove <package>`
- Update lockfile: `uv lock`
- Sync environment: `uv sync`

If dependency changes are made, ensure lock data is updated in the same change.

## Completion Checklist

Before handing off work, confirm:

- lint/format checks are satisfied
- tests pass
- new behavior has tests with clear names and descriptions
- no CI/CD pipeline files were changed without permission
- branch strategy follows GitFlow (`main`/`master`, `dev`, `feature/*`, `release/*`)
- commits (when requested) follow Conventional Commits
