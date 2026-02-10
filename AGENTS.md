# Repository Guidelines

## Project Structure & Module Organization
- `src/facet/` is the main Python package (core modules like `inspection/`, `simulation/`, `selection/`, `validation/`).
- `test/test/` contains pytest suites and `conftest.py` helpers.
- `sphinx/source/` holds documentation sources; `notebooks/` contains tutorial notebooks.
- `config/`, `condabuild/`, and `dist/` support build and distribution workflows.

## Build, Test, and Development Commands
- `./dev-setup.sh` creates the conda dev environment and installs pre-commit hooks.
- `tox -e py311` (or another env in `tox.ini`) runs the full test environment with configured deps.
- `pytest test/ -s` runs tests directly (same path used by tox).
- `pre-commit run --all-files` runs formatting and lint hooks locally.

## Documentation
- `cd sphinx && python make.py html` builds the Sphinx docs locally.
- Generated HTML lives in `sphinx/build/html`.

## Coding Style & Naming Conventions
- Python uses 4-space indentation; keep modules and symbols in `snake_case` and classes in `PascalCase`.
- Formatting: Black (`line-length = 88`) and isort (`profile = black`).
- Linting: flake8 with config in `tox.ini`.
- Typing: mypy is enabled in strict mode; `src/facet/py.typed` indicates inline type hints.

## Testing Guidelines
- Framework: pytest with pytest-cov in the `testing` extra (see `pyproject.toml`).
- Tests live under `test/test/` and should mirror the package layout (e.g., `test/test/facet/...`).
- Prefer running via tox to match CI configuration and coverage settings.

## Commit & Pull Request Guidelines
- Commit messages follow a `TAG: summary` convention in history (e.g., `FIX: ...`, `BUILD: ...`, `DOC: ...`, `RF: ...`).
- PRs should include a clear description, test evidence (tox or pytest output), and doc updates when APIs or behavior change. Add screenshots only if user-facing docs or notebooks are updated.

## Configuration & Tooling Notes
- Project metadata and tool config live in `pyproject.toml` and `tox.ini`.
- Pre-commit hooks include black, isort, flake8, and mypy; install them early to catch issues before CI.
