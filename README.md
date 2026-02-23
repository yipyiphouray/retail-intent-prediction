# Online Shopping ML Prediction by SKU YOU LATER

End-to-end ML pipeline to predict purchase intent from online retail sessions.

## Team Setup

This project uses a local virtual environment (`.venv`) managed by `uv`.
Do not rely on globally installed Python tools (for example `ruff`, `pytest`).

### Prerequisites

- Python 3.11
- `uv`
- Optional: Conda (for bootstrap via `environment.yml`)

### Bootstrap (Option A: with Conda)

```bash
conda env create -f environment.yml
conda activate online_shopping_ml_prediction_by_sku_you_later
```

### Create and sync project environment

```bash
make create_environment
source .venv/bin/activate
```

If `.venv` already exists, use:

```bash
make sync
```

## Common Commands

```bash
make help      # list available targets
make sync      # sync dependencies from pyproject + uv.lock
make lint      # ruff format check + lint
make format    # auto-fix lint and format
make test      # run pytest
make lock      # refresh uv.lock
make data      # run dataset pipeline entrypoint
make clean     # remove Python cache artifacts
```

## Bagging Model Training

Train and fine-tune bagging classifiers (base estimators: logistic regression, decision tree, and KNN)
using processed features/labels:

```bash
uv run python -m online_retail_prediction.modeling.bagging_train
```

Optional arguments:

```bash
uv run python -m online_retail_prediction.modeling.bagging_train \
    --features-path data/processed/features.csv \
    --labels-path data/processed/labels.csv \
    --output-dir models \
    --test-size 0.2 \
    --cv 5 \
    --random-state 42
```

Fine-tuning is optimized on ROC-AUC only. Evaluation reports:

- ROC-AUC
- Precision
- Recall
- F1
- Macro-F1
- Cohen's Kappa
- Accuracy

Outputs are written to `models/`:

- `bagging_logistic_regression.pkl` and `bagging_logistic_regression_metrics.txt`
- `bagging_decision_tree.pkl` and `bagging_decision_tree_metrics.txt`
- `bagging_knn.pkl` and `bagging_knn_metrics.txt`
- `bagging_model_comparison.csv`

## Cluster Labeling Workflow

Session clustering and manual cluster-label propagation are documented in:

- `docs/cluster_labeling.md`

Key outputs:

- Session-level clustering exports: `data/cluster_outputs/`
- Clustering metrics and artifacts: `models/`

## Daily Development Workflow

1. Start from integration branch:
```bash
git checkout dev
git pull origin dev
```
2. Create a feature branch:
```bash
git checkout -b feature/<short-topic>
```
3. Sync and validate locally:
```bash
make sync
make lint
make test
```
4. Commit using Conventional Commits:
```bash
git commit -m "feat(<scope>): <description>"
```
5. Push branch and open a PR into `dev`.

## Dependency Management

- Runtime dependencies live in `pyproject.toml` under `[project.dependencies]`.
- Development/tooling dependencies live under `[project.optional-dependencies].dev`.
- Lockfile is `uv.lock` and must be updated with dependency changes.

Add/update dependencies using `uv`:

```bash
uv add <package>                    # runtime dependency
uv add --optional dev <package>     # dev dependency
make lock                           # update lockfile
```

Commit `pyproject.toml` and `uv.lock` together when dependencies change.

## Branching and Collaboration

This repo follows GitFlow:

- Long-lived branches: `main`/`master`, `dev`
- Feature work: `feature/<short-topic>` from `dev`
- Release work: `release/<version>` from `dev`
- Release merges go to `main`/`master` and are back-merged to `dev`

Rules:

- Do not push directly to `main`/`master`.
- Do not commit directly to `dev`.
- Do not modify CI/CD pipelines without explicit permission.

## Testing Expectations

- Keep tests in `tests/`.
- Use descriptive test names and file names (`test_<feature>_<behavior>.py`).
- Include a short module-level description in each test file.
- Run `make test` before opening/updating a PR.

## Project Organization

```text
retail-intent-prediction/
|-- LICENSE
|-- Makefile
|-- README.md
|-- environment.yml
|-- pyproject.toml
|-- uv.lock
|-- data/
|   |-- external/
|   |-- interim/
|   |-- processed/
|   `-- raw/
|-- docs/
|   |-- mkdocs.yml
|   `-- docs/
|-- models/
|-- notebooks/
|-- references/
|-- reports/
|   `-- figures/
|-- tests/
|   `-- test_data.py
`-- online_retail_prediction/
    |-- __init__.py
    |-- config.py
    |-- dataset.py
    |-- features.py
    |-- plots.py
    `-- modeling/
        |-- __init__.py
        |-- feature_importance.py
        |-- predict.py
        `-- train.py
```
