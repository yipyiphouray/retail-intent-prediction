# Online Shopping ML Prediction by SKU YOU LATER

End-to-end ML pipeline to predict purchase intent from online retail sessions.

## Story So Far

We started with a raw, click-level dataset and a simple question: can the first
few clicks of a session tell us who is likely to buy? The team aligned on a
leakage-safe approach, building **session-level features from the first N clicks**
while reserving **full-session context for labels**. That split let us move fast
without leaking future behavior into the model.

From there, we built a labeling system that supports proxy labels today and
manual/cluster labels tomorrow, keeping the schema stable as the strategy
evolves. We benchmarked baselines (Logistic Regression and Random Forest) to
establish an initial performance floor, then expanded to bagging ensembles for
stronger lift and comparability across model families.

To make this useful beyond the notebook, we built a clustering workflow to
summarize sessions, support manual labeling at scale, and generate exports for
sequence-model handoff. In parallel, the team documented the business context
(Poland-first, CEE-aware), defined stakeholder ownership, and framed KPI
guardrails to keep the model tied to real operating decisions.

Everything below captures the current state of that journey: how to run the
pipeline, how the features and labels are defined, where the models live, and
how the business case is structured.

## Story So Far

We started with a raw, click-level dataset and a simple question: can the first
few clicks of a session tell us who is likely to buy? The team aligned on a
leakage-safe approach, building **session-level features from the first N clicks**
while reserving **full-session context for labels**. That split let us move fast
without leaking future behavior into the model.

From there, we built a labeling system that supports proxy labels today and
manual/cluster labels tomorrow, keeping the schema stable as the strategy
evolves. We benchmarked baselines (Logistic Regression and Random Forest) to
establish an initial performance floor, then expanded to bagging ensembles for
stronger lift and comparability across model families.

To make this useful beyond the notebook, we built a clustering workflow to
summarize sessions, support manual labeling at scale, and generate exports for
sequence-model handoff. In parallel, the team documented the business context
(Poland-first, CEE-aware), defined stakeholder ownership, and framed KPI
guardrails to keep the model tied to real operating decisions.

Everything below captures the current state of that journey: how to run the
pipeline, how the features and labels are defined, where the models live, and
how the business case is structured.

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

### Bagging Metrics (from `models/`)

Bagging + Logistic Regression (`models/bagging_logistic_regression_metrics.txt`, test set):

- Accuracy: 0.7524
- Precision: 0.3978
- Recall: 0.9488
- F1: 0.5606
- ROC-AUC: 0.8909
- Best CV ROC-AUC: 0.8971

Bagging + Decision Tree (`models/bagging_decision_tree_metrics.txt`, test set):

- Accuracy: 0.7665
- Precision: 0.4113
- Recall: 0.9338
- F1: 0.5711
- ROC-AUC: 0.8919
- Best CV ROC-AUC: 0.8966

## Cluster Labeling Workflow

Session clustering and manual cluster-label propagation are documented in:

- `docs/CLUSTER_LABELING.md`
- `docs/CLUSTER_LABELING.md`

Key outputs:

- Session-level clustering exports: `data/cluster_outputs/`
- Clustering metrics and artifacts: `models/`

### Clustering Metrics (from `models/`)

From `models/clustering_metrics.json`:

- k: 10
- Sessions: 24,026
- Features used: 15
- Inertia: 171,549.9455
- Silhouette: 0.1819
- Calinski-Harabasz: 2,937.3955
- Davies-Bouldin: 1.7086
- Largest cluster share: 17.16%

## Feature Engineering

Session-level feature construction (first-N clicks only) is documented in:

- `docs/FEATURE_ENGINEERING.md`

Key points:

- Builds one row per `session_id` from the first `N` clicks to prevent leakage.
- Supports column normalization from raw UCI schema.
- Writes features to `data/processed/features.csv` via `online_retail_prediction/features.py`.

## Labeling Strategies

Session-level intent labeling (full-session context) is documented in:

- `docs/LABELING.md`

Key points:

- Strategy-based labeling (`ProxyHybrid`, `ExternalPartial`, `Override`).
- Standard output schema: `session_id`, `label`, `label_source`, `label_confidence`.
- Writes labels to `data/processed/labels.csv` via `online_retail_prediction/features.py`.

## Baseline Model Comparison

Baseline model evaluation and metrics are documented in:

- `docs/model_comparison.md`

Summary:

- Logistic Regression and Random Forest baselines on 27 engineered features.
- Random Forest is the best baseline by accuracy/F1 while LR has higher recall.

### Baseline Metrics (from `models/`)

Logistic Regression (`models/baseline_model_metrics.txt`):

- Accuracy: 0.7528
- Precision: 0.3985
- Recall: 0.9525
- F1: 0.5619
- ROC-AUC: 0.8913

Random Forest (`models/baseline_rf_model_metrics.txt`):

- Accuracy: 0.7834
- Precision: 0.4286
- Recall: 0.9038
- F1: 0.5814
- ROC-AUC: 0.8920

## Business Context and Value Proposition (Issue #10)

Business context, market sizing, stakeholder map, and KPI framework:

- `reports/issue-10-business-context.md`
- `reports/issue-10-business-context-presentation.md`

Highlights:

- Poland-first operating context with CEE comparator markets.
- Clear constraints from the source dataset and governance guardrails.
- 90-day execution plan with KPI placeholders.

## Docs Site (MkDocs)

Project docs are maintained under `docs/` (MkDocs structure):

- `docs/README.md` (build/serve instructions)
- `docs/docs/index.md` (landing page)
- `docs/docs/getting-started.md` (setup placeholder)

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
