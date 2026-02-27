#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = online_shopping_ml_prediction_by_sku_you_later
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = python
UV = uv

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(UV) sync --extra dev

## Sync project dependencies into the local virtual environment
.PHONY: sync
sync:
	$(UV) sync --extra dev


## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	$(UV) run ruff format --check
	$(UV) run ruff check

## Format source code with ruff
.PHONY: format
format:
	$(UV) run ruff check --fix
	$(UV) run ruff format



## Run tests
.PHONY: test
test:
	$(UV) run pytest tests

## Run Streamlit cluster labeling UI locally
.PHONY: cluster_ui
cluster_ui:
	$(UV) run streamlit run apps/cluster_labeling_app.py

## Run demo inference API locally
.PHONY: api
api:
	$(UV) run uvicorn online_retail_prediction.api.app:app --reload

## Run React demo storefront locally
.PHONY: demo_ui
demo_ui:
	cd apps/demo_storefront && npm install && npm run dev

## Update uv lockfile
.PHONY: lock
lock:
	$(UV) lock


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	$(UV) venv --python $(PYTHON_VERSION)
	$(UV) sync --extra dev
	@echo ">>> local virtual environment created. Activate with:\nsource .venv/bin/activate"



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make dataset
.PHONY: data
data: requirements
	$(UV) run $(PYTHON_INTERPRETER) online_retail_prediction/dataset.py


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
