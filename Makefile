.PHONY: setup setup-dev lint format typecheck test test-cov train clean

PYTHON := python3.11

# --- Environment setup ---------------------------------------------------------
setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

setup-dev: setup
	$(PYTHON) -m pip install -r requirements-dev.txt
	pre-commit install

# --- Code quality ----------------------------------------------------------------
format:
	black src tests scripts
	isort src tests scripts

lint:
	ruff check src tests scripts

typecheck:
	mypy src

# --- Testing -----------------------------------------------------------------------
test:
	pytest -m "unit or integration" -q

test-cov:
	pytest --cov=ecg_digitization --cov-report=html --cov-report=term-missing

# --- Training (scaffold only until a model is implemented) -------------------------
train:
	$(PYTHON) scripts/train.py

train-baseline:
	$(PYTHON) scripts/train.py experiment=exp001_baseline

# --- Housekeeping ---------------------------------------------------------------------
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	rm -rf outputs multirun
