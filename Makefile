# Development and experiment entry points.
# Every committed number in results/ comes from one of these targets.

PY := ./.venv/Scripts/python.exe
ifeq ($(OS),)
PY := ./.venv/bin/python
endif

# The exact overrides behind the committed tables. The base config is already
# sized for this machine; only the epoch count differs between the headline runs
# and the ablations, and that difference is stated in docs/RESULTS.md.
MAIN := optim.epochs=16
ABLATE := optim.epochs=12

.PHONY: help setup test test-all lint fmt smoke bench compare ablate splits \
        dataeff all figures notebooks clean clean-results

help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

setup:  ## Create the environment with uv and install the package
	uv python install 3.12
	uv venv --python 3.12 .venv
	uv pip install --python $(PY) --index-url https://download.pytorch.org/whl/cpu \
	  --extra-index-url https://pypi.org/simple torch
	uv pip install --python $(PY) numpy scipy pandas scikit-learn pyyaml matplotlib \
	  pytest ruff nbformat nbconvert ipykernel
	uv pip install --python $(PY) -e . --no-deps

test:  ## Fast unit tests (~1 minute)
	$(PY) -m pytest tests -q -m "not slow"

test-all:  ## Everything, including end-to-end training tests
	$(PY) -m pytest tests -q

lint:  ## Static checks
	$(PY) -m ruff check src scripts tests

fmt:  ## Auto-fix what ruff can
	$(PY) -m ruff check --fix src scripts tests

smoke:  ## Full pipeline end to end on a tiny config (~1 min)
	$(PY) scripts/train.py --config configs/smoke.yaml

bench:  ## Params, MACs, latency, and the DTW-vs-model cost curve (no training)
	$(PY) scripts/benchmark_efficiency.py

compare:  ## The headline comparison with paired statistics and attribution fidelity
	$(PY) scripts/compare_methods.py --set $(MAIN)

ablate:  ## Seed study then component ablations
	$(PY) scripts/run_ablations.py --only seeds --set $(MAIN)
	$(PY) scripts/run_ablations.py --only ablations --set $(ABLATE)

splits:  ## Random vs cross-subject vs cross-degradation (the leakage estimate)
	$(PY) scripts/run_ablations.py --only splits --set $(MAIN)

dataeff:  ## Spearman against the labelled-sequence budget
	$(PY) scripts/data_efficiency.py --set $(MAIN)

all:  ## The whole experiment matrix, in the order the results are read in
	$(PY) scripts/run_all.py

figures:  ## Redraw figures from artefacts already on disk
	$(PY) scripts/make_figures.py

notebooks:  ## Regenerate and execute every notebook
	$(PY) scripts/build_notebooks.py --execute

clean:  ## Remove caches and build artefacts
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

clean-results:  ## Remove run artefacts (keeps committed tables and figures)
	rm -rf results/runs
