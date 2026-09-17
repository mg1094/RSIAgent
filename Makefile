# Convenience targets.  Everything here works without an API key.

PY ?= python3

.PHONY: help install test lint format demo example clean

help:
	@echo "make install   install the package in editable mode with dev tools"
	@echo "make test      run the test suite (no API key, no network)"
	@echo "make lint      ruff check + format check (what CI runs)"
	@echo "make format    apply ruff formatting"
	@echo "make demo      run the full offline demo, all four stage conditions"
	@echo "make example   run the custom-task example end to end"
	@echo "make clean     remove runs, caches, and build artefacts"

install:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format:
	$(PY) -m ruff format .
	$(PY) -m ruff check --fix .

demo:
	$(PY) examples/offline_demo/run.py --arm all

example:
	cd examples/custom_task && \
	  $(PY) -m rsiagent.cli run --task task.py:task --evaluator task.py:evaluator \
	    --policies policies.py:policies --arm baseline --out runs/base && \
	  $(PY) -m rsiagent.cli run --task task.py:task --evaluator task.py:evaluator \
	    --policies policies.py:policies --arm rsi --out runs/rsi && \
	  $(PY) -m rsiagent.cli inspect runs/rsi

clean:
	rm -rf runs examples/*/runs build dist *.egg-info
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
