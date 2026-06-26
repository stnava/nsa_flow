.PHONY: help install test benchmark clean

help:
	@echo "Available make targets:"
	@echo "  install   - Install package in editable mode with dependencies"
	@echo "  test      - Run the pytest suite"
	@echo "  benchmark - Run the retraction performance benchmark"
	@echo "  clean     - Remove build artifacts, cache, and temp files"

install:
	pip install -e .

test:
	PYTHONPATH=. pytest

benchmark:
	PYTHONPATH=. python tests/benchmark_retractions.py

clean:
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ retraction_benchmark_results.csv
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete
