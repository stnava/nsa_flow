.PHONY: help install test theory experiments paper clean

help:
	@echo "install      - editable install with experiment + test extras"
	@echo "test         - run the full test suite (theory, solver, layers)"
	@echo "theory       - run only the theory property battery"
	@echo "experiments  - run all paper experiments, writing paper/results/"
	@echo "paper        - build paper/nsaflow.pdf (runs experiments if needed)"
	@echo "clean        - remove build artefacts and caches"

install:
	pip install -e ".[experiments,test]"

test:
	PYTHONPATH=. pytest tests/

theory:
	PYTHONPATH=. pytest tests/test_theory.py -v

experiments:
	PYTHONPATH=. python experiments/run_all.py

paper:
	$(MAKE) experiments
	cd paper && latexmk -pdf -quiet nsaflow.tex

clean:
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ paper/*.aux paper/*.log paper/*.out paper/*.fls paper/*.fdb_latexmk
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
