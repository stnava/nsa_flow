.PHONY: help install test theory experiments tables paper clean

help:
	@echo "install      - editable install with experiment + test extras"
	@echo "test         - run the full test suite (theory, solver, layers)"
	@echo "theory       - run only the theory property battery"
	@echo "experiments  - run all paper experiments -> paper/results, paper/figs"
	@echo "tables       - rebuild paper/results/*.tex from the saved CSVs (cheap)"
	@echo "new-benchmarks - run Tecator, Sonar & Prostate benchmarks + sweeps + tables"
	@echo "paper        - build paper/nsaflow.pdf from existing results"
	@echo "all          - experiments + paper, from scratch"
	@echo "clean        - remove build artefacts and caches"
	@echo ""
	@echo "Selective reruns:  python experiments/run_all.py --only e2 e6"
	@echo "Quick pass:        python experiments/run_all.py --quick"

install:
	pip install -e ".[experiments,test]"

test:
	PYTHONPATH=. pytest tests/

theory:
	PYTHONPATH=. pytest tests/test_theory.py -v

experiments:
	PYTHONPATH=. python experiments/run_all.py

tables:
	PYTHONPATH=. python experiments/build_tables.py

new-benchmarks:
	PYTHONPATH=. python experiments/benchmark_new_public_data.py
	PYTHONPATH=. python experiments/sweep_new_public_data.py
	PYTHONPATH=. python experiments/plot_new_public_benchmarks.py
	PYTHONPATH=. python experiments/build_new_public_tables.py

paper:
	cd paper && latexmk -pdf -quiet nsaflow.tex

all: experiments paper

clean:
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/
	cd paper && latexmk -C >/dev/null 2>&1 || true
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
