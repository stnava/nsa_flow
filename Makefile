.PHONY: help install test theory bench bench-full bench-smoke bench-check bench-report speed experiments tables paper rmd-paper dist release clean docs docs-clean

help:
	@echo "install      - editable install with experiment + test extras"
	@echo "test         - run the full test suite (theory, solver, layers)"
	@echo "theory       - run only the theory property battery"
	@echo "experiments  - run all paper experiments -> paper/results, paper/figs"
	@echo "bench        - standard-fast suite (~10 min) -> bench_results/ (nsa-flow-bench run)"
	@echo "bench-full   - standard-full suite (~40 min)"
	@echo "bench-smoke  - smoke suite (seconds, synthetic only)"
	@echo "bench-check  - smoke suite vs stored baseline (solver regression guard)"
	@echo "bench-report - markdown report of the latest standard run"
	@echo "speed        - NSA-Flow vs PCA timing -> paper/results"
	@echo "tables       - rebuild paper/results/*.tex from the saved CSVs (cheap)"
	@echo "new-benchmarks - run Tecator, Sonar & Prostate benchmarks + sweeps + tables"
	@echo "docs         - build Sphinx HTML documentation -> docs/_build/html"
	@echo "docs-clean   - remove Sphinx build artefacts"
	@echo "paper        - build paper/nsaflow.pdf from existing results"
	@echo "rmd-paper    - build paper/nsa_flow.pdf from Rmd"
	@echo "dist         - build and check distribution wheels & sdist"
	@echo "release      - upload dist/* to PyPI"
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

bench:
	python -m nsa_flow.bench run --suite standard-fast --profile fast --jobs 5 --out bench_results

bench-full:
	python -m nsa_flow.bench run --suite standard-full --profile standard --jobs 5 --out bench_results

bench-smoke:
	python -m nsa_flow.bench run --suite smoke --profile smoke --out bench_results

bench-check:
	python -m nsa_flow.bench check

bench-report:
	python -m nsa_flow.bench report standard-fast

speed:
	PYTHONPATH=. python experiments/speed_vs_pca.py

new-benchmarks:
	PYTHONPATH=. python experiments/benchmark_new_public_data.py
	PYTHONPATH=. python experiments/sweep_new_public_data.py
	PYTHONPATH=. python experiments/plot_new_public_benchmarks.py
	PYTHONPATH=. python experiments/build_new_public_tables.py

docs:
	PYTHONPATH=. sphinx-build -b html docs/source docs/_build/html

docs-clean:
	rm -rf docs/_build

paper:
	cd paper && latexmk -pdf -quiet nsaflow.tex

rmd-paper:
	Rscript -e 'rmarkdown::render("paper/nsa_flow.Rmd")'

dist:
	rm -rf dist/ build/ *.egg-info
	python -m build
	twine check dist/*

release: dist
	twine upload dist/*

all: experiments paper

clean:
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/
	cd paper && latexmk -C >/dev/null 2>&1 || true
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
