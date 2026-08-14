.PHONY: setup ingest query eval demo clean

setup:
	python -m venv .venv
	.venv/bin/pip install -e ".[dev]"
	@echo "Run: source .venv/bin/activate"

ingest:
	python -m scripts.ingest

eval:
	python -m scripts.run_eval

demo:
	streamlit run demo/app.py

clean:
	rm -rf data/raw data/parsed data/index
