.PHONY: test format

format:
	ruff format .
	ruff check . --fix

test:
	pytest -q --cov-config=pyproject.toml
