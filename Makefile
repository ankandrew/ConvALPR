# Directories
SRC_PATHS := alpr/ test/

# Tasks
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  help             : Show this help message"
	@echo "  format           : Format code using Ruff format"
	@echo "  check_format     : Check code formatting with Ruff format"
	@echo "  ruff             : Run Ruff linter"
	@echo "  pylint           : Run Pylint linter"
	@echo "  mypy             : Run MyPy static type checker"
	@echo "  lint             : Run linters (Ruff, Pylint and Mypy)"
	@echo "  test             : Run tests using pytest"
	@echo "  checks           : Check format, lint, and test"
	@echo "  clean            : Clean up caches and build artifacts"

.PHONY: format
format:
	@echo "==> Sorting imports..."
	@# Currently, the Ruff formatter does not sort imports, see https://docs.astral.sh/ruff/formatter/#sorting-imports
	@poetry run ruff check --select I --fix $(SRC_PATHS)
	@echo "=====> Formatting code..."
	@poetry run ruff format $(SRC_PATHS)

.PHONY: check_format
check_format:
	@echo "=====> Checking format..."
	@poetry run ruff format --check --diff $(SRC_PATHS)
	@echo "=====> Checking imports are sorted..."
	@poetry run ruff check --select I --exit-non-zero-on-fix $(SRC_PATHS)

.PHONY: ruff
ruff:
	@echo "=====> Running Ruff..."
	@poetry run ruff check $(SRC_PATHS)

.PHONY: pylint
pylint:
	@echo "=====> Running Pylint..."
	@poetry run pylint $(SRC_PATHS)

.PHONY: mypy
mypy:
	@echo "=====> Running Mypy..."
	@poetry run mypy $(SRC_PATHS)

.PHONY: lint
lint: ruff pylint mypy

.PHONY: test
test:
	@echo "=====> Running tests..."
	@poetry run pytest test/

.PHONY: clean
clean:
	@echo "=====> Cleaning caches..."
	@poetry run ruff clean
	@rm -rf .cache .pytest_cache .mypy_cache build dist *.egg-info

checks: format lint test
