#!/usr/bin/env bash
# Run all quranic_nlp tests using pytest.
# Usage: ./tests/run_tests.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$REPO_ROOT"
python -m pytest tests/ -v --tb=short "$@"
