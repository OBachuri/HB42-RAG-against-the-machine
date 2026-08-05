.ONESHELL:
SHELL := /bin/bash


RM = rm -fr

VENV_DIR := .cmm-venv
PYTHON := python3
PIP := pip
C_DIR := pwd


UV := $(shell command -v uv 2>/dev/null)

ACOMMAND := chunk index search "How to configure OpenAI server?" answer "How to configure OpenAI server?" 
ARGS := --max_chunk_size 2000 --k=10
# ARGS := $(wordlist 2, 999, $(MAKECMDGOALS))

help:
	@echo  "uv run python -m src index"

install:
#	mkdir -p .cache/uv_cache .cache/hf_cache
#	UV_CACHE_DIR=.cache/uv_cache \ 
#	HF_HOME=.cache/hf_cache \ 

	uv sync --python 3

run:
	uv run python -m src $(ACOMMAND) $(ARGS)



clean:
	@$(RM) .mypy_cache
	@$(RM) __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -name .pytest_cache -exec rm -rf {} +
	find . -name .ruff_cache -exec rm -rf {} +
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	@$(RM) data/cache/

fclean: clean
	@$(RM) .cache
	@$(RM) .venv
	@$(RM) data/processed/
	@$(RM) data/output/

lint:
	uv run flake8 src/*.py
	uv run mypy src/ --warn-return-any \
	--warn-unused-ignores \
	--ignore-missing-imports \
	--disallow-untyped-defs \
	--check-untyped-defs \
	--follow-imports=silent

lint-strict: 
	uv run flake8 src/*.py
	uv run mypy src/ --strict --follow-imports=silent


.PHONY: install, run, debug, clean, lint, lint-strict fclean

#  ln -s ~/goinfre/uv uv
