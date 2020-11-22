.PHONY: install format lint

help:
	@echo "Available commands:"
	@echo "install          install required packages."
	@echo "format           run formatters."
	@echo "lint             run linters."

install:
	pip install -r requirements.txt
	pip install -e .

format:
	isort -rc --atomic hoqunm
	yapf -i --recursive hoqunm
	docformatter -i -r hoqunm

lint:
	yapf --diff --recursive hoqunm
	mypy hoqunm
	pylint -v hoqunm
