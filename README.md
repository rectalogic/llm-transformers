# llm-transformers

[![PyPI](https://img.shields.io/pypi/v/llm-transformers.svg)](https://pypi.org/project/llm-transformers/)
[![Changelog](https://img.shields.io/github/v/release/rectalogic/llm-transformers?include_prereleases&label=changelog)](https://github.com/rectalogic/llm-transformers/releases)
[![Tests](https://github.com/rectalogic/llm-transformers/actions/workflows/test.yml/badge.svg)](https://github.com/rectalogic/llm-transformers/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/rectalogic/llm-transformers/blob/main/LICENSE)

Plugin for llm adding support for Hugging Face Transformers

## Installation

Install this plugin in the same environment as [LLM](https://llm.datasette.io/).
```bash
llm install llm-transformers
```
## Usage

Usage instructions go here.

## Development

To set up this plugin locally, first checkout the code. Then create a new virtual environment:
```bash
cd llm-transformers
python -m venv venv
source venv/bin/activate
```
Now install the dependencies and test dependencies:
```bash
llm install -e '.[test]'
```
To run the tests:
```bash
python -m pytest
```
