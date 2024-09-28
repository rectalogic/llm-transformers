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

Models that generate audio will save the audio to a file, the pathname is the output of the `llm` command.
Some models can also be parameterized with keyword arguments.
```sh-session
$ llm -m transformers -o kwargs '{"generate_kwargs": {"max_new_tokens": 100}}' -o model facebook/musicgen-small "techno music"
/var/folders/b1/1j9kkk053txc5krqbh0lj5t00000gn/T/tmpoueh05y6.wav
```

Some models can take a URL or path to a file as input, for example:
```sh-session
$ llm -m transformers -o task audio-classification https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac
[
    {
        "score": 0.9972336888313293,
        "label": "_unknown_"
    },
    {
        "score": 0.0019911774434149265,
        "label": "left"
    },
    {
        "score": 0.0003051063104066998,
        "label": "yes"
    },
    {
        "score": 0.0002108386834152043,
        "label": "down"
    },
    {
        "score": 0.00011406492558307946,
        "label": "stop"
    }
]
```
```sh-session
$ llm -m transformers -o task automatic-speech-recognition https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac
HE HOPED THERE WOULD BE STEW FOR DINNER TURNIPS AND CARROTS AND BRUISED POTATOES AND FAT MUTTON PIECES TO BE LADLED OUT IN THICK PEPPERED FLOWER FAT AND SAUCE
```
```sh-session
$ llm -m transformers -o task sentiment-analysis "We are very happy to show you the 🤗 Transformers library"
[
    {
        "label": "POSITIVE",
        "score": 0.9997681975364685
    }
]
```
Some pipeline tasks accept an image url or path as input and generate an image file as output:
```sh-session
$ llm -m transformers -o task depth-estimation http://images.cocodataset.org/val2017/000000039769.jpg
/var/folders/b1/1j9kkk053txc5krqbh0lj5t00000gn/T/tmpjvp9uo7x.png
```
XXX embed image here?

Some pipeline tasks require specific kwarg options:
```sh-session
$ llm -m transformers -o task document-question-answering -o kwargs '{"image": "https://huggingface.co/spaces/impira/docquery/resolve/2359223c1837a7587402bda0f2643382a6eefeab/invoice.png"}' "What is the invoice number?"
us-001
```

```sh-session
$ llm -m transformers -o task fill-mask "My <mask> is about to explode"
My brain is about to explode (score=0.09140042215585709)
My heart is about to explode (score=0.07742168009281158)
My head is about to explode (score=0.05137857422232628)
My fridge is about to explode (score=0.029346412047743797)
My house is about to explode (score=0.02866862528026104)
```

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
