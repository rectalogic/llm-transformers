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
Some models can also be parameterized with keyword arguments specified as a string of JSON.
```sh-session
# text-to-audio pipeline
$ llm -m transformers -o kwargs '{"generate_kwargs": {"max_new_tokens": 100}}' -o model facebook/musicgen-small "techno music"
/var/folders/b1/1j9kkk053txc5krqbh0lj5t00000gn/T/tmpoueh05y6.wav
```

Some models can take a URL or path to an audio or video as input, for example:
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
$ llm -m transformers -o task image-classification https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png
[
    {
        "label": "macaw",
        "score": 0.9905233979225159
    },
    {
        "label": "African grey, African gray, Psittacus erithacus",
        "score": 0.005603480152785778
    },
    {
        "label": "toucan",
        "score": 0.001056905253790319
    },
    {
        "label": "sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita",
        "score": 0.0006811501225456595
    },
    {
        "label": "lorikeet",
        "score": 0.0006714339251630008
    }
]
```
```sh-session
$ llm -m transformers -o task image-segmentation https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png
/var/folders/b1/1j9kkk053txc5krqbh0lj5t00000gn/T/tmp0z8zvd8i.png (bird: 0.999439)
/var/folders/b1/1j9kkk053txc5krqbh0lj5t00000gn/T/tmpik_7r5qn.png (bird: 0.998787)
```
```sh-session
$ llm -m transformers -o task image-to-image http://images.cocodataset.org/val2017/000000039769.jpg
/var/folders/b1/1j9kkk053txc5krqbh0lj5t00000gn/T/tmpczogz6cb.png
```
```sh-session
$ llm -m transformers -o task image-to-text https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png
two birds are standing next to each other
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

``sh-session
$ llm -m transformers -o task object-detection https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png
[
    {
        "score": 0.9966394901275635,
        "label": "bird",
        "box": {
            "xmin": 69,
            "ymin": 171,
            "xmax": 396,
            "ymax": 507
        }
    },
    {
        "score": 0.999381422996521,
        "label": "bird",
        "box": {
            "xmin": 398,
            "ymin": 105,
            "xmax": 767,
            "ymax": 507
        }
    }
]
```

Some pipeline tasks require additional specific kwarg options, these can be specified as `kwargs` JSON or
more conveniently by specifying an additional model option:
```sh-session
$ llm -m transformers -o task document-question-answering -o kwargs '{"image": "https://huggingface.co/spaces/impira/docquery/resolve/2359223c1837a7587402bda0f2643382a6eefeab/invoice.png"}' "What is the invoice number?"
us-001
$ llm -m transformers -o task document-question-answering -o image https://huggingface.co/spaces/impira/docquery/resolve/2359223c1837a7587402bda0f2643382a6eefeab/invoice.png "What is the invoice number?"
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

```sh-session
$ llm -m transformers -o task question-answering -o context "My name is Wolfgang and I live in Berlin" "Where do I live?"
Berlin
```

``sh-session
$ llm -m transformers -o task summarization "An apple a day, keeps the doctor away"
 An apple a day, keeps the doctor away from your doctor away . An apple every day is an apple that keeps you from going to the doctor . The apple is the best way to keep your doctor from getting a doctor's orders, according to the author of The Daily Mail
$ llm -m transformers -o task summarization -o kwargs '{"min_length": 2, "max_length": 7}' "An apple a day, keeps the doctor away"
 An apple a day
```

``sh-session
$ cat <<EOF > /tmp/t.csv
> Repository,Stars,Contributors,Programming language
Transformers,36542,651,Python
Datasets,4512,77,Python
Tokenizers,3934,34,"Rust, Python and NodeJS"
> EOF
$ llm -m transformers -o task table-question-answering -o table /tmp/t.csv "How many stars does the transformers repository have?"
AVERAGE > 36542
$ llm -m transformers -o task table-question-answering -o table /tmp/t.csv "How many contributors do all Python language repositories have?"
SUM > 651, 77
```

```sh-session
$ llm -m transformers -o task text2text-generation "question: What is 42 ? context: 42 is the answer to life, the universe and everything"
the answer to life, the universe and everything
```

```sh-session
$ llm -m transformers -o task text-classification "This movie is disgustingly good !"
[
    {
        "label": "POSITIVE",
        "score": 0.9998536109924316
    }
]
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
