# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: Apache-2.0
import json
import pathlib
import re
import sys
import tempfile
from contextlib import ExitStack, contextmanager

import pytest
import soundfile as sf
from llm.cli import cli
from llm.plugins import pm
from PIL import Image


def image_validator(*sizes: tuple[int, int]):
    def validate(out: str):
        paths = out.splitlines()
        result = all(Image.open(path).size == size for size, path in zip(sizes, paths, strict=True))
        for path in paths:
            pathlib.Path(path).unlink(missing_ok=True)
        assert result

    return validate


def audio_validator(sample_rate: int):
    def validate(out: str):
        path = out.strip()
        actual_sample_rate = sf.read(path)[1]
        pathlib.Path(path).unlink(missing_ok=True)
        assert actual_sample_rate == sample_rate

    return validate


def equals_validator(value):
    def validate(out):
        assert value == out

    return validate


def json_validator(value: dict):
    def validate(out):
        assert value == json.loads(out)

    return validate


def regex_validator(regex: re.Pattern):
    def validate(out: str):
        assert regex.match(out)

    return validate


def startswith_validator(start: str):
    def validate(out: str):
        assert out.startswith(start)

    return validate


def segment_validator(out: str) -> bool:
    lines = out.splitlines()
    result = (
        all(
            line.split(" ", maxsplit=1)[1] == expected
            for line, expected in zip(lines, ["(bird: 0.999439)", "(bird: 0.998787)"], strict=True)
        ),
    )
    for line in lines:
        pathlib.Path(line.split(" ", maxsplit=1)[0]).unlink(missing_ok=True)
    assert result


@contextmanager
def prepare_table():
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete_on_close=False) as f:
        f.write(
            "Repository,Stars,Contributors,Programming language\n"
            "Transformers,36542,651,Python\n"
            "Datasets,4512,77,Python\n"
            'Tokenizers,3934,34,"Rust, Python and NodeJS'
        )
        f.close()
        yield f.name


# Assets from
# https://huggingface.co/datasets/Narsil/image_dummy/raw/main/lena.png
# https://huggingface.co/datasets/alfredplpl/video-to-video-dataset/resolve/main/easy/raising.mov
# http://images.cocodataset.org/val2017/000000039769.jpg
# https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png
# https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac
# https://huggingface.co/datasets/s3prl/Nonspeech/resolve/main/animal_sound/n52.wav
# https://huggingface.co/spaces/impira/docquery/resolve/2359223c1837a7587402bda0f2643382a6eefeab/invoice.png

testdata = {
    "audio-classification": (
        [
            "llm",
            "-m",
            "transformers",
            "-o",
            "verbose",
            "true",
            "-o",
            "task",
            "audio-classification",
            str(pathlib.Path(__file__).parent / "assets" / "1.flac"),
        ],
        startswith_validator("_unknown_ "),
    ),
    "automatic-speech-recognition": (
        [
            "llm",
            "-m",
            "transformers",
            "-o",
            "verbose",
            "true",
            "-o",
            "task",
            "automatic-speech-recognition",
            str(pathlib.Path(__file__).parent / "assets" / "1.flac"),
        ],
        equals_validator(
            (
                "HE HOPED THERE WOULD BE STEW FOR DINNER TURNIPS AND CARROTS AND BRUISED POTATOES AND FAT "
                "MUTTON PIECES TO BE LADLED OUT IN THICK PEPPERED FLOWER FAT AND SAUCE\n"
            ),
        ),
    ),
    "depth-estimation": (
        [
            "llm",
            "-m",
            "transformers",
            "-o",
            "verbose",
            "true",
            "-o",
            "task",
            "depth-estimation",
            str(pathlib.Path(__file__).parent / "assets" / "000000039769.jpg"),
        ],
        image_validator((640, 480)),
    ),
    "document-question-answering": (
        [
            "llm",
            "-m",
            "transformers",
            "-o",
            "verbose",
            "true",
            "-o",
            "task",
            "document-question-answering",
            "-o",
            "context",
            str(pathlib.Path(__file__).parent / "assets" / "invoice.png"),
            "What is the invoice number?",
        ],
        equals_validator("us-001\n"),
    ),
    "fill-mask": (
        [
            "llm",
            "-m",
            "transformers",
            "-o",
            "verbose",
            "true",
            "-o",
            "task",
            "fill-mask",
            "My <mask> is about to explode",
        ],
        regex_validator(
            re.compile(
                (
                    "My brain is about to explode \\(score=0.091\\d+\\)\n"
                    "My heart is about to explode \\(score=0.077\\d+\\)\n"
                    "My head is about to explode \\(score=0.051\\d+\\)\n"
                    "My fridge is about to explode \\(score=0.029\\d+\\)\n"
                    "My house is about to explode \\(score=0.028\\d+\\)\n"
                ),
                re.MULTILINE,
            )
        ),
    ),
    "image-classification": (
        [
            "llm",
            "-m",
            "transformers",
            "-o",
            "verbose",
            "true",
            "-o",
            "task",
            "image-classification",
            str(pathlib.Path(__file__).parent / "assets" / "parrots.png"),
        ],
        regex_validator(
            re.compile(
                (
                    "macaw \\(0.990\\d+\\)\n"
                    "African grey, African gray, Psittacus erithacus \\(0.005\\d+\\)\n"
                    "toucan \\(0.001\\d+\\)\n"
                    "sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita \\(0.000\\d+\\)\n"
                    "lorikeet \\(0.000\\d+\\)\n"
                ),
                re.MULTILINE,
            )
        ),
    ),
    "image-segmentation": (
        [
            "llm",
            "-m",
            "transformers",
            "-o",
            "verbose",
            "true",
            "-o",
            "task",
            "image-segmentation",
            str(pathlib.Path(__file__).parent / "assets" / "parrots.png"),
        ],
        segment_validator,
    ),
    "image-to-image": (
        [
            "llm",
            "-m",
            "transformers",
            "-o",
            "verbose",
            "true",
            "-o",
            "task",
            "image-to-image",
            str(pathlib.Path(__file__).parent / "assets" / "000000039769.jpg"),
        ],
        image_validator((1296, 976)),
    ),
    "image-to-text": (
        [
            "llm",
            "-m",
            "transformers",
            "-o",
            "verbose",
            "true",
            "-o",
            "task",
            "image-to-text",
            str(pathlib.Path(__file__).parent / "assets" / "parrots.png"),
        ],
        equals_validator("two birds are standing next to each other \n"),
    ),
    "object-detection": (
        [
            "llm",
            "-m",
            "transformers",
            "-o",
            "verbose",
            "true",
            "-o",
            "task",
            "object-detection",
            str(pathlib.Path(__file__).parent / "assets" / "parrots.png"),
        ],
        json_validator(
            [
                {
                    "score": pytest.approx(0.9966, abs=0.00009),
                    "label": "bird",
                    "box": {"xmin": 69, "ymin": 171, "xmax": 396, "ymax": 507},
                },
                {
                    "score": pytest.approx(0.9993, abs=0.00009),
                    "label": "bird",
                    "box": {"xmin": 398, "ymin": 105, "xmax": 767, "ymax": 507},
                },
            ],
        ),
    ),
    "question-answering": (
        [
            "llm",
            "-m",
            "transformers",
            "-o",
            "verbose",
            "true",
            "-o",
            "task",
            "question-answering",
            "-o",
            "context",
            "My name is Wolfgang and I live in Berlin",
            "Where do I live?",
        ],
        equals_validator("Berlin\n"),
    ),
    "summarization": (
        [
            "llm",
            "-m",
            "transformers",
            "-o",
            "verbose",
            "true",
            "-o",
            "task",
            "summarization",
            "-o",
            "kwargs",
            '{"min_length": 2, "max_length": 7}',
            "An apple a day, keeps the doctor away",
        ],
        equals_validator(" An apple a day\n"),
    ),
    "table-question-answering": (
        [
            "llm",
            "-m",
            "transformers",
            "-o",
            "verbose",
            "true",
            "-o",
            "task",
            "table-question-answering",
            "-o",
            "context",
            prepare_table,
            "How many stars does the transformers repository have?",
        ],
        equals_validator("AVERAGE > 36542\n"),
    ),
    "text2text-generation": (
        [
            "llm",
            "-m",
            "transformers",
            "-o",
            "verbose",
            "true",
            "-o",
            "task",
            "text2text-generation",
            "question: What is 42 ? context: 42 is the answer to life, the universe and everything",
        ],
        equals_validator("the answer to life, the universe and everything\n"),
    ),
    "text-classification": (
        [
            "llm",
            "-m",
            "transformers",
            "-o",
            "verbose",
            "true",
            "-o",
            "task",
            "text-classification",
            "We are very happy to show you the 🤗 Transformers library",
        ],
        regex_validator(re.compile("POSITIVE \\(0.999\\d+\\)\n", re.MULTILINE)),
    ),
    "text-generation": (
        [
            "llm",
            "-m",
            "transformers",
            "-o",
            "verbose",
            "true",
            "-o",
            "task",
            "text-generation",
            "I am going to elect",
        ],
        startswith_validator("I am going to elect"),
    ),
    "text-to-audio": (
        [
            "llm",
            "-m",
            "transformers",
            "-o",
            "verbose",
            "true",
            "-o",
            "kwargs",
            '{"generate_kwargs": {"max_new_tokens": 100}}',
            "-o",
            "model",
            "facebook/musicgen-small",
            "techno music",
        ],
        audio_validator(32000),
    ),
    "token-classification": (
        [
            "llm",
            "-m",
            "transformers",
            "-o",
            "verbose",
            "true",
            "-o",
            "task",
            "token-classification",
            "My name is Sarah and I live in London",
        ],
        regex_validator(
            re.compile("Sarah \\(I-PER: 0.998\\d+\\)\nLondon \\(I-LOC: 0.998\\d+\\)\n", re.MULTILINE)
        ),
    ),
    "translation_en_to_fr": (
        [
            "llm",
            "-m",
            "transformers",
            "-o",
            "verbose",
            "true",
            "-o",
            "task",
            "translation_en_to_fr",
            "How old are you?",
        ],
        equals_validator(" quel âge êtes-vous?\n"),
    ),
    "video-classification": (
        [
            "llm",
            "-m",
            "transformers",
            "-o",
            "verbose",
            "true",
            "-o",
            "task",
            "video-classification",
            str(pathlib.Path(__file__).parent / "assets" / "raising.mov"),
        ],
        regex_validator(
            re.compile(
                (
                    "stretching arm \\(0.\\d+\\)\n"
                    "yoga \\(0.03\\d+\\)\n"
                    "contact juggling \\(0.02\\d+\\)\n"
                    "belly dancing \\(0.01\\d+\\)\n"
                    "exercising arm \\(0.01\\d+\\)\n"
                ),
                re.MULTILINE,
            )
        ),
    ),
    "visual-question-answering": (
        [
            "llm",
            "-m",
            "transformers",
            "-o",
            "verbose",
            "true",
            "-o" "task",
            "visual-question-answering",
            "-o",
            "context",
            str(pathlib.Path(__file__).parent / "assets" / "lena.png"),
            "What is she wearing?",
        ],
        regex_validator(
            re.compile(
                (
                    "hat \\(0.948\\d+\\)\n"
                    "fedora \\(0.008\\d+\\)\n"
                    "clothes \\(0.003\\d+\\)\n"
                    "sun hat \\(0.002\\d+\\)\n"
                    "nothing \\(0.002\\d+\\)\n"
                ),
                re.MULTILINE,
            )
        ),
    ),
    "zero-shot-classification": (
        [
            "llm",
            "-m",
            "transformers",
            "-o",
            "verbose",
            "true",
            "-o",
            "task",
            "zero-shot-classification",
            "-o",
            "context",
            "urgent,not urgent,phone,tablet,computer",
            "I have a problem with my iphone that needs to be resolved asap!!",
        ],
        regex_validator(
            re.compile(
                (
                    "urgent \\(0.503\\d+\\)\n"
                    "phone \\(0.478\\d+\\)\n"
                    "computer \\(0.012\\d+\\)\n"
                    "not urgent \\(0.002\\d+\\)\n"
                    "tablet \\(0.002\\d+\\)\n"
                ),
                re.MULTILINE,
            )
        ),
    ),
    "zero-shot-image-classification": (
        [
            "llm",
            "-m",
            "transformers",
            "-o",
            "verbose",
            "true",
            "-o",
            "task",
            "zero-shot-image-classification",
            "-o",
            "context",
            "black and white,photorealist,painting",
            str(pathlib.Path(__file__).parent / "assets" / "parrots.png"),
        ],
        regex_validator(
            re.compile(
                (
                    "black and white \\(0.973\\d+\\)\n"
                    "photorealist \\(0.021\\d+\\)\n"
                    "painting \\(0.004\\d+\\)\n"
                ),
                re.MULTILINE,
            )
        ),
    ),
    "zero-shot-audio-classification": (
        [
            "llm",
            "-m",
            "transformers",
            "-o",
            "verbose",
            "true",
            "-o",
            "task",
            "zero-shot-audio-classification",
            "-o",
            "context",
            "Sound of a bird,Sound of a dog",
            str(pathlib.Path(__file__).parent / "assets" / "n52.wav"),
        ],
        regex_validator(
            re.compile(
                ("Sound of a bird \\(0.999\\d+\\)\n" "Sound of a dog \\(0.000\\d+\\)\n"), re.MULTILINE
            )
        ),
    ),
    "zero-shot-object-detection": (
        [
            "llm",
            "-m",
            "transformers",
            "-o",
            "verbose",
            "true",
            "-o",
            "task",
            "zero-shot-object-detection",
            "-o",
            "context",
            "cat,couch",
            str(pathlib.Path(__file__).parent / "assets" / "000000039769.jpg"),
        ],
        json_validator(
            [
                {
                    "score": pytest.approx(0.2868, abs=0.00009),
                    "label": "cat",
                    "box": {"xmin": 324, "ymin": 20, "xmax": 640, "ymax": 373},
                },
                {
                    "score": pytest.approx(0.2537, abs=0.00009),
                    "label": "cat",
                    "box": {"xmin": 1, "ymin": 55, "xmax": 315, "ymax": 472},
                },
                {
                    "score": pytest.approx(0.1208, abs=0.00009),
                    "label": "couch",
                    "box": {"xmin": 4, "ymin": 0, "xmax": 642, "ymax": 476},
                },
            ],
        ),
    ),
}

marks = {
    0: pytest.mark.llm0,
    1: pytest.mark.llm1,
    2: pytest.mark.llm2,
    3: pytest.mark.llm3,
}


@pytest.mark.parametrize(
    "llm_args,validator",
    [
        pytest.param(
            *values,
            id=key,
            # We bucket tests into different mark groups so they can be run on separate GitHub
            # actions runners (due to disk space issues on the runners)
            marks=[
                pytest.mark.llm,
                marks[hash(key) % len(marks)],
            ],
        )
        for key, values in testdata.items()
    ],
)
def test_transformer(monkeypatch, capsys, llm_args, validator):
    with ExitStack() as stack:
        # Replace any callable generator args with their generated result
        prepared_args = [stack.enter_context(arg()) if callable(arg) else arg for arg in llm_args]
        monkeypatch.setattr(sys, "argv", prepared_args)
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)  # prevent llm from trying to read from stdin
        monkeypatch.setattr(sys, "exit", lambda x=None: None)
        cli()
    captured = capsys.readouterr()
    validator(captured.out)


def test_plugin_is_installed():
    names = [mod.__name__ for mod in pm.get_plugins()]
    assert "llm_transformers" in names
