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
    def validator(out: str) -> bool:
        paths = out.splitlines()
        result = all(Image.open(path).size == size for size, path in zip(sizes, paths, strict=True))
        for path in paths:
            pathlib.Path(path).unlink(missing_ok=True)
        assert result

    return validator


def audio_validator(sample_rate: int):
    def validator(out: str) -> bool:
        path = out.strip()
        actual_sample_rate = sf.read(path)[1]
        pathlib.Path(path).unlink(missing_ok=True)
        assert actual_sample_rate == sample_rate

    return validator


def equals_validator(a, b):
    assert a == b


def regex_validator(value, regex):
    assert re.match(regex, value, re.MULTILINE)


def startswith_validator(out: str, start: str):
    assert out.startswith(start)


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
            "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac",
        ],
        lambda out: startswith_validator(out, "_unknown_ "),
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
            "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac",
        ],
        lambda out: equals_validator(
            out,
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
            "http://images.cocodataset.org/val2017/000000039769.jpg",
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
            "https://huggingface.co/spaces/impira/docquery/resolve/2359223c1837a7587402bda0f2643382a6eefeab/invoice.png",
            "What is the invoice number?",
        ],
        lambda out: equals_validator(out, "us-001\n"),
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
        lambda out: equals_validator(
            out,
            (
                "My brain is about to explode (score=0.09140042215585709)\n"
                "My heart is about to explode (score=0.07742168009281158)\n"
                "My head is about to explode (score=0.05137857422232628)\n"
                "My fridge is about to explode (score=0.029346412047743797)\n"
                "My house is about to explode (score=0.02866862528026104)\n"
            ),
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
            "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png",
        ],
        lambda out: equals_validator(
            out,
            (
                "macaw (0.9905233979225159)\n"
                "African grey, African gray, Psittacus erithacus (0.005603480152785778)\n"
                "toucan (0.001056905253790319)\n"
                "sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita (0.0006811501225456595)\n"
                "lorikeet (0.0006714339251630008)\n"
            ),
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
            "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png",
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
            "http://images.cocodataset.org/val2017/000000039769.jpg",
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
            "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png",
        ],
        lambda out: equals_validator(out, "two birds are standing next to each other \n"),
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
            "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png",
        ],
        lambda out: equals_validator(
            json.loads(out),
            [
                {
                    "score": 0.9966394901275635,
                    "label": "bird",
                    "box": {"xmin": 69, "ymin": 171, "xmax": 396, "ymax": 507},
                },
                {
                    "score": 0.999381422996521,
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
        lambda out: equals_validator(out, "Berlin\n"),
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
        lambda out: equals_validator(out, " An apple a day\n"),
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
        lambda out: equals_validator(out, "AVERAGE > 36542\n"),
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
        lambda out: equals_validator(out, "the answer to life, the universe and everything\n"),
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
        lambda out: equals_validator(out, "POSITIVE (0.9997681975364685)\n"),
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
        lambda out: startswith_validator(out, "I am going to elect"),
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
        lambda out: equals_validator(
            out, "Sarah (I-PER: 0.9982994198799133)\nLondon (I-LOC: 0.998397171497345)\n"
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
        lambda out: equals_validator(out, " quel âge êtes-vous?\n"),
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
            "https://huggingface.co/datasets/Xuehai/MMWorld/resolve/main/Amazing%20street%20dance%20performance%20from%20Futunity%20UK%20-%20Move%20It%202013/Amazing%20street%20dance%20performance%20from%20Futunity%20UK%20-%20Move%20It%202013.mp4",
        ],
        lambda out: equals_validator(
            out,
            (
                "dancing ballet (0.006608937866985798)\n"
                "spinning poi (0.006111182738095522)\n"
                "air drumming (0.005756791681051254)\n"
                "singing (0.005747966933995485)\n"
                "punching bag (0.00565463537350297)\n"
            ),
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
            "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/lena.png",
            "What is she wearing?",
        ],
        lambda out: regex_validator(
            out,
            (
                "hat \\(0.948026\\d+\\)\n"
                "fedora \\(0.00863\\d+\\)\n"
                "clothes \\(0.003124\\d+\\)\n"
                "sun hat \\(0.002937\\d+\\)\n"
                "nothing \\(0.002096\\d+\\)\n"
            ),
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
        lambda out: equals_validator(
            out,
            (
                "urgent (0.5036348700523376)\n"
                "phone (0.4788002371788025)\n"
                "computer (0.012600351125001907)\n"
                "not urgent (0.0026557915844023228)\n"
                "tablet (0.0023087668232619762)\n"
            ),
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
            "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png",
        ],
        lambda out: equals_validator(
            out,
            (
                "black and white (0.9736384749412537)\n"
                "photorealist (0.02141517587006092)\n"
                "painting (0.004946451168507338)\n"
            ),
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
            "https://huggingface.co/datasets/s3prl/Nonspeech/resolve/main/animal_sound/n52.wav",
        ],
        lambda out: equals_validator(
            out,
            ("Sound of a bird (0.9998763799667358)\n" "Sound of a dog (0.00012355657236184925)\n"),
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
            "http://images.cocodataset.org/val2017/000000039769.jpg",
        ],
        lambda out: equals_validator(
            json.loads(out),
            [
                {
                    "score": 0.2868139445781708,
                    "label": "cat",
                    "box": {"xmin": 324, "ymin": 20, "xmax": 640, "ymax": 373},
                },
                {
                    "score": 0.2537268102169037,
                    "label": "cat",
                    "box": {"xmin": 1, "ymin": 55, "xmax": 315, "ymax": 472},
                },
                {
                    "score": 0.12082991003990173,
                    "label": "couch",
                    "box": {"xmin": 4, "ymin": 0, "xmax": 642, "ymax": 476},
                },
            ],
        ),
    ),
}


@pytest.mark.parametrize("llm_args,validator", testdata.values(), ids=testdata.keys())
def test_transformer(monkeypatch, capsys, llm_args, validator):
    with ExitStack() as stack:
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
