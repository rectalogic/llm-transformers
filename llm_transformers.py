import typing as ta
import tempfile
import json
import csv
import re
import itertools
import click
from pydantic import field_validator, model_validator, Field, ConfigDict
import soundfile as sf
import llm
import torch
from transformers.pipelines import get_supported_tasks, Pipeline
from transformers.utils import get_available_devices
from transformers import pipeline
from PIL import Image


TASK_BLACKLIST = (
    "feature-extraction",
    "image-feature-extraction",
    "mask-generation",  # Generates list of "masks" (numpy.ndarray(H, W) of dtype('bool')) and "scores" (Tensors)
)


def supported_tasks() -> ta.Iterator[str]:
    for task in get_supported_tasks():
        if task not in TASK_BLACKLIST:
            yield task


def save_image(image: Image.Image) -> str:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False, delete_on_close=False) as f:
        image.save(f, format="png")
        return f.name


def handle_required_kwarg(kwargs: dict, options: llm.Options, name: str, format: str, task: str) -> None:
    if name not in kwargs:
        kwargs[name] = getattr(options, name, None)
    if kwargs[name] is None:
        raise llm.ModelError(f"Must specify '-o {name} {format}' option for {task} pipeline task.")


@llm.hookimpl
def register_commands(cli):
    @cli.group(name="transformers")
    def transformers_group():
        "Commands for working with Hugging Face Transformers models"

    @transformers_group.command(name="list-tasks")
    def list_tasks():
        """List supported transformers task names."""
        for task in supported_tasks():
            click.echo(task)

    @transformers_group.command(name="list-devices")
    def list_devices():
        """List available device names."""
        for device in get_available_devices():
            click.echo(device)


@llm.hookimpl
def register_models(register):
    register(Transformers())


class Transformers(llm.Model):
    model_id = "transformers"

    pipe: Pipeline | None = None

    class Options(llm.Options):
        task: str | None = Field(
            description="Transformer pipeline task name. `llm transformers list-tasks`.", default=None
        )
        model: str | None = Field(description="Transformer model name.", default=None)
        kwargs: dict | None = Field(
            description="Pipeline keyword args JSON dict. Specify additional kwargs for some pipelines.",
            default=None,
        )
        device: str | None = Field(
            description="Device name. `llm transformers list-devices`.", default=None
        )
        # Pass through additional options
        model_config = ConfigDict(extra="allow")

        @field_validator("kwargs", mode="before")
        @classmethod
        def validate_kwargs(cls, kwargs) -> dict | None:
            if kwargs is None or isinstance(kwargs, dict):
                return kwargs
            d = json.loads(kwargs)
            if not isinstance(d, dict):
                raise ValueError("Invalid pipeline kwargs JSON option.")
            return d

        @field_validator("task")
        @classmethod
        def validate_task(cls, task) -> str | None:
            if task is None:
                return None
            if task not in supported_tasks():
                if re.match("translation_.._to_..", task):
                    return task
                raise ValueError("Invalid pipeline task name option.")
            return task

        @field_validator("device")
        @classmethod
        def validate_device(cls, device) -> str | None:
            if device is None:
                return None
            if device not in get_available_devices():
                raise ValueError("Invalid device name option.")
            return device

        @model_validator(mode="after")
        def check_task_model(self) -> ta.Self:
            if self.task is None and self.model is None:
                raise ValueError("Must specify pipeline task and/or model options.")
            return self

    def execute(
        self,
        prompt: llm.Prompt,
        stream: bool,
        response: llm.Response,
        conversation: llm.Conversation | None = None,
    ) -> ta.Iterator[str]:
        if self.pipe is None:
            self.pipe = pipeline(
                task=prompt.options.task,
                model=prompt.options.model,
                device=torch.device(prompt.options.device) if prompt.options.device is not None else None,
                framework="pt",
            )
        elif (prompt.options.task and self.pipe.task != prompt.options.task) or (
            prompt.options.model and self.pipe.model.name_or_path != prompt.options.model
        ):
            raise llm.ModelError("'task' or 'model' options have changed")

        if self.pipe.task in TASK_BLACKLIST:
            raise llm.ModelError(f"{self.pipe.task} pipeline task is not supported.")

        args = []
        kwargs = prompt.options.kwargs if prompt.options.kwargs is not None else {}
        match self.pipe.task:
            case "document-question-answering" | "visual-question-answering":
                kwargs["question"] = prompt.prompt
                handle_required_kwarg(kwargs, prompt.options, "image", "<imagefile/URL>", self.pipe.task)
            case "question-answering":
                kwargs["question"] = prompt.prompt
                handle_required_kwarg(kwargs, prompt.options, "context", "<text>", self.pipe.task)
            case "table-question-answering":
                kwargs["query"] = prompt.prompt
                handle_required_kwarg(kwargs, prompt.options, "table", "<csvfile>", self.pipe.task)
                # Convert CSV to a dict of lists, keys are the header names and values are a list of the column values
                with open(kwargs["table"]) as f:
                    reader = csv.reader(f)
                    headers = next(reader)  # get the column headers
                    table = {header: [] for header in headers}
                    for row in reader:
                        for i, header in enumerate(headers):
                            table[header].append(row[i])
                kwargs["table"] = table
            case "video-classification":
                # Prompt should be a video file/URL
                kwargs["videos"] = prompt.prompt
            case "zero-shot-classification":
                kwargs["sequences"] = prompt.prompt
                handle_required_kwarg(kwargs, prompt.options, "candidate_labels", "<label,label,...>", self.pipe.task)
            case "zero-shot-image-classification" | "zero-shot-audio-classification" | "zero-shot-object-detection":
                # prompt is audio or image url/path
                args.append(prompt.prompt)
                handle_required_kwarg(kwargs, prompt.options, "candidate_labels", "<label,label,...>", self.pipe.task)
                kwargs["candidate_labels"] = kwargs["candidate_labels"].split(",")
            case _:
                if self.pipe.tokenizer.chat_template is not None:
                    messages = []
                    if conversation is not None:
                        messages.extend(
                            itertools.chain.from_iterable(
                                (
                                    {"role": "user", "content": prev_response.prompt.prompt},
                                    {"role": "assistant", "content": prev_response.text()},
                                )
                                for prev_response in conversation.responses
                            )
                        )
                    messages.append({"role": "user", "content": prompt.prompt})
                    args.append(messages)
                else:
                    args.append(prompt.prompt)

        result = self.pipe(*args, **kwargs)

        match self.pipe.task, result:
            case "image-to-image", Image.Image() as image:
                path = save_image(image)
                response.response_json = {self.pipe.task: {"output": path}}
                yield path
            case "automatic-speech-recognition", {"text": str(text)}:
                response.response_json = {self.pipe.task: result}
                yield text
            case "text-to-audio", {"audio": audio, "sampling_rate": int(sampling_rate)}:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, delete_on_close=False) as f:
                    # musicgen is shape (batch_size, num_channels, sequence_length)
                    # https://huggingface.co/docs/transformers/v4.45.1/en/model_doc/musicgen#unconditional-generation
                    # XXX check shape of other audio pipelines
                    sf.write(f, audio[0].T, sampling_rate)
                    response.response_json = {self.pipe.task: {"output": f.name}}
                    yield f.name
            case "object-detection", [
                {
                    "score": float(),
                    "label": str(),
                    "box": {"xmin": int(), "ymin": int(), "xmax": int(), "ymax": int()},
                },
                *_,
            ]:
                yield json.dumps(result, indent=4)
            case "image-segmentation", [{"score": float(), "label": str(), "mask": Image.Image()}, *_]:
                responses = []
                for item in result:
                    path = save_image(item["mask"])
                    responses.append({"score": item["score"], "label": item["label"], "output": path})
                response.response_json = {self.pipe.task: responses}
                yield "\n".join(
                    f"{item['output']} ({item['label']}: {item['score']})" for item in responses
                )
            case "audio-classification" | "image-classification" | "sentiment-analysis" | "text-classification", [
                {"score": float(), "label": str()},
                *_,
            ]:
                response.response_json = {self.pipe.task: result}
                yield json.dumps(result, indent=4)
            case "question-answering", {
                "score": float(),
                "start": int(),
                "end": int(),
                "answer": str(answer),
            }:
                response.response_json = {self.pipe.task: result}
                yield answer
            case "fill-mask", [
                {"sequence": str(), "token": int(), "token_str": str(), "score": float()},
                *_,
            ]:
                response.response_json = {self.pipe.task: result}
                yield "\n".join(f"{item['sequence']} (score={item['score']})" for item in result)
            case "depth-estimation", {"predicted_depth": torch.Tensor(), "depth": Image.Image(depth)}:
                path = save_image(depth)
                response.response_json = {self.pipe.task: {"output": path}}
                yield path
            case "document-question-answering", [
                {"score": float(), "answer": str(), "start": int(), "end": int()}
            ]:
                response.response_json = {self.pipe.task: result}
                yield result[0]["answer"]
            case "image-to-text" | "text2text-generation" | "text-generation", [{"generated_text": str(text)}]:
                response.response_json = {self.pipe.task: result}
                yield text
            case "text-generation", [
                {"generated_text": [{"role": ("user" | "assistant"), "content": str()}, *_]}
            ]:
                response.response_json = {self.pipe.task: result}
                yield result[0]["generated_text"][-1]["content"]
            case "summarization", [{"summary_text": str(text)}]:
                response.response_json = {self.pipe.task: result}
                yield text
            case "table-question-answering", {
                "answer": str(answer),
                "coordinates": [(int(), int()), *_],
                "cells": [str(), *_],
                "aggregator": str(),
            }:
                response.response_json = {self.pipe.task: result}
                yield answer
            case "token-classification", [{'entity': str(), 'score': _, 'index': int(), 'word': str(), 'start': int(), 'end': int()}, *_]:
                response.response_json = {self.pipe.task: result}
                yield "\n".join(f"{item['word']} ({item['entity']}: {item['score']})" for item in result)
            case task, [{'translation_text': str(text)}] if task.startswith("translation"):
                # translation_xx_to_yy tasks encode the language codes e.g. translation_en_to_fr
                response.response_json = {self.pipe.task: result}
                yield text
            case "zero-shot-object-detection", [{'score': float(), 'label': str(), "box": {'xmin': int(), 'ymin': int(), 'xmax': int(), 'ymax': int()}}, *_]:
                response.response_json = {self.pipe.task: result}
                yield json.dumps(result, indent=4)
            case "video-classification" | "zero-shot-image-classification" | "zero-shot-audio-classification", [{'score': float(), 'label': str()}, *_]:
                response.response_json = {self.pipe.task: result}
                yield "\n".join(f"{item['label']} ({item['score']})" for item in result)
            case "visual-question-answering", [{'score': float(), 'answer': str()}, *_]:
                response.response_json = {self.pipe.task: result}
                yield "\n".join(f"{item['answer']} ({item['score']})" for item in result)
            case "zero-shot-classification", {'sequence': str(), 'labels': [str(), *_] as labels, 'scores': [float(), *_] as scores}:
                response.response_json = {self.pipe.task: result}
                yield "\n".join(f"{label} ({score})" for label, score in zip(labels, scores, strict=True))
            case _, _:
                breakpoint()
                print("DEFAULT CASE")  # XXX
                yield json.dumps(result, indent=4)
