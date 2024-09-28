import typing as ta
import tempfile
import json
import click
from pydantic import field_validator, model_validator, Field
import soundfile as sf
import llm
import torch
from transformers.pipelines import get_supported_tasks
from transformers.utils import get_available_devices
from transformers import pipeline
import PIL

# XXX get list of models? https://github.com/huggingface/transformers/blob/2e24ee4dfa39cc0bc264b89edbccc373c8337086/src/transformers/models/auto/configuration_auto.py#L637
# XXX allow local paths?
#XXX query pipeline for model used to determine output format info? pipe.model
#XXX query pipeline for model and see if it supports streaming, and pass model stream kwargs


@llm.hookimpl
def register_commands(cli):
    @cli.group(name="transformers")
    def transformers_group():
        "Commands for working with Hugging Face Transformers models"

    @transformers_group.command(name="list-tasks")
    def list_tasks():
        """List supported transformers task names."""
        for task in get_supported_tasks():
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

    class Options(llm.Options):
        task: str | None = Field(
            description="Transformer pipeline task name. `llm transformers list-tasks`.",
            default=None
        )
        model: str | None = Field(
            description="Transformer model name.",
            default=None
        )
        kwargs: dict | None = Field(
            description="Pipeline keyword args JSON dict. Specify additional kwargs for some pipelines.",
            default=None
        )
        device: str | None = Field(
            description="Device name. `llm transformers list-devices`.",
            default=None
        )

        @field_validator("kwargs", mode='before')
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
            if task not in get_supported_tasks():
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

        @model_validator(mode='after')
        def check_task_model(self) -> ta.Self:
            if self.task is None and self.model is None:
                raise ValueError("Must specify pipeline task and/or model options.")
            return self

    def execute(self,
        prompt: llm.Prompt,
        stream: bool,
        response: llm.Response,
        conversation: llm.Conversation | None =None,
    ) -> ta.Iterator[str]:
        pipe = pipeline(
            task=prompt.options.task,
            model=prompt.options.model,
            device=torch.device(prompt.options.device) if prompt.options.device is not None else None,
        )

        args = []
        kwargs = prompt.options.kwargs if prompt.options.kwargs is not None else {}
        if pipe.task == "document-question-answering":
            if "image" not in kwargs:
                raise ValueError("Must specify 'image' path/URL kwargs option for document-question-answering pipeline task.")
            kwargs["question"] = prompt.prompt
        else:
            args.append(prompt.prompt)

        result = pipe(*args, **kwargs)

        match result:
            case str():
                yield result
            case {"text": text}:  # automatic-speech-recognition
                response.response_json = {pipe.task: result}
                yield text
            case {"audio": audio, "sampling_rate": sampling_rate}:  # text-to-audio
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, delete_on_close=False) as f:
                    # musicgen is shape (batch_size, num_channels, sequence_length)
                    # https://huggingface.co/docs/transformers/v4.45.1/en/model_doc/musicgen#unconditional-generation
                    #XXX check shape of other audio pipelines
                    sf.write(f, audio[0].T, sampling_rate)
                    response.response_json = {pipe.task: {"output": f.name}}
                    yield f.name
            case [{"score": float(), "label": str()}, *_]:  # audio-classification, sentiment-analysis
                response.response_json = {pipe.task: result}
                yield json.dumps(result, indent=4)
            case [{"sequence": str(), "token": int(), "token_str": str(), "score": float()}, *_]:  # fill-mask
                response.response_json = {pipe.task: result}
                yield "\n".join(f"{item['sequence']} (score={item['score']})" for item in result)
            case {"predicted_depth": _, "depth": depth}:  # depth-estimation
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False, delete_on_close=False) as f:
                    depth.save(f, format="png")
                    response.response_json = {pipe.task: {"output": f.name}}
                    yield f.name
            case [{"score": float(), "answer": str(), "start": int(), "end": int()}]:  # document-question-answering
                response.response_json = {pipe.task: result}
                yield result[0]["answer"]
            case _:
                breakpoint()
                print("DEFAULT CASE") #XXX
                yield json.dumps(result, indent=4)
