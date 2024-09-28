import typing as ta
import json
import click
from pydantic import field_validator, model_validator, Field
import llm
import torch
from transformers.pipelines import get_supported_tasks
from transformers.utils import get_available_devices
from transformers import pipeline

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
            description="Transformer task name. `llm transformers list-tasks`.",
            default=None
        )
        model: str | None = Field(
            description="Transformer model name.",
            default=None
        )
        device: str | None = Field(
            description="Device name. `llm transformers list-devices`.",
            default=None
        )
        # model: str | None = None
        # XXX add tokenizer, device (cpu/gpu)

        @field_validator("task")
        @classmethod
        def validate_task(cls, task) -> ta.Any:
            if task is None:
                return None
            if task not in get_supported_tasks():
                raise ValueError("Invalid task name.")
            return task

        @field_validator("device")
        @classmethod
        def validate_device(cls, device) -> ta.Any:
            if device is None:
                return None
            if device not in get_available_devices():
                raise ValueError("Invalid device name.")
            return device

        @model_validator(mode='after')
        def check_task_model(self) -> ta.Self:
            if self.task is None and self.model is None:
                raise ValueError("Must specify task and/or model.")
            return self

    def execute(self,
        prompt: llm.Prompt,
        stream: bool,
        response: llm.Response,
        conversation: llm.Conversation | None =None,
    ):
        #XXX hrm, passing model alone only works if it is already associated with a task - seems ok, e.g. llama3-3.1-8B has pipeline_tag text-generation
        pipe = pipeline(task=prompt.options.task, model=prompt.options.model, device=torch.device(prompt.options.device) if prompt.options.device is not None else None)
        result = pipe(prompt.prompt)
        if not isinstance(result, str):
            return json.dumps(result)
        return result
