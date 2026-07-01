import json
import os
import re
from typing import Any

import fsspec
import yaml
from coqpit import Coqpit

from TTS.config.shared_configs import BaseTrainingConfig


def read_json_with_comments(json_path):
    with fsspec.open(json_path, "r", encoding="utf-8") as f:
        input_str = f.read()
    input_str = re.sub(
        r"(\"(?:[^\"\\]|\\.)*\")|(/\*(?:.|[\\n\\r])*?\*/)|(//.*)",
        lambda m: m.group(1) or m.group(2) or "",
        input_str,
    )
    return json.loads(input_str)


def register_config(model_name: str) -> type[BaseTrainingConfig]:
    model_name = model_name.lower()
    if model_name == "vits":
        from TTS.tts.configs.vits_config import VitsConfig
        return VitsConfig
    raise ModuleNotFoundError(f" [!] Config for {model_name} cannot be found.")


def _process_model_name(config_dict: dict) -> str:
    if "model" not in config_dict:
        raise KeyError("Missing 'model' in config.")
    return str(config_dict["model"]).replace("_generator", "").replace("_discriminator", "")


def load_config(config_path: str | os.PathLike[Any]) -> BaseTrainingConfig:
    config_path = str(config_path)
    ext = os.path.splitext(config_path)[1]

    if ext in (".yml", ".yaml"):
        with fsspec.open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    elif ext == ".json":
        try:
            with fsspec.open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = read_json_with_comments(config_path)
    else:
        raise TypeError(f" [!] Unknown config file type {ext}")

    model_name = _process_model_name(data)
    config_class = register_config(model_name)
    config = config_class()
    config.from_dict(data)
    return config


def get_from_config_or_model_args(config: Coqpit, arg_name: str, def_val: Any = None) -> Any:
    if getattr(config, "model_args", None) is not None and arg_name in config.model_args:
        return config.model_args[arg_name]
    if hasattr(config, arg_name):
        return config[arg_name]
    return def_val