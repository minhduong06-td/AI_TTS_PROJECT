# -*- coding: utf-8 -*-
import datetime
import importlib
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict
import fsspec
import torch

def to_cuda(x: torch.Tensor) -> torch.Tensor:
    if x is None:
        return None
    if torch.is_tensor(x):
        x = x.contiguous()
        if torch.cuda.is_available():
            x = x.cuda(non_blocking=True)
    return x


def get_cuda():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return use_cuda, device

def get_git_branch():
    return "local"

def get_commit_hash():
    return "local"


def get_experiment_folder_path(root_path, model_name):
    date_str = datetime.datetime.now().strftime("%B-%d-%Y_%I+%M%p")
    commit_hash = get_commit_hash()
    output_folder = os.path.join(root_path, model_name + "-" + date_str + "-" + commit_hash)
    return output_folder


def remove_experiment_folder(experiment_path):
    fs = fsspec.get_mapper(experiment_path).fs
    checkpoint_files = fs.glob(experiment_path + "/*.pth")
    if not checkpoint_files:
        if fs.exists(experiment_path):
            fs.rm(experiment_path, recursive=True)
            print(" ! Run is removed from {}".format(experiment_path))
    else:
        print(" ! Run is kept in {}".format(experiment_path))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_camel(text):
    text = text.capitalize()
    text = re.sub(r"(?!^)_([a-zA-Z])", lambda m: m.group(1).upper(), text)
    text = text.replace("Tts", "TTS")
    text = text.replace("vc", "VC")
    return text


def find_module(module_path: str, module_name: str) -> object:
    module_name = module_name.lower()
    module = importlib.import_module(module_path + "." + module_name)
    class_name = to_camel(module_name)
    return getattr(module, class_name)


def import_class(module_path: str) -> object:
    class_name = module_path.split(".")[-1]
    module_path = ".".join(module_path.split(".")[:-1])
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def get_import_path(obj: object) -> str:
    return ".".join([type(obj).__module__, type(obj).__name__])


def get_user_data_dir(appname):
    TTS_HOME = os.environ.get("TTS_HOME")
    XDG_DATA_HOME = os.environ.get("XDG_DATA_HOME")
    if TTS_HOME is not None:
        ans = Path(TTS_HOME).expanduser().resolve(strict=False)
    elif XDG_DATA_HOME is not None:
        ans = Path(XDG_DATA_HOME).expanduser().resolve(strict=False)
    elif sys.platform == "win32":
        import winreg  # pylint: disable=import-outside-toplevel

        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders"
        )
        dir_, _ = winreg.QueryValueEx(key, "Local AppData")
        ans = Path(dir_).resolve(strict=False)
    elif sys.platform == "darwin":
        ans = Path("~/Library/Application Support/").expanduser()
    else:
        ans = Path.home().joinpath(".local/share")
    return ans.joinpath(appname)


def set_init_dict(model_dict, checkpoint_state, c):
    for k, v in checkpoint_state.items():
        if k not in model_dict:
            print(" | > Layer missing in the model definition: {}".format(k))
    pretrained_dict = {k: v for k, v in checkpoint_state.items() if k in model_dict}
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if v.numel() == model_dict[k].numel()}
    if c.has("reinit_layers") and c.reinit_layers is not None:
        for reinit_layer_name in c.reinit_layers:
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if reinit_layer_name not in k}
    model_dict.update(pretrained_dict)
    print(" | > {} / {} layers are restored.".format(len(pretrained_dict), len(model_dict)))
    return model_dict


def format_aux_input(def_args: Dict, kwargs: Dict) -> Dict:
    kwargs = kwargs.copy()
    for name in def_args:
        if name not in kwargs or kwargs[name] is None:
            kwargs[name] = def_args[name]
    return kwargs


class KeepAverage:
    def __init__(self):
        self.avg_values = {}
        self.iters = {}

    def __getitem__(self, key):
        return self.avg_values[key]

    def items(self):
        return self.avg_values.items()

    def add_value(self, name, init_val=0, init_iter=0):
        self.avg_values[name] = init_val
        self.iters[name] = init_iter

    def update_value(self, name, value, weighted_avg=False):
        if name not in self.avg_values:
            self.add_value(name, init_val=value)
        else:
            if weighted_avg:
                self.avg_values[name] = 0.99 * self.avg_values[name] + 0.01 * value
                self.iters[name] += 1
            else:
                self.avg_values[name] = self.avg_values[name] * self.iters[name] + value
                self.iters[name] += 1
                self.avg_values[name] /= self.iters[name]

    def add_values(self, name_dict):
        for key, value in name_dict.items():
            self.add_value(key, init_val=value)

    def update_values(self, value_dict):
        for key, value in value_dict.items():
            self.update_value(key, value)


def get_timestamp():
    return datetime.now().strftime("%y%m%d-%H%M%S")


def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter("%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s", datefmt="%y-%m-%d %H:%M:%S")
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, phase + "_{}.log".format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode="w")
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)
import re
import unicodedata


def is_pytorch_at_least_2_4():
    try:
        import torch
        version = getattr(torch, "__version__", "0.0")
        version = version.split("+")[0]
        parts = version.split(".")
        major = int(parts[0]) if len(parts) > 0 and parts[0].isdigit() else 0
        minor_str = parts[1] if len(parts) > 1 else "0"
        minor_match = re.match(r"(\d+)", minor_str)
        minor = int(minor_match.group(1)) if minor_match else 0
        return (major, minor) >= (2, 4)
    except Exception:
        return False


def slugify(text):
    text = str(text)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    text = re.sub(r"[-\s]+", "_", text)
    return text


def warn_synthesize_config_deprecated(*args, **kwargs):
    return None


def warn_synthesize_speaker_id_deprecated(*args, **kwargs):
    return None


def optional_to_str(value):
    return None if value is None else str(value)


class ConsoleFormatter:
    def format(self, record):
        try:
            return record.getMessage()
        except Exception:
            return str(record)