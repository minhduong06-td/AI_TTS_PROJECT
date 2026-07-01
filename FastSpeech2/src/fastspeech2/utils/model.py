import os
import json
from pathlib import Path
import torch
from fastspeech2.models import hifigan
from fastspeech2.models import FastSpeech2, ScheduledOptim

PROJECT_ROOT = Path(__file__).resolve().parents[3]

def project_path(*parts):
    return PROJECT_ROOT.joinpath(*parts)

def get_model(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    model = FastSpeech2(preprocess_config, model_config).to(device)
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, args.restore_step
        )
        if args.restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def get_vocoder(config, device):
    name = config["vocoder"]["model"]
    speaker = config["vocoder"]["speaker"]
    if name != "HiFi-GAN":
        raise ValueError("Local CLI build supports only the bundled HiFi-GAN vocoder")

    with open(project_path("assets", "vocoders", "hifigan", "config.json"), "r", encoding="utf-8") as stream:
        hifigan_config = hifigan.AttrDict(json.load(stream))
    vocoder = hifigan.Generator(hifigan_config)

    checkpoints = {
        "LJSpeech": project_path("assets", "vocoders", "hifigan", "generator_LJSpeech.pth.tar"),
        "universal": project_path("assets", "vocoders", "hifigan", "generator_universal.pth.tar"),
    }
    if speaker not in checkpoints:
        raise ValueError(f"Unsupported HiFi-GAN speaker profile: {speaker}")
    ckpt = torch.load(checkpoints[speaker], map_location=device, weights_only=False)
    vocoder.load_state_dict(ckpt["generator"])
    vocoder.eval()
    vocoder.to(device)
    vocoder.remove_weight_norm()
    return vocoder


def vocoder_infer(mels, vocoder, model_config, preprocess_config, lengths=None):
    if model_config["vocoder"]["model"] != "HiFi-GAN":
        raise ValueError("Local CLI build supports only HiFi-GAN")

    with torch.no_grad():
        wavs = vocoder(mels.float()).squeeze(1)
    wavs = (
        wavs.cpu().numpy()
        * preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    ).astype("int16")
    wavs = [wav for wav in wavs]

    if lengths is not None:
        for index in range(len(wavs)):
            wavs[index] = wavs[index][: lengths[index]]
    return wavs
