from typing import Dict
import numpy as np
import torch
from matplotlib import pyplot as plt

from TTS.tts.utils.visual import plot_spectrogram
from TTS.utils.audio import AudioProcessor


def interpolate_vocoder_input(scale_factor, spec):
    print(" > before interpolation :", spec.shape)
    spec = torch.tensor(spec).unsqueeze(0).unsqueeze(0)  # pylint: disable=not-callable
    spec = torch.nn.functional.interpolate(
        spec, scale_factor=scale_factor, recompute_scale_factor=True, mode="bilinear", align_corners=False
    ).squeeze(0)
    print(" > after interpolation :", spec.shape)
    return spec


def plot_results(y_hat: torch.tensor, y: torch.tensor, ap: AudioProcessor, name_prefix: str = None) -> Dict:
    if name_prefix is None:
        name_prefix = ""

    y_hat = y_hat[0].squeeze().detach().cpu().numpy()
    y = y[0].squeeze().detach().cpu().numpy()

    spec_fake = ap.melspectrogram(y_hat).T
    spec_real = ap.melspectrogram(y).T
    spec_diff = np.abs(spec_fake - spec_real)

    fig_wave = plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(y)
    plt.title("groundtruth speech")
    plt.subplot(2, 1, 2)
    plt.plot(y_hat)
    plt.title("generated speech")
    plt.tight_layout()
    plt.close()

    figures = {
        name_prefix + "spectrogram/fake": plot_spectrogram(spec_fake),
        name_prefix + "spectrogram/real": plot_spectrogram(spec_real),
        name_prefix + "spectrogram/diff": plot_spectrogram(spec_diff),
        name_prefix + "speech_comparison": fig_wave,
    }
    return figures
