import torch
from torch import nn
from torch.nn.modules.conv import Conv1d
from TTS.vocoder.models.hifigan_discriminator import LRELU_SLOPE, DiscriminatorP

class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super().__init__()
        norm_f = nn.utils.spectral_norm if use_spectral_norm else nn.utils.parametrizations.weight_norm
        self.convs = nn.ModuleList(
            [
                norm_f(Conv1d(1, 16, 15, 1, padding=7)),
                norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
                norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
                norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
                norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        feat = []
        for l in self.convs:
            x = l(x)
            x = torch.nn.functional.leaky_relu(x, LRELU_SLOPE)
            feat.append(x)
        x = self.conv_post(x)
        feat.append(x)
        x = torch.flatten(x, 1, -1)
        return x, feat


class VitsDiscriminator(nn.Module):
    def __init__(self, periods=(2, 3, 5, 7, 11), use_spectral_norm=False):
        super().__init__()
        self.nets = nn.ModuleList()
        self.nets.append(DiscriminatorS(use_spectral_norm=use_spectral_norm))
        self.nets.extend([DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods])

    def forward(self, x, x_hat=None):
        x_scores = []
        x_hat_scores = [] if x_hat is not None else None
        x_feats = []
        x_hat_feats = [] if x_hat is not None else None
        for net in self.nets:
            x_score, x_feat = net(x)
            x_scores.append(x_score)
            x_feats.append(x_feat)
            if x_hat is not None:
                x_hat_score, x_hat_feat = net(x_hat)
                x_hat_scores.append(x_hat_score)
                x_hat_feats.append(x_hat_feat)
        return x_scores, x_feats, x_hat_scores, x_hat_feats
