import math
import torch
from torch import nn
from TTS.tts.layers.generic.wavenet import WN
from TTS.tts.layers.glow_tts.transformer import RelativePositionTransformer
from TTS.tts.utils.helpers import sequence_mask

LRELU_SLOPE = 0.1


class TextEncoder(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        out_channels: int,
        hidden_channels: int,
        hidden_channels_ffn: int,
        num_heads: int,
        num_layers: int,
        kernel_size: int,
        dropout_p: float,
        language_emb_dim: int = None,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        self.emb = nn.Embedding(n_vocab, hidden_channels)

        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

        if language_emb_dim:
            hidden_channels += language_emb_dim

        self.encoder = RelativePositionTransformer(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            hidden_channels=hidden_channels,
            hidden_channels_ffn=hidden_channels_ffn,
            num_heads=num_heads,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout_p=dropout_p,
            layer_norm_type="2",
            rel_attn_window_size=4,
        )

        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, lang_emb=None):
        assert x.shape[0] == x_lengths.shape[0]
        x = self.emb(x) * math.sqrt(self.hidden_channels)
        if lang_emb is not None:
            x = torch.cat((x, lang_emb.transpose(2, 1).expand(x.size(0), x.size(1), -1)), dim=-1)
        x = torch.transpose(x, 1, -1)  
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)  

        x = self.encoder(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs, x_mask


class ResidualCouplingBlock(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        num_layers,
        dropout_p=0,
        cond_channels=0,
        mean_only=False,
    ):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.half_channels = channels // 2
        self.mean_only = mean_only
        # input layer
        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        # coupling layers
        self.enc = WN(
            hidden_channels,
            hidden_channels,
            kernel_size,
            dilation_rate,
            num_layers,
            dropout_p=dropout_p,
            c_in_channels=cond_channels,
        )
        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, log_scale = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            log_scale = torch.zeros_like(m)

        if not reverse:
            x1 = m + x1 * torch.exp(log_scale) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(log_scale, [1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-log_scale) * x_mask
            x = torch.cat([x0, x1], 1)
            return x


class ResidualCouplingBlocks(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        num_layers: int,
        num_flows=4,
        cond_channels=0,
    ):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.num_layers = num_layers
        self.num_flows = num_flows
        self.cond_channels = cond_channels

        self.flows = nn.ModuleList()
        for _ in range(num_flows):
            self.flows.append(
                ResidualCouplingBlock(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    num_layers,
                    cond_channels=cond_channels,
                    mean_only=True,
                )
            )

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
                x = torch.flip(x, [1])
        else:
            for flow in reversed(self.flows):
                x = torch.flip(x, [1])
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


class PosteriorEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        num_layers: int,
        cond_channels=0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.num_layers = num_layers
        self.cond_channels = cond_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WN(
            hidden_channels, hidden_channels, kernel_size, dilation_rate, num_layers, c_in_channels=cond_channels
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None, tau=1.0):
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        mean, log_scale = torch.split(stats, self.out_channels, dim=1)
        z = (mean + torch.randn_like(mean) * tau * torch.exp(log_scale)) * x_mask
        return z, mean, log_scale, x_mask
