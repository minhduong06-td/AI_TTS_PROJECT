import torch
from torch import nn
from ..generic.normalization import LayerNorm
class DurationPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size, dropout_p, cond_channels=None, language_emb_dim=None):
        super().__init__()
        if language_emb_dim:
            in_channels += language_emb_dim
        self.in_channels = in_channels
        self.filter_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dropout_p = dropout_p
        self.drop = nn.Dropout(dropout_p)
        self.conv_1 = nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=kernel_size // 2)
        self.norm_1 = LayerNorm(hidden_channels)
        self.conv_2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2)
        self.norm_2 = LayerNorm(hidden_channels)
        self.proj = nn.Conv1d(hidden_channels, 1, 1)
        if cond_channels is not None and cond_channels != 0:
            self.cond = nn.Conv1d(cond_channels, in_channels, 1)

        if language_emb_dim != 0 and language_emb_dim is not None:
            self.cond_lang = nn.Conv1d(language_emb_dim, in_channels, 1)

    def forward(self, x, x_mask, g=None, lang_emb=None):
        if g is not None:
            x = x + self.cond(g)

        if lang_emb is not None:
            x = x + self.cond_lang(lang_emb)

        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask
