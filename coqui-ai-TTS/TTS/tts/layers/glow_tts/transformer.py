import math
import torch
from torch import nn
from torch.nn import functional as F
from TTS.tts.layers.generic.normalization import LayerNorm, LayerNorm2
from TTS.tts.utils.helpers import convert_pad_shape


class RelativePositionMultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels,
        out_channels,
        num_heads,
        rel_attn_window_size=None,
        heads_share=True,
        dropout_p=0.0,
        input_length=None,
        proximal_bias=False,
        proximal_init=False,
    ):
        super().__init__()
        assert channels % num_heads == 0, " [!] channels should be divisible by num_heads."
        # class attributes
        self.channels = channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.rel_attn_window_size = rel_attn_window_size
        self.heads_share = heads_share
        self.input_length = input_length
        self.proximal_bias = proximal_bias
        self.dropout_p = dropout_p
        self.attn = None
        self.k_channels = channels // num_heads
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.dropout = nn.Dropout(dropout_p)
        if rel_attn_window_size is not None:
            n_heads_rel = 1 if heads_share else num_heads
            rel_stddev = self.k_channels**-0.5
            emb_rel_k = nn.Parameter(
                torch.randn(n_heads_rel, rel_attn_window_size * 2 + 1, self.k_channels) * rel_stddev
            )
            emb_rel_v = nn.Parameter(
                torch.randn(n_heads_rel, rel_attn_window_size * 2 + 1, self.k_channels) * rel_stddev
            )
            self.register_parameter("emb_rel_k", emb_rel_k)
            self.register_parameter("emb_rel_v", emb_rel_v)
        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.xavier_uniform_(self.conv_k.weight)
        if proximal_init:
            self.conv_k.weight.data.copy_(self.conv_q.weight.data)
            self.conv_k.bias.data.copy_(self.conv_q.bias.data)
        nn.init.xavier_uniform_(self.conv_v.weight)

    def forward(self, x, c, attn_mask=None):
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)
        x, self.attn = self.attention(q, k, v, mask=attn_mask)
        x = self.conv_o(x)
        return x

    def attention(self, query, key, value, mask=None):
        b, d, t_s, t_t = (*key.size(), query.size(2))
        query = query.view(b, self.num_heads, self.k_channels, t_t).transpose(2, 3)
        key = key.view(b, self.num_heads, self.k_channels, t_s).transpose(2, 3)
        value = value.view(b, self.num_heads, self.k_channels, t_s).transpose(2, 3)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.k_channels)
        if self.rel_attn_window_size is not None:
            assert t_s == t_t, "Relative attention is only available for self-attention."
            key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, t_s)
            rel_logits = self._matmul_with_relative_keys(query, key_relative_embeddings)
            rel_logits = self._relative_position_to_absolute_position(rel_logits)
            scores_local = rel_logits / math.sqrt(self.k_channels)
            scores = scores + scores_local
        if self.proximal_bias:
            assert t_s == t_t, "Proximal bias is only available for self-attention."
            scores = scores + self._attn_proximity_bias(t_s).to(device=scores.device, dtype=scores.dtype)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
            if self.input_length is not None:
                block_mask = torch.ones_like(scores).triu(-1 * self.input_length).tril(self.input_length)
                scores = scores * block_mask + -1e4 * (1 - block_mask)
        p_attn = F.softmax(scores, dim=-1) 
        p_attn = self.dropout(p_attn)
        # compute output
        output = torch.matmul(p_attn, value)
        if self.rel_attn_window_size is not None:
            relative_weights = self._absolute_position_to_relative_position(p_attn)
            value_relative_embeddings = self._get_relative_embeddings(self.emb_rel_v, t_s)
            output = output + self._matmul_with_relative_values(relative_weights, value_relative_embeddings)
        output = output.transpose(2, 3).contiguous().view(b, d, t_t) 
        return output, p_attn

    @staticmethod
    def _matmul_with_relative_values(p_attn, re):
        logits = torch.matmul(p_attn, re.unsqueeze(0))
        return logits

    @staticmethod
    def _matmul_with_relative_keys(query, re):
        logits = torch.matmul(query, re.unsqueeze(0).transpose(-2, -1))
        return logits

    def _get_relative_embeddings(self, relative_embeddings, length):
        pad_length = max(length - (self.rel_attn_window_size + 1), 0)
        slice_start_position = max((self.rel_attn_window_size + 1) - length, 0)
        slice_end_position = slice_start_position + 2 * length - 1
        if pad_length > 0:
            padded_relative_embeddings = F.pad(relative_embeddings, [0, 0, pad_length, pad_length, 0, 0])
        else:
            padded_relative_embeddings = relative_embeddings
        used_relative_embeddings = padded_relative_embeddings[:, slice_start_position:slice_end_position]
        return used_relative_embeddings

    @staticmethod
    def _relative_position_to_absolute_position(x):
        batch, heads, length, _ = x.size()
        x = F.pad(x, [0, 1, 0, 0, 0, 0, 0, 0])
        x_flat = x.view([batch, heads, length * 2 * length])
        x_flat = F.pad(x_flat, [0, length - 1, 0, 0, 0, 0])
        x_final = x_flat.view([batch, heads, length + 1, 2 * length - 1])[:, :, :length, length - 1 :]
        return x_final

    @staticmethod
    def _absolute_position_to_relative_position(x):
        batch, heads, length, _ = x.size()
        x = F.pad(x, [0, length - 1, 0, 0, 0, 0, 0, 0])
        x_flat = x.view([batch, heads, length**2 + length * (length - 1)])
        x_flat = F.pad(x_flat, [length, 0, 0, 0, 0, 0])
        x_final = x_flat.view([batch, heads, length, 2 * length])[:, :, :, 1:]
        return x_final

    @staticmethod
    def _attn_proximity_bias(length):
        r = torch.arange(length, dtype=torch.float32)
        diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
        diff = -torch.log1p(torch.abs(diff))
        return diff.unsqueeze(0).unsqueeze(0)


class FeedForwardNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, kernel_size, dropout_p=0.0, causal=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dropout_p = dropout_p

        if causal:
            self.padding = self._causal_padding
        else:
            self.padding = self._same_padding

        self.conv_1 = nn.Conv1d(in_channels, hidden_channels, kernel_size)
        self.conv_2 = nn.Conv1d(hidden_channels, out_channels, kernel_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, x_mask):
        x = self.conv_1(self.padding(x * x_mask))
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv_2(self.padding(x * x_mask))
        return x * x_mask

    def _causal_padding(self, x):
        if self.kernel_size == 1:
            return x
        pad_l = self.kernel_size - 1
        pad_r = 0
        padding = [[0, 0], [0, 0], [pad_l, pad_r]]
        x = F.pad(x, convert_pad_shape(padding))
        return x

    def _same_padding(self, x):
        if self.kernel_size == 1:
            return x
        pad_l = (self.kernel_size - 1) // 2
        pad_r = self.kernel_size // 2
        padding = [[0, 0], [0, 0], [pad_l, pad_r]]
        x = F.pad(x, convert_pad_shape(padding))
        return x


class RelativePositionTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        hidden_channels_ffn: int,
        num_heads: int,
        num_layers: int,
        kernel_size=1,
        dropout_p=0.0,
        rel_attn_window_size: int = None,
        input_length: int = None,
        layer_norm_type: str = "1",
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.hidden_channels_ffn = hidden_channels_ffn
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.dropout_p = dropout_p
        self.rel_attn_window_size = rel_attn_window_size

        self.dropout = nn.Dropout(dropout_p)
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()

        for idx in range(self.num_layers):
            self.attn_layers.append(
                RelativePositionMultiHeadAttention(
                    hidden_channels if idx != 0 else in_channels,
                    hidden_channels,
                    num_heads,
                    rel_attn_window_size=rel_attn_window_size,
                    dropout_p=dropout_p,
                    input_length=input_length,
                )
            )
            if layer_norm_type == "1":
                self.norm_layers_1.append(LayerNorm(hidden_channels))
            elif layer_norm_type == "2":
                self.norm_layers_1.append(LayerNorm2(hidden_channels))
            else:
                raise ValueError("Unknown layer norm type")

            if hidden_channels != out_channels and (idx + 1) == self.num_layers:
                self.proj = nn.Conv1d(hidden_channels, out_channels, 1)

            self.ffn_layers.append(
                FeedForwardNetwork(
                    hidden_channels,
                    hidden_channels if (idx + 1) != self.num_layers else out_channels,
                    hidden_channels_ffn,
                    kernel_size,
                    dropout_p=dropout_p,
                )
            )

            if layer_norm_type == "1":
                self.norm_layers_2.append(LayerNorm(hidden_channels if (idx + 1) != self.num_layers else out_channels))
            elif layer_norm_type == "2":
                self.norm_layers_2.append(LayerNorm2(hidden_channels if (idx + 1) != self.num_layers else out_channels))
            else:
                raise ValueError("Unknown layer norm type")

    def forward(self, x, x_mask):
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        for i in range(self.num_layers):
            x = x * x_mask
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.dropout(y)
            x = self.norm_layers_1[i](x + y)

            y = self.ffn_layers[i](x, x_mask)
            y = self.dropout(y)

            if (i + 1) == self.num_layers and hasattr(self, "proj"):
                x = self.proj(x)

            x = self.norm_layers_2[i](x + y)
        x = x * x_mask
        return x
