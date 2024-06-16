"""CPC model: convolutional encoder, auto-regressive component and 1-layer Transofmer predictor."""

import torch
import torch.nn.functional as F  # noqa: N812
from huggingface_hub import PyTorchModelHubMixin
from torch import Tensor, nn

from .config import CONFIG
from .rotary import apply_rotary_emb, precompute_freqs_cis
from .utils import assert_compatibility

__all__ = ["CPC"]


class ConvLayerBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.layer_norm = nn.LayerNorm(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.layer_norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        return F.relu(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, din: int, dout: int, num_heads: int, dropout: float, block_size: int) -> None:
        super().__init__()
        if dout % num_heads != 0:
            raise ValueError(str(dout % num_heads))
        self.num_heads = num_heads
        self.head_dim = din // num_heads
        self.dout = dout
        self.dropout = dropout
        self.qkv = nn.Linear(din, dout * 3, bias=False)
        self.proj = nn.Linear(din, dout, bias=False)
        self.register_buffer("freqs_cis", precompute_freqs_cis(block_size, self.head_dim))

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.shape
        q, k, v = self.qkv(x).split([self.dout, self.dout, self.dout], dim=-1)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        q = apply_rotary_emb(q, self.freqs_cis)
        k = apply_rotary_emb(k, self.freqs_cis)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        dropout_p = 0.0 if not self.training else self.dropout
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dout)
        return self.proj(y)


class Predictor(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_predicts: int,
        seq_len: int,
        dropout: float,
        fully_connected_dim: float,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_predicts = num_predicts

        self.fully_connected = nn.Sequential(
            nn.Linear(hidden_size, fully_connected_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fully_connected_dim, hidden_size * num_predicts),
        )
        self.multihead = MultiHeadAttention(hidden_size, hidden_size, num_heads, dropout, seq_len)
        self.last_linear = nn.Linear(hidden_size, hidden_size)
        self.ln_multihead = nn.LayerNorm(hidden_size)
        self.ln_fully_connected = nn.LayerNorm(hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        y: Tensor = self.ln_multihead(x + self.multihead(x))
        b, s, _ = y.size()
        x = self.fully_connected(y).view(b, s, self.num_predicts, self.hidden_size)
        y = y.view(b, s, 1, self.hidden_size).expand(b, s, self.num_predicts, self.hidden_size)
        return self.ln_fully_connected(self.last_linear(x + y))


class CPC(nn.Module, PyTorchModelHubMixin):
    """CPC model: convolutional block, LSTM and 1-layer Transformer predictor."""

    def __init__(
        self,
        hidden_size: int = CONFIG.hidden_size,
        window_size: int = CONFIG.window_size,
        num_predicts: int = CONFIG.num_predicts,
        num_lstm_layers: int = CONFIG.num_lstm_layers,
        fully_connected_dim: int = CONFIG.fully_connected_dim,
        num_heads: int = CONFIG.num_heads,
        dropout: float = CONFIG.dropout,
    ) -> None:
        super().__init__()
        assert_compatibility(window_size, num_predicts)
        self.window_size = window_size
        self.downsampling_factor = 160
        self.encoder = nn.Sequential(
            ConvLayerBlock(1, hidden_size, 10, 5, 3),
            ConvLayerBlock(hidden_size, hidden_size, 8, 4, 2),
            ConvLayerBlock(hidden_size, hidden_size, 4, 2, 1),
            ConvLayerBlock(hidden_size, hidden_size, 4, 2, 1),
            ConvLayerBlock(hidden_size, hidden_size, 4, 2, 1),
        )
        self.auto_regressive = nn.LSTM(
            hidden_size,
            hidden_size,
            num_lstm_layers,
            batch_first=True,
        )
        self.predictor = Predictor(
            hidden_size,
            num_predicts,
            window_size // self.downsampling_factor - num_predicts,
            dropout,
            fully_connected_dim,
            num_heads,
        )

    def forward(self, past: Tensor, future: Tensor) -> tuple[Tensor, Tensor]:
        if not (past.shape[2] == future.shape[2] == self.window_size):
            lengths = (past.shape[2], future.shape[2], self.window_size)
            raise ValueError(str(lengths))
        latent = self.encoder(torch.cat((past, future), dim=0)).permute(0, 2, 1)
        latent_past, latent_future = torch.tensor_split(latent, 2, dim=0)
        contextual_past, _ = self.auto_regressive(latent_past)
        predictions = self.predictor(contextual_past[:, : -self.predictor.num_predicts])
        return predictions, latent_future

    def extract_features(self, x: Tensor) -> Tensor:
        latent = self.encoder(x).permute(0, 2, 1)
        contextual, _ = self.auto_regressive(latent)
        return contextual
