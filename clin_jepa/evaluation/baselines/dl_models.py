"""DL baseline models.

Four model classes sharing a unified encoder -> multi-head architecture:
- LSTMEncoder: 2-layer LSTM
- TransformerEncoder: from-scratch Transformer
- GRUDEncoder: GRU with mask + time-since-last-obs + decay gates
- TCNEncoder: dilated causal 1D convolutions

All models predict (B, n_outputs) where n_outputs = n_vars × max_h ×
n_aggregators for continuous, or n_outcomes for binary. Static features are
z-scored before concatenation; sequences are length C after imputation, so no
padding mask is needed. Outputs are un-normalized in the training script.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============= Encoders =============

class LSTMEncoder(nn.Module):
    """2-layer LSTM encoder. Returns last hidden state."""
    def __init__(self, input_dim: int, hidden_size: int = 256, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.out_dim = hidden_size

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq: (B, C, V_dyn)
        out, (h_n, _) = self.lstm(x_seq)
        # Use last hidden state of last layer
        return h_n[-1]  # (B, hidden_size)


class TransformerEncoderModule(nn.Module):
    """Small Transformer encoder. Returns mean-pooled output."""
    def __init__(self, input_dim: int, d_model: int = 256, nhead: int = 8,
                 num_layers: int = 3, dropout: float = 0.2, max_len: int = 256):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_dim = d_model

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        B, C, V = x_seq.shape
        h = self.input_proj(x_seq) + self.pos_embed[:, :C, :]
        h = self.encoder(h)  # (B, C, d_model)
        return h.mean(dim=1)  # (B, d_model)


class GRUDCell(nn.Module):
    """GRU-D cell — handles missingness via decay gates.

    Inputs at each step: x (B, V), m (B, V) mask, delta (B, V) time-since-last-obs.
    Maintains running last-observed values internally via x_last_obs state.
    W_gamma_x is a per-variable diagonal decay (a (V,) vector).
    """
    def __init__(self, input_dim: int, hidden_size: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size

        self.w_gamma_x = nn.Parameter(torch.zeros(input_dim))  # diagonal vector
        self.b_gamma_x = nn.Parameter(torch.zeros(input_dim))
        self.W_gamma_h = nn.Parameter(torch.zeros(input_dim, hidden_size))
        self.b_gamma_h = nn.Parameter(torch.zeros(hidden_size))

        # GRU weights (concatenated [x_imp, m, h_decay] → 3 gates: r, z, h_tilde)
        gate_input = input_dim * 2 + hidden_size
        self.W_z = nn.Linear(gate_input, hidden_size)
        self.W_r = nn.Linear(gate_input, hidden_size)
        self.W_h = nn.Linear(gate_input, hidden_size)

        # Init: GRU linears via xavier; decay params kept at zero (no decay initially)
        nn.init.xavier_uniform_(self.W_z.weight)
        nn.init.xavier_uniform_(self.W_r.weight)
        nn.init.xavier_uniform_(self.W_h.weight)
        nn.init.zeros_(self.W_z.bias)
        nn.init.zeros_(self.W_r.bias)
        nn.init.zeros_(self.W_h.bias)
        # Optional small init for W_gamma_h to break symmetry (zeros also fine)
        nn.init.xavier_uniform_(self.W_gamma_h)

    def forward(self, x: torch.Tensor, m: torch.Tensor, delta: torch.Tensor,
                h_prev: torch.Tensor, x_last_obs: torch.Tensor, x_mean: torch.Tensor) -> tuple:
        # gamma_x: (B, V) per-variable elementwise
        gamma_x = torch.exp(-F.relu(delta * self.w_gamma_x + self.b_gamma_x))
        # γ_h: (B, hidden_size) full V→H matrix
        gamma_h = torch.exp(-F.relu(delta @ self.W_gamma_h + self.b_gamma_h))

        # Imputed x: m * x + (1-m) * (gamma_x * x_last_obs + (1-gamma_x) * x_mean)
        x_imp = m * x + (1 - m) * (gamma_x * x_last_obs + (1 - gamma_x) * x_mean)

        # Hidden decay
        h_decay = gamma_h * h_prev

        # Standard GRU update with [x_imp, m] as input
        gate_in = torch.cat([x_imp, m, h_decay], dim=-1)
        z = torch.sigmoid(self.W_z(gate_in))
        r = torch.sigmoid(self.W_r(gate_in))
        gate_in_r = torch.cat([x_imp, m, r * h_decay], dim=-1)
        h_tilde = torch.tanh(self.W_h(gate_in_r))
        h_new = (1 - z) * h_decay + z * h_tilde

        # Update last observed: where m==1, use x; otherwise carry forward
        x_last_obs_new = m * x + (1 - m) * x_last_obs

        return h_new, x_last_obs_new


class GRUDEncoder(nn.Module):
    """GRU-D encoder over (x_seq, m_seq, delta_seq), each (B, C, V): x_seq is
    z-scored + imputed, m_seq is the observed mask (1=observed), delta_seq is
    hours since last observation.
    """
    def __init__(self, input_dim: int, hidden_size: int = 256, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell = GRUDCell(input_dim, hidden_size)  # only 1 GRU-D layer
        if num_layers > 1:
            self.extra_gru = nn.GRU(
                hidden_size, hidden_size,
                num_layers=num_layers - 1,
                batch_first=True, dropout=dropout if num_layers - 1 > 1 else 0.0,
            )
        else:
            self.extra_gru = None
        self.dropout = nn.Dropout(dropout)
        self.out_dim = hidden_size

    def forward(self, x_seq: torch.Tensor, m_seq: torch.Tensor,
                delta_seq: torch.Tensor) -> torch.Tensor:
        # x_seq, m_seq, delta_seq: (B, C, V)
        B, C, V = x_seq.shape
        device = x_seq.device

        # x_mean: 0 in normalized space (input is z-scored upstream)
        x_mean = torch.zeros(V, device=device)

        h = torch.zeros(B, self.hidden_size, device=device)
        x_last_obs = torch.zeros(B, V, device=device)
        outputs = []
        for t in range(C):
            x_t = x_seq[:, t, :]
            m_t = m_seq[:, t, :]
            d_t = delta_seq[:, t, :]
            h, x_last_obs = self.cell(x_t, m_t, d_t, h, x_last_obs, x_mean.unsqueeze(0).expand(B, -1))
            outputs.append(h)
        h_seq = torch.stack(outputs, dim=1)  # (B, C, H)

        if self.extra_gru is not None:
            h_seq, _ = self.extra_gru(h_seq)

        return self.dropout(h_seq[:, -1, :])


class _Chomp1d(nn.Module):
    """Remove right-padding from causal conv output."""
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x


class _TemporalBlock(nn.Module):
    def __init__(self, n_in: int, n_out: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(n_in, n_out, kernel_size, padding=padding, dilation=dilation)
        )
        self.chomp1 = _Chomp1d(padding)
        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(n_out, n_out, kernel_size, padding=padding, dilation=dilation)
        )
        self.chomp2 = _Chomp1d(padding)
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = F.relu(self.chomp1(self.conv1(x)))
        out = self.dropout(out)
        out = F.relu(self.chomp2(self.conv2(out)))
        out = self.dropout(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNEncoder(nn.Module):
    """Temporal Convolutional Network. Returns last-position activation."""
    def __init__(self, input_dim: int, hidden_size: int = 128, num_layers: int = 4,
                 kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_ch = input_dim if i == 0 else hidden_size
            dilation = 2 ** i
            layers.append(_TemporalBlock(in_ch, hidden_size, kernel_size, dilation, dropout))
        self.network = nn.Sequential(*layers)
        self.out_dim = hidden_size

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq: (B, C, V) → transpose to (B, V, C) for conv1d
        h = x_seq.transpose(1, 2)
        h = self.network(h)  # (B, hidden, C)
        return h[:, :, -1]  # take last position


# ============= Heads =============

class MultiOutputHead(nn.Module):
    """Generic multi-output head: encoder embedding + statics → many regressors.

    For continuous: n_outputs = n_vars × max_h × n_aggregators (e.g., 27 × 60 × 3 = 4860).
    For binary: n_outputs = n_outcomes (e.g., 8).

    Static features are z-score normalized before this layer.
    """
    def __init__(self, encoder_dim: int, static_dim: int, n_outputs: int,
                 hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        in_dim = encoder_dim + static_dim
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_outputs),
        )

    def forward(self, h_enc: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h_enc, x_static], dim=-1)
        return self.head(x)


# ============= Convenience model wrappers =============

class BaselineDL(nn.Module):
    """Unified baseline: encoder + multi-output head."""
    def __init__(self, encoder: nn.Module, static_dim: int, n_outputs: int,
                 head_hidden: int = 256, dropout: float = 0.2,
                 mode: str = 'plain'):
        """mode: 'plain' for LSTM/Transformer/TCN ; 'grud' for GRU-D (which needs mask+delta)."""
        super().__init__()
        self.encoder = encoder
        self.head = MultiOutputHead(encoder.out_dim, static_dim, n_outputs, head_hidden, dropout)
        self.mode = mode

    def forward(self, x_seq: torch.Tensor, x_static: torch.Tensor,
                m_seq: torch.Tensor = None, delta_seq: torch.Tensor = None) -> torch.Tensor:
        if self.mode == 'grud':
            h_enc = self.encoder(x_seq, m_seq, delta_seq)
        else:
            h_enc = self.encoder(x_seq)
        return self.head(h_enc, x_static)


def make_model(model_name: str, input_dim: int, static_dim: int, n_outputs: int,
               hidden_size: int = 256, num_layers: int = 2, dropout: float = 0.2,
               head_hidden: int = 256) -> BaselineDL:
    """Factory: model_name ∈ {'lstm', 'transformer', 'grud', 'tcn'}."""
    if model_name == 'lstm':
        enc = LSTMEncoder(input_dim, hidden_size, num_layers, dropout)
        mode = 'plain'
    elif model_name == 'transformer':
        # Use d_model=hidden_size, num_layers=num_layers
        enc = TransformerEncoderModule(input_dim, hidden_size, nhead=8, num_layers=num_layers, dropout=dropout)
        mode = 'plain'
    elif model_name == 'grud':
        enc = GRUDEncoder(input_dim, hidden_size, num_layers=1, dropout=dropout)
        mode = 'grud'
    elif model_name == 'tcn':
        enc = TCNEncoder(input_dim, hidden_size, num_layers=num_layers + 2, kernel_size=3, dropout=dropout)
        mode = 'plain'
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    return BaselineDL(enc, static_dim, n_outputs, head_hidden, dropout, mode=mode)
