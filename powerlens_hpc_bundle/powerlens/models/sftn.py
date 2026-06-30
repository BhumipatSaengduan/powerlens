"""
SFTN Feature Extraction
=======================
Sequential-gated Temporal Feature Network — first stage of DRL-STFN.

Purpose:
    แยก temporal features จาก input sequence ก่อนส่งให้ GRU-BiLSTM
    ใช้ Conv1D + gating mechanism เพื่อ:
    - จับ local temporal pattern (Conv1D)
    - กรอง feature ที่ไม่สำคัญออก (gating)
    - ลด noise ของ raw features ก่อน feed sequential model

Architecture:
    Input (B, T, F) → transpose → Conv1D → Gating → transpose → Output (B, T, C)
    
    Gating: y = conv_value(x) ⊙ sigmoid(conv_gate(x))
    คล้าย GLU (Gated Linear Unit) แต่ใช้ Conv1D
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalGatingBlock(nn.Module):
    """
    Single Conv1D block with gating mechanism.
    
    y = Conv_value(x) ⊙ σ(Conv_gate(x))
    
    Args:
        in_channels:  input feature dim
        out_channels: output feature dim
        kernel_size:  temporal kernel size (default 3 = capture ±1 timestep)
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2  # 'same' padding

        self.conv_value = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.conv_gate = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, T) — note: channels-first for Conv1D
        Returns:
            (B, C_out, T)
        """
        value = self.conv_value(x)
        gate = torch.sigmoid(self.conv_gate(x))
        out = value * gate
        out = self.norm(out)
        out = F.relu(out)
        return out


class SFTNFeatureExtractor(nn.Module):
    """
    Full SFTN Feature Extraction stack.
    
    Stack of TemporalGatingBlock layers — กรอง + transform features
    ก่อนเข้า sequential model.
    
    Args:
        n_features:   input feature dim (e.g. 16)
        out_channels: output feature dim (e.g. 64)
        kernel_size:  Conv1D kernel size
        n_blocks:     number of gating blocks (default 2 = enough for temporal smoothing)
    """
    def __init__(
        self,
        n_features: int = 16,
        out_channels: int = 64,
        kernel_size: int = 3,
        n_blocks: int = 2,
    ):
        super().__init__()
        self.n_features = n_features
        self.out_channels = out_channels

        layers = []
        in_ch = n_features
        for i in range(n_blocks):
            layers.append(TemporalGatingBlock(in_ch, out_channels, kernel_size))
            in_ch = out_channels  # subsequent blocks use out_channels
        self.blocks = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, F) — batch, timesteps, features (standard sequence format)
        Returns:
            (B, T, C) — same time dim, transformed feature dim
        """
        # (B, T, F) → (B, F, T) for Conv1D
        x = x.transpose(1, 2)
        x = self.blocks(x)
        # (B, C, T) → (B, T, C) for downstream sequential model
        x = x.transpose(1, 2)
        return x
