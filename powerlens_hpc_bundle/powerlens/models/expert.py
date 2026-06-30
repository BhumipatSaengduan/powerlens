"""
DRL-STFN Expert Model (Stage 2)
================================
Per-category disaggregation model — รับ feature sequence แล้วทำนาย
appliance state (status / power / current) สำหรับ category นั้นๆ

Architecture flow:
    Input (B, T=60, F=16)
        ↓
    SFTN Feature Extraction        → (B, T, 64)
        ↓
    GRU-BiLSTM (hidden=128)        → (B, T, 256)
        ↓
    Multi-Head Self-Attention      → (B, T, 256)
        ↓
    Pool (last + mean)             → (B, 512)
        ↓
    ┌─────────┬─────────┬─────────┐
    Status     Power     Current
    (sigmoid)  (linear)  (linear)
    
Pool strategy:
    ใช้ทั้ง last hidden state (current state) และ mean (overall context)
    concat กัน — ให้ output heads เห็นทั้งจุดล่าสุดและภาพรวม window

Usage:
    >>> from powerlens.models.expert import DRLSTFNExpert
    >>> from powerlens.models.config import EXPERT_CONFIG
    >>> model = DRLSTFNExpert(category="AC", config=EXPERT_CONFIG)
    >>> x = torch.randn(8, 60, 16)
    >>> status, power, current = model(x)
    >>> # status: (8,1) sigmoid, power: (8,1) linear, current: (8,1) linear
"""
from typing import Tuple
import torch
import torch.nn as nn

from .sftn import SFTNFeatureExtractor
from .config import ExpertConfig, EXPERT_CONFIG


class DRLSTFNExpert(nn.Module):
    """
    Single-category expert model สำหรับ load disaggregation.
    
    1 instance ต่อ 1 category (Plug / Light / AC / Water Heater)
    Architecture เหมือนกันทุก category — ต่างกันแค่ weights หลัง train
    
    Args:
        category: category name (สำหรับ logging / ONNX export naming)
        config:   ExpertConfig instance — hyperparameters
    """
    def __init__(self, category: str, config: ExpertConfig = EXPERT_CONFIG):
        super().__init__()
        self.category = category
        self.config = config

        # Stage 1: SFTN Feature Extraction
        self.sftn = SFTNFeatureExtractor(
            n_features=config.n_features,
            out_channels=config.sftn_channels,
            kernel_size=config.sftn_kernel,
            n_blocks=2,
        )

        # Stage 2: GRU-BiLSTM Backbone
        # Note: ใช้ GRU ก่อน BiLSTM ตาม design — GRU จับ short-term, LSTM จับ long-term
        self.gru = nn.GRU(
            input_size=config.sftn_channels,
            hidden_size=config.gru_hidden,
            num_layers=config.gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.gru_dropout if config.gru_layers > 1 else 0.0,
        )
        # Output dim of bidirectional GRU
        gru_out_dim = config.gru_hidden * 2  # = 256

        # Stage 3: Multi-Head Self-Attention
        self.attn = nn.MultiheadAttention(
            embed_dim=gru_out_dim,
            num_heads=config.attn_heads,
            dropout=config.attn_dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(gru_out_dim)

        # Pooling: concat last + mean → 2× dim
        pooled_dim = gru_out_dim * 2  # = 512

        # Stage 4: Output heads (3 parallel heads)
        # แต่ละ head: pooled → FC(hidden) → ReLU → FC(1)
        self.head_status = self._build_head(pooled_dim, config.head_hidden)
        self.head_power = self._build_head(pooled_dim, config.head_hidden)
        self.head_current = self._build_head(pooled_dim, config.head_hidden)

    @staticmethod
    def _build_head(in_dim: int, hidden: int) -> nn.Sequential:
        """Helper: build 2-layer MLP head."""
        return nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: (B, T, F) — batch, timesteps, features
                expected: T=60, F=16
        
        Returns:
            status:  (B, 1) — sigmoid output (0-1, on/off probability)
            power:   (B, 1) — linear output (kW, can be 0+)
            current: (B, 1) — linear output (A, can be 0+)
        """
        # SFTN Feature Extraction: (B,T,F) → (B,T,C_sftn)
        h = self.sftn(x)

        # GRU-BiLSTM: (B,T,C_sftn) → (B,T,2*hidden)
        h, _ = self.gru(h)

        # Multi-Head Self-Attention with residual + norm
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        h = self.attn_norm(h + attn_out)

        # Pooling: combine last timestep + mean over time
        last = h[:, -1, :]              # (B, 2*hidden)
        mean = h.mean(dim=1)            # (B, 2*hidden)
        pooled = torch.cat([last, mean], dim=-1)  # (B, 4*hidden)

        # Output heads
        status_logit = self.head_status(pooled)
        status = torch.sigmoid(status_logit)
        power = self.head_power(pooled)
        # Power และ current ไม่ติดลบ → clamp to ≥ 0
        # ใช้ ReLU แทน softplus เพราะ ONNX export ดีกว่า
        power = torch.relu(power)
        current = torch.relu(self.head_current(pooled))

        return status, power, current

    def count_parameters(self) -> int:
        """Total trainable parameters — สำหรับ debug / report."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MultiCategoryExperts(nn.ModuleDict):
    """
    Container holding 4 expert models — one per category.
    
    Usage:
        >>> experts = MultiCategoryExperts(EXPERT_CONFIG)
        >>> # Forward through specific category (เลือกโดย DRL Router):
        >>> status, power, current = experts["AC"](x)
        >>> # หรือ all categories:
        >>> outputs = experts.forward_all(x)
        >>> # outputs = {"Plug": (s,p,c), "Light": (s,p,c), ...}
    """
    def __init__(self, config: ExpertConfig = EXPERT_CONFIG):
        super().__init__()
        self.config = config
        for cat in config.categories:
            self[cat] = DRLSTFNExpert(category=cat, config=config)

    def forward_all(self, x: torch.Tensor) -> dict:
        """Forward through all 4 experts — สำหรับ training stage."""
        return {cat: self[cat](x) for cat in self.config.categories}

    def forward(self, x: torch.Tensor, category: str):
        """Forward through one specific expert — สำหรับ inference หลัง router."""
        return self[category](x)
