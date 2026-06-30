"""
DRL Router (Stage 1) — Branching DQN
=====================================
Multi-label category selector — รับ feature sequence แล้วตัดสินใจว่า
appliance category ไหนกำลัง active (เปิด) ใน window นี้

Why Branching DQN (not vanilla DQN)?
    Multi-action problem (categories เปิดพร้อมกันได้):
    - Vanilla DQN: 2^N actions → combinatorial explosion เมื่อ N โต
    - Branching DQN: N heads × 2 Q-values → linear scale, independent decisions
    
    Each head ตัดสินใจ binary (off/on) ของ category นั้นๆ — เหมือน reality
    ที่ AC เปิดไม่เกี่ยวกับ Light เปิด

Reference:
    Tavakoli et al., "Action Branching Architectures for Deep RL" (AAAI 2018)

Architecture flow:
    Input (B, T=60, F=16)
         ↓
    SFTN Feature Extraction (light, 1 block)   → (B, T, 32)
         ↓
    GRU (single direction, hidden=64)          → (B, T, 64)
         ↓
    Mean pool over time                        → (B, 64)
         ↓
    Shared FC trunk                            → (B, 128)
         ↓
    ┌─────────┬─────────┬─────────┬──────────────┐
    Q_Plug   Q_Light    Q_AC    Q_WaterHeater
    (B, 2)   (B, 2)     (B, 2)    (B, 2)         ← branching heads
         ↓
    Stack → Q-values: (B, N=4, 2)
         ↓
    argmax over last dim → action mask: (B, 4) ∈ {0, 1}

Inference output:
    active_categories = [cat for cat, active in zip(categories, mask) if active]
    → ส่ง input ไปให้ experts ที่ active เท่านั้น
"""
from typing import Dict, List, Tuple
import torch
import torch.nn as nn

from .sftn import SFTNFeatureExtractor
from .config import RouterConfig, ROUTER_CONFIG


class BranchingDQNRouter(nn.Module):
    """
    Branching DQN router — Stage 1 ของ DRL-STFN.
    
    เลือก subset ของ categories ที่ active โดยใช้ N independent binary heads
    
    Args:
        config: RouterConfig instance
    """
    def __init__(self, config: RouterConfig = ROUTER_CONFIG):
        super().__init__()
        self.config = config
        self.categories = config.categories
        self.n_heads = config.n_heads

        # Stage 1: SFTN Feature Extraction (lighter than Expert)
        self.sftn = SFTNFeatureExtractor(
            n_features=config.n_features,
            out_channels=config.sftn_channels,
            kernel_size=config.sftn_kernel,
            n_blocks=config.sftn_blocks,
        )

        # Stage 2: GRU Encoder (single direction)
        self.gru = nn.GRU(
            input_size=config.sftn_channels,
            hidden_size=config.gru_hidden,
            num_layers=config.gru_layers,
            batch_first=True,
            bidirectional=config.bidirectional,
        )
        gru_out = config.gru_hidden * (2 if config.bidirectional else 1)

        # Stage 3: Shared FC trunk
        self.trunk = nn.Sequential(
            nn.Linear(gru_out, config.trunk_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Stage 4: Branching heads — one per category
        # แต่ละ head: trunk → FC → Q-values (2 actions: off/on)
        self.heads = nn.ModuleList([
            self._build_head(config.trunk_hidden, config.head_hidden,
                             config.n_actions_per_head)
            for _ in range(self.n_heads)
        ])

    @staticmethod
    def _build_head(in_dim: int, hidden: int, n_actions: int) -> nn.Sequential:
        """Helper: build Q-value head for one category."""
        return nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shared encoding: input → trunk features.
        
        Args:
            x: (B, T, F)
        Returns:
            (B, trunk_hidden) — encoded representation
        """
        # SFTN
        h = self.sftn(x)              # (B, T, C_sftn)
        # GRU
        h, _ = self.gru(h)            # (B, T, gru_out)
        # Mean pool over time
        h = h.mean(dim=1)             # (B, gru_out)
        # Trunk
        h = self.trunk(h)             # (B, trunk_hidden)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass — return Q-values for all heads.
        
        Args:
            x: (B, T, F)
        Returns:
            q_values: (B, N_heads, n_actions) — Q-values สำหรับทุก (category, action)
        """
        h = self.encode(x)            # (B, trunk_hidden)

        # Apply each head, stack → (B, N, n_actions)
        head_outputs = [head(h) for head in self.heads]
        q_values = torch.stack(head_outputs, dim=1)  # (B, N, 2)
        return q_values

    @torch.no_grad()
    def select_action(
        self, x: torch.Tensor, epsilon: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select action mask using ε-greedy policy.
        
        ใช้ตอน inference + ตอน collect experience (training)
        
        Args:
            x:       (B, T, F)
            epsilon: exploration rate (0.0 = pure greedy)
        Returns:
            action_mask: (B, N) ∈ {0, 1} — binary mask: 1 = category active
            q_values:    (B, N, 2) — Q-values (สำหรับ logging / debug)
        """
        q_values = self.forward(x)    # (B, N, 2)
        batch_size = x.size(0)

        # Greedy action: argmax over action dim
        greedy_action = q_values.argmax(dim=-1)   # (B, N) ∈ {0, 1}

        if epsilon > 0.0:
            # ε-greedy: บางส่วน random
            random_action = torch.randint(
                0, self.config.n_actions_per_head,
                size=(batch_size, self.n_heads),
                device=x.device,
            )
            mask = torch.rand(batch_size, self.n_heads, device=x.device) < epsilon
            action = torch.where(mask, random_action, greedy_action)
        else:
            action = greedy_action

        return action, q_values

    def mask_to_categories(self, action_mask: torch.Tensor) -> List[List[str]]:
        """
        Convert binary action mask → list of active category names.
        
        Args:
            action_mask: (B, N) ∈ {0, 1}
        Returns:
            List of active category names per batch element
            e.g. [["Plug", "AC"], ["Light"], []]
        """
        active = []
        for row in action_mask:
            cats = [self.categories[i] for i, v in enumerate(row.tolist()) if v == 1]
            active.append(cats)
        return active

    def count_parameters(self) -> int:
        """Total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def compute_dqn_loss(
    online_net: BranchingDQNRouter,
    target_net: BranchingDQNRouter,
    states: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    next_states: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.95,
    use_double_dqn: bool = True,
    is_weights: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Branching DQN loss — per-head MSE with optional importance sampling weights.
    
    Per-head reward design:
        Each head ได้ reward เฉพาะ category ของตัวเอง — ทำให้ credit assignment ชัด
        (Q_AC head เรียนจาก reward ของ AC expert เท่านั้น เป็นต้น)
    
    Args:
        online_net:    Q-network ที่กำลัง train
        target_net:    target network (frozen, periodic update)
        states:        (B, T, F)
        actions:       (B, N) ∈ {0, 1} — action mask ที่เคยเลือก
        rewards:       (B, N) per-head OR (B,) scalar — รองรับทั้ง 2 mode
        next_states:   (B, T, F)
        dones:         (B,) — episode terminated flag
        gamma:         discount factor
        use_double_dqn: ใช้ Double DQN (ลด overestimation)
        is_weights:    (B,) importance sampling weights สำหรับ Prioritized Replay
                       — None ถ้าใช้ uniform replay
    
    Returns:
        loss:        scalar tensor — total loss across all heads
        td_errors:   (B,) per-sample TD errors — ใช้ update priorities ใน PER
    """
    batch_size = states.size(0)
    n_heads = online_net.n_heads

    # Current Q-values: Q(s, a) สำหรับ action ที่เลือกจริง
    q_values = online_net(states)                              # (B, N, 2)
    q_taken = q_values.gather(
        2, actions.unsqueeze(-1).long()
    ).squeeze(-1)                                              # (B, N)

    # Target Q-values
    with torch.no_grad():
        if use_double_dqn:
            # Double DQN: online net เลือก action, target net evaluate
            next_q_online = online_net(next_states)            # (B, N, 2)
            next_action = next_q_online.argmax(dim=-1)         # (B, N)
            next_q_target = target_net(next_states)            # (B, N, 2)
            next_q = next_q_target.gather(
                2, next_action.unsqueeze(-1).long()
            ).squeeze(-1)                                      # (B, N)
        else:
            next_q_target = target_net(next_states)
            next_q = next_q_target.max(dim=-1)[0]              # (B, N)

        # Per-head reward broadcasting
        if rewards.dim() == 1:
            # Scalar reward → broadcast ทุก head
            rewards_b = rewards.unsqueeze(-1)                  # (B, 1) → (B, N)
        else:
            # Per-head reward shape (B, N) ใช้ตรงๆ
            assert rewards.shape == (batch_size, n_heads), \
                f"rewards shape: {rewards.shape}, expected ({batch_size}, {n_heads})"
            rewards_b = rewards                                # (B, N)

        dones_b = dones.unsqueeze(-1)                          # (B, 1)
        target = rewards_b + gamma * (1.0 - dones_b) * next_q  # (B, N)

    # Per-sample TD error (mean over heads) — ใช้ update PER priorities
    td_errors_per_head = q_taken - target                      # (B, N)
    td_errors = td_errors_per_head.abs().mean(dim=-1)          # (B,)

    # MSE loss per sample (sum over heads — แต่ละ head contribute เท่ากัน)
    loss_per_sample = (td_errors_per_head ** 2).sum(dim=-1)    # (B,)

    # Importance sampling correction (Prioritized Replay)
    if is_weights is not None:
        loss_per_sample = loss_per_sample * is_weights         # (B,)

    loss = loss_per_sample.mean()
    return loss, td_errors.detach()
