"""
DRL-STFN Configuration
======================
Centralized hyperparameters for PowerLens DRL-STFN system.
ปรับค่าตรงนี้ที่เดียว ไม่ต้องแก้ในไฟล์อื่น
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class FeatureConfig:
    """
    Feature configuration.

    Default stays at the original 16-feature electrical schema, but real
    training can override this list from CSV columns. The intended workflow is
    all-feature training first, then ablation/feature selection before AWS
    deployment to reduce runtime cost.
    """
    feature_names: List[str] = field(default_factory=lambda: [
        "V_rms", "I_rms", "P", "Q", "PF", "THD",
        "H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9", "H10"
    ])
    source: str = "edge_16_default"

    @property
    def n_features(self) -> int:
        return len(self.feature_names)


@dataclass
class ExpertConfig:
    """DRL-STFN Expert Model hyperparameters"""
    # Input shape
    seq_len: int = 60          # 60 timesteps (1 minute @ 1 Hz)
    n_features: int = 16        # ตาม FeatureConfig

    # SFTN Feature Extraction
    sftn_channels: int = 64     # Conv1D output channels
    sftn_kernel: int = 3        # Conv1D kernel size

    # GRU-BiLSTM Backbone
    gru_hidden: int = 128       # → output 256 (bidirectional)
    gru_layers: int = 1         # เพิ่มได้ถ้าต้องการ deeper
    gru_dropout: float = 0.1

    # Multi-Head Attention
    attn_heads: int = 4
    attn_dropout: float = 0.1

    # Output heads
    head_hidden: int = 64       # FC layer ก่อน output

    # Categories
    categories: List[str] = field(default_factory=lambda: [
        "Plug", "Light", "AC", "Water_Heater"
    ])


@dataclass
class RouterConfig:
    """DRL Router (Branching DQN) hyperparameters"""
    # Input shape (เหมือน Expert)
    seq_len: int = 60
    n_features: int = 16

    # SFTN Feature Extraction (เบากว่า Expert)
    sftn_channels: int = 32     # ครึ่งของ Expert (64)
    sftn_kernel: int = 3
    sftn_blocks: int = 1        # 1 block พอ — Router ตัดสินใจง่ายกว่า

    # GRU Encoder (single direction, lighter)
    gru_hidden: int = 64        # ครึ่งของ Expert (128)
    gru_layers: int = 1
    bidirectional: bool = False  # single direction — เร็วกว่า

    # Branching DQN heads
    trunk_hidden: int = 128      # shared FC ก่อน split heads
    head_hidden: int = 64
    n_actions_per_head: int = 2  # binary (off=0, on=1) per category

    # Categories (ต้องตรงกับ ExpertConfig)
    categories: List[str] = field(default_factory=lambda: [
        "Plug", "Light", "AC", "Water_Heater"
    ])

    @property
    def n_heads(self) -> int:
        """จำนวน branching heads = จำนวน categories"""
        return len(self.categories)


@dataclass
class TrainConfig:
    """Training hyperparameters (สำหรับ module ถัดไป)"""
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 100

    # Multi-task loss weights
    loss_weight_status: float = 1.0
    loss_weight_power: float = 0.5
    loss_weight_current: float = 0.3


@dataclass
class RLConfig:
    """Reinforcement Learning hyperparameters สำหรับ Router DQN"""
    gamma: float = 0.95              # discount factor (NILM: window-level, ไม่ยาวมาก)
    epsilon_start: float = 1.0       # exploration: เริ่ม 100% random
    epsilon_end: float = 0.05        # exploration: จบที่ 5% random
    epsilon_decay_steps: int = 50_000

    # DQN training
    target_update_freq: int = 1000   # update target network ทุก N steps
    replay_buffer_size: int = 100_000
    min_replay_size: int = 5000      # อย่าเริ่ม train จนกว่า buffer จะมีพอ

    # Double DQN — ลด overestimation bias
    use_double_dqn: bool = True

    # Prioritized Experience Replay
    use_prioritized_replay: bool = True
    per_alpha: float = 0.6           # priority exponent (0=uniform, 1=full prioritization)
    per_beta_start: float = 0.4      # IS weight exponent — เริ่ม
    per_beta_end: float = 1.0        # IS weight exponent — จบ (anneal linear)
    per_beta_anneal_steps: int = 100_000
    per_epsilon: float = 1e-6        # ป้องกัน priority = 0


@dataclass
class RewardConfig:
    """Reward function hyperparameters"""
    # Per-head reward components
    accuracy_weight: float = 1.0     # weight ของ -|y_pred - y_true|
    false_positive_penalty: float = 0.5   # ลงโทษถ้าเลือก category ที่ไม่ active
    false_negative_penalty: float = 1.0   # ลงโทษหนักกว่าถ้าพลาด category ที่ active
    
    # Reward scaling
    power_scale: float = 1000.0      # normalize power error (W → kW scale)
    current_scale: float = 10.0      # normalize current error
    reward_clip: float = 10.0        # clip reward เพื่อ training stability


@dataclass
class ConfidenceConfig:
    """Confidence check hyperparameters (Stage 3 of inference pipeline)"""
    # Status sigmoid threshold to consider "on"
    status_threshold: float = 0.5

    # Per-category minimum power to consider "active" (Watts)
    # Below this → considered noise / off
    power_threshold: dict = field(default_factory=lambda: {
        "Plug": 5.0,            # phone charger, small device
        "Light": 3.0,           # LED bulb minimum
        "AC": 100.0,            # standby vs running
        "Water_Heater": 50.0,   # idle vs heating
    })

    # Confidence scoring
    high_conf_threshold: float = 0.7    # ≥ this → high confident → straight to fusion
    low_conf_threshold: float = 0.3     # < this → flag as new device candidate
    # In between [0.3, 0.7) → low conf (known device) → fallback path

    # Inconsistency penalty: multiply status_conf by this if status/power conflict
    inconsistency_penalty: float = 0.3


@dataclass
class ConstraintConfig:
    """Physical constraint check hyperparameters"""
    # Energy conservation tolerance — Σ disaggregated power ≤ aggregate × (1 + tol)
    energy_tolerance: float = 0.15      # 15% slack for measurement noise

    # Power-current consistency tolerance — |P - V·I·PF| / P
    power_current_tolerance: float = 0.20    # 20% — กรณี non-resistive load PF ปรวนแปร

    # Retry policy
    max_retries: int = 3
    retry_epsilons: list = field(default_factory=lambda: [0.5, 0.8, 1.0])
    # หลัง retry หมด → return current best (no further retry)


@dataclass
class FusionConfig:
    """Fusion & Postprocessing hyperparameters (Stage 4 of inference)"""
    # Final on/off threshold (status sigmoid)
    active_threshold: float = 0.5

    # Minimum confidence to count as active (below → treat as off, log as flag)
    min_confidence_to_activate: float = 0.3   # = ConfidenceConfig.low_conf_threshold

    # Energy rebalancing
    enable_rebalancing: bool = True
    overshoot_tolerance: float = 0.05         # ถ้า over-budget ≤ 5% → ไม่ rebalance (noise)

    # Residual allocation
    enable_residual: bool = True
    residual_threshold: float = 0.10          # under-budget ≥ 10% → จัดเป็น "Other"
    residual_category_name: str = "Other"

    # Output formatting
    round_decimal_places: int = 2             # ปัด power, current ก่อน output


# Default instances
FEATURE_CONFIG = FeatureConfig()
EXPERT_CONFIG = ExpertConfig()
ROUTER_CONFIG = RouterConfig()
TRAIN_CONFIG = TrainConfig()
RL_CONFIG = RLConfig()
REWARD_CONFIG = RewardConfig()
CONFIDENCE_CONFIG = ConfidenceConfig()
CONSTRAINT_CONFIG = ConstraintConfig()
FUSION_CONFIG = FusionConfig()
