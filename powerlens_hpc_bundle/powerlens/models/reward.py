"""
Reward Function for DRL-STFN
=============================
แปลง expert disaggregation accuracy → per-head reward สำหรับ DRL Router

Reward design (per category):
    1. Accuracy reward: 
       - ถ้า category active (truth=1): reward ลดตาม |power_pred - power_true|
       - ถ้า category inactive (truth=0): reward เต็มถ้า predicted=0
    
    2. Routing penalty:
       - False Positive (FP): Router เลือก category ที่ไม่ active → penalty
       - False Negative (FN): Router ไม่เลือก category ที่ active → penalty หนักกว่า
       
       FN หนักกว่า FP เพราะ FN = miss disaggregation = lose entire energy attribution
                       FP = waste compute (รัน expert ที่ไม่จำเป็น) แต่ output น่าจะ ≈ 0

Per-head reward formula:
    For each category c:
        if truth_active[c] == 1:                        # appliance is on
            if router_chose[c] == 1:                    # router routed correctly
                r_c = -|power_pred - power_true| / scale  (accuracy reward, near 0 is best)
            else:                                       # FN
                r_c = -|power_true| / scale - fn_penalty  (heavy penalty)
        else:                                           # appliance is off
            if router_chose[c] == 0:                    # router skipped correctly
                r_c = 0                                 (no penalty, no error)
            else:                                       # FP
                r_c = -|power_pred| / scale - fp_penalty  (light penalty)
    
    All rewards clipped to [-clip, clip] for training stability
"""
from typing import Dict, Tuple
import torch

from .config import RewardConfig, REWARD_CONFIG


def compute_per_head_reward(
    action_mask: torch.Tensor,
    truth_active: torch.Tensor,
    expert_power_preds: torch.Tensor,
    truth_power: torch.Tensor,
    config: RewardConfig = REWARD_CONFIG,
) -> torch.Tensor:
    """
    Per-head reward computation.
    
    Args:
        action_mask:        (B, N) ∈ {0, 1} — Router's decision per category
        truth_active:       (B, N) ∈ {0, 1} — ground truth: category really active?
        expert_power_preds: (B, N) — expert predicted power per category (W)
                                    ใส่ 0 สำหรับ category ที่ Router ไม่เลือก
        truth_power:        (B, N) — ground truth power per category (W)
        config:             RewardConfig
    
    Returns:
        rewards: (B, N) per-head reward tensor
    """
    assert action_mask.shape == truth_active.shape == expert_power_preds.shape == truth_power.shape, \
        f"Shape mismatch: action={action_mask.shape}, truth_active={truth_active.shape}, " \
        f"pred={expert_power_preds.shape}, truth={truth_power.shape}"

    # Convert to float สำหรับ arithmetic
    action = action_mask.float()
    active = truth_active.float()

    # Power error scaled
    power_err = (expert_power_preds - truth_power).abs() / config.power_scale  # (B, N)
    abs_truth = truth_power.abs() / config.power_scale                          # (B, N)
    abs_pred = expert_power_preds.abs() / config.power_scale                    # (B, N)

    # Case classification (per element in batch × heads)
    # TP: action=1 AND active=1 → -power_err
    # TN: action=0 AND active=0 → 0
    # FP: action=1 AND active=0 → -abs_pred - fp_penalty
    # FN: action=0 AND active=1 → -abs_truth - fn_penalty

    is_tp = (action == 1) & (active == 1)
    is_tn = (action == 0) & (active == 0)
    is_fp = (action == 1) & (active == 0)
    is_fn = (action == 0) & (active == 1)

    rewards = torch.zeros_like(action)
    rewards = torch.where(is_tp, -config.accuracy_weight * power_err, rewards)
    rewards = torch.where(is_tn, torch.zeros_like(rewards), rewards)
    rewards = torch.where(is_fp, -abs_pred - config.false_positive_penalty, rewards)
    rewards = torch.where(is_fn, -abs_truth - config.false_negative_penalty, rewards)

    # Clip for training stability
    rewards = torch.clamp(rewards, -config.reward_clip, config.reward_clip)

    return rewards


def compute_expert_supervised_loss(
    expert_outputs: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    truth_status: Dict[str, torch.Tensor],
    truth_power: Dict[str, torch.Tensor],
    truth_current: Dict[str, torch.Tensor],
    loss_weight_status: float = 1.0,
    loss_weight_power: float = 0.5,
    loss_weight_current: float = 0.3,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Multi-task supervised loss for Experts (Stage 2 pretraining).
    
    Each expert ทำ 3 tasks: status (BCE), power (MSE), current (MSE)
    
    Args:
        expert_outputs: dict[category → (status_pred, power_pred, current_pred)]
                        จาก MultiCategoryExperts.forward_all()
        truth_status:   dict[category → (B, 1)] — ground truth on/off
        truth_power:    dict[category → (B, 1)] — ground truth power (W)
        truth_current:  dict[category → (B, 1)] — ground truth current (A)
    
    Returns:
        total_loss: scalar — weighted sum across categories and tasks
        loss_breakdown: dict for logging — per-category, per-task loss values
    """
    bce = torch.nn.functional.binary_cross_entropy
    mse = torch.nn.functional.mse_loss

    total = 0.0
    breakdown = {}

    for cat, (s_pred, p_pred, c_pred) in expert_outputs.items():
        s_true = truth_status[cat]
        p_true = truth_power[cat]
        c_true = truth_current[cat]

        # Status: BCE (sigmoid output already)
        l_status = bce(s_pred.clamp(1e-7, 1 - 1e-7), s_true)
        # Power: MSE (kW scale via dividing by 1000)
        l_power = mse(p_pred / 1000.0, p_true / 1000.0)
        # Current: MSE (A scale)
        l_current = mse(c_pred, c_true)

        weighted = (
            loss_weight_status * l_status
            + loss_weight_power * l_power
            + loss_weight_current * l_current
        )
        total = total + weighted

        breakdown[f"{cat}/status"] = l_status.item()
        breakdown[f"{cat}/power"] = l_power.item()
        breakdown[f"{cat}/current"] = l_current.item()

    # Average over categories
    total = total / len(expert_outputs)
    return total, breakdown
