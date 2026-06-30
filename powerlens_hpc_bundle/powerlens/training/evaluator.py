"""
Validation / Evaluation Metrics
================================
Standard NILM evaluation metrics + DRL-STFN-specific metrics

Per-category metrics:
    - Status accuracy (binary classification)
    - Power MAE (W)
    - Power MAPE (%)
    - Current MAE (A)

Disaggregation metrics (NILM standard):
    - NDE (Normalized Disaggregation Error): Σ|y_pred - y_true| / Σ y_true
    - F1 score for on/off detection
    - Total Energy Correctly Assigned (TECA) — NILM-specific

Router metrics:
    - Action accuracy (per-category)
    - Action accuracy (joint — all 4 categories correct)
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import torch


@dataclass
class CategoryMetrics:
    """Metrics for one category"""
    status_accuracy: float       # ∈ [0, 1]
    status_f1: float
    power_mae: float             # Watts
    power_mape: float            # %, 0 if denominator zero
    current_mae: float           # Amperes
    nde: float                   # Normalized Disaggregation Error


@dataclass
class EvalReport:
    """Full evaluation report — all categories + aggregate"""
    per_category: Dict[str, CategoryMetrics] = field(default_factory=dict)
    router_action_accuracy: float = 0.0       # per-head element-wise
    router_joint_accuracy: float = 0.0        # all categories correct simultaneously
    teca: float = 0.0                          # Total Energy Correctly Assigned
    n_samples: int = 0

    def to_dict(self) -> dict:
        """Flatten for CSV/JSON logging."""
        d = {
            "router/action_accuracy": self.router_action_accuracy,
            "router/joint_accuracy": self.router_joint_accuracy,
            "agg/teca": self.teca,
            "agg/n_samples": self.n_samples,
        }
        for cat, m in self.per_category.items():
            d[f"{cat}/status_acc"] = m.status_accuracy
            d[f"{cat}/status_f1"] = m.status_f1
            d[f"{cat}/power_mae"] = m.power_mae
            d[f"{cat}/power_mape"] = m.power_mape
            d[f"{cat}/current_mae"] = m.current_mae
            d[f"{cat}/nde"] = m.nde
        return d


# ============================================================
# Metric helpers
# ============================================================

def _f1_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Binary F1 score — handles edge cases (all zeros)."""
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    tp = (y_true & y_pred).sum()
    fp = (~y_true & y_pred).sum()
    fn = (y_true & ~y_pred).sum()
    if tp == 0:
        return 0.0 if (fp + fn) > 0 else 1.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-3) -> float:
    """MAPE with safe division — only on samples where y_true > epsilon."""
    mask = np.abs(y_true) > epsilon
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs(y_pred[mask] - y_true[mask]) / np.abs(y_true[mask])) * 100)


def _nde(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Normalized Disaggregation Error: Σ|y_pred - y_true| / Σ y_true"""
    denom = np.abs(y_true).sum()
    if denom < 1e-6:
        return 0.0 if np.abs(y_pred).sum() < 1e-6 else float('inf')
    return float(np.abs(y_pred - y_true).sum() / denom)


# ============================================================
# Main evaluator
# ============================================================

class Evaluator:
    """
    Evaluate DRL-STFN on validation data.
    
    Usage:
        >>> evaluator = Evaluator(trainer, data_module, categories)
        >>> report = evaluator.run(n_batches=20, batch_size=32)
        >>> print(report.to_dict())
    """
    def __init__(self, trainer, data_module, categories: List[str], device: str = "cpu"):
        self.trainer = trainer
        self.data_module = data_module
        self.categories = categories
        self.device = device

    @torch.no_grad()
    def run(self, n_batches: int = 20, batch_size: int = 32) -> EvalReport:
        """
        Evaluate on n_batches × batch_size validation samples.
        
        Returns EvalReport with per-category + router + TECA metrics.
        """
        self.trainer.experts.eval()
        self.trainer.router.eval()

        # Accumulate predictions/labels across batches
        all_preds = {cat: {"status": [], "power": [], "current": []} for cat in self.categories}
        all_truth = {cat: {"status": [], "power": [], "current": []} for cat in self.categories}
        all_action_pred = []
        all_action_true = []

        for _ in range(n_batches):
            batch = self.data_module.get_supervised_batch(batch_size)
            x = batch.features

            # Expert predictions (all categories, regardless of router)
            expert_outputs = self.trainer.experts.forward_all(x)

            for cat in self.categories:
                s_pred, p_pred, c_pred = expert_outputs[cat]
                all_preds[cat]["status"].append(s_pred.cpu().numpy().flatten())
                all_preds[cat]["power"].append(p_pred.cpu().numpy().flatten())
                all_preds[cat]["current"].append(c_pred.cpu().numpy().flatten())
                all_truth[cat]["status"].append(batch.truth_status[cat].cpu().numpy().flatten())
                all_truth[cat]["power"].append(batch.truth_power[cat].cpu().numpy().flatten())
                all_truth[cat]["current"].append(batch.truth_current[cat].cpu().numpy().flatten())

            # Router predictions
            action_pred, _ = self.trainer.router.select_action(x, epsilon=0.0)
            # Truth active mask: stack per-category truth_status > 0.5
            truth_action = torch.stack([
                (batch.truth_status[cat].squeeze(-1) > 0.5).long()
                for cat in self.categories
            ], dim=-1)    # (B, N)
            all_action_pred.append(action_pred.cpu().numpy())
            all_action_true.append(truth_action.cpu().numpy())

        # ---------- Per-category metrics ----------
        report = EvalReport()
        report.n_samples = n_batches * batch_size

        for cat in self.categories:
            s_pred = np.concatenate(all_preds[cat]["status"])
            p_pred = np.concatenate(all_preds[cat]["power"])
            c_pred = np.concatenate(all_preds[cat]["current"])
            s_true = np.concatenate(all_truth[cat]["status"])
            p_true = np.concatenate(all_truth[cat]["power"])
            c_true = np.concatenate(all_truth[cat]["current"])

            # Status: binarize prediction at 0.5
            s_pred_bin = (s_pred > 0.5).astype(np.float32)

            report.per_category[cat] = CategoryMetrics(
                status_accuracy=float((s_pred_bin == s_true).mean()),
                status_f1=_f1_binary(s_true, s_pred_bin),
                power_mae=float(np.mean(np.abs(p_pred - p_true))),
                power_mape=_safe_mape(p_true, p_pred),
                current_mae=float(np.mean(np.abs(c_pred - c_true))),
                nde=_nde(p_true, p_pred),
            )

        # ---------- Router metrics ----------
        action_pred_arr = np.concatenate(all_action_pred, axis=0)    # (total, N)
        action_true_arr = np.concatenate(all_action_true, axis=0)    # (total, N)
        report.router_action_accuracy = float((action_pred_arr == action_true_arr).mean())
        # Joint: all N positions correct simultaneously
        report.router_joint_accuracy = float(
            (action_pred_arr == action_true_arr).all(axis=-1).mean()
        )

        # ---------- TECA (Total Energy Correctly Assigned) ----------
        # TECA = 1 - Σ|y_pred - y_true| / (2 × Σ y_true)
        # Aggregate across categories
        total_err = 0.0
        total_truth = 0.0
        for cat in self.categories:
            p_pred = np.concatenate(all_preds[cat]["power"])
            p_true = np.concatenate(all_truth[cat]["power"])
            total_err += np.abs(p_pred - p_true).sum()
            total_truth += np.abs(p_true).sum()
        report.teca = float(1.0 - total_err / (2 * max(total_truth, 1e-6)))

        return report
