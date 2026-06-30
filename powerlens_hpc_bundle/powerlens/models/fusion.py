"""
Fusion & Constraints Postprocessing (Stage 4)
==============================================
Final stage of DRL-STFN inference — แก้ output ให้ physically valid + format
ก่อนส่งไป MQTT/dashboard

Pipeline ตาม flowchart:
    Confidence Check → [Fusion & Constraints] → Constraint Check → Output
                                ▲
                                │
                            อยู่ตรงนี้

What this module does:
    1. Decision Fusion       — รวม confidence + status → final on/off
    2. Energy Rebalancing    — ถ้า Σpower > aggregate → confidence-weighted scaling
    3. Residual Allocation   — ถ้า Σpower << aggregate → "Other" category รับ residual
    4. Output Formatting     — JSON schema สำหรับ MQTT/dashboard

Confidence-weighted scaling design:
    เมื่อ over-budget ต้อง scale ลง — แต่ไม่อยากลด category ที่มั่นใจสูง
    
    weight_i = confidence_i^α       (α=2 → emphasize confidence แบบ quadratic)
    
    excess = Σpower_i - aggregate
    
    # Distribute excess inversely to weight (low conf → take more cut)
    inv_weight_i = 1 / (weight_i + ε)
    cut_i = excess × (inv_weight_i / Σ inv_weight_j)
    power_i_new = max(0, power_i - cut_i)
    
    Iterate ถ้าหลัง scale ยัง over (because of clipping at 0)
"""
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from datetime import datetime, timezone

from .inference import CategoryPrediction, ConstraintResult, ConfidenceLevel
from .config import FusionConfig, FUSION_CONFIG


# ============================================================
# Output schema
# ============================================================

@dataclass
class CategoryOutput:
    """Per-category output for dashboard/MQTT"""
    category: str
    active: bool
    power: float          # final power (W) — possibly rebalanced
    current: float        # final current (A)
    confidence: float     # ∈ [0, 1]
    raw_power: float      # power ก่อน rebalancing (for debugging)


@dataclass
class FusionOutput:
    """
    Final fused output ของ inference pipeline.
    
    ส่งไป MQTT/dashboard via to_dict()
    """
    timestamp: str                              # ISO 8601 UTC
    aggregate_power: float                      # meter reading (W)
    sum_disaggregated: float                    # Σ category powers หลัง fusion
    residual: float                             # aggregate - sum (W) — unaccounted
    categories: List[CategoryOutput]
    rebalanced: bool                            # มี rebalancing หรือไม่
    rebalance_factor: float                     # scale factor ที่ใช้ (1.0 = ไม่ scale)
    flags: List[str]                            # warnings (low confidence, inconsistent)
    constraint_passed: bool

    def to_dict(self) -> dict:
        """Serialize to JSON-friendly dict"""
        return {
            "timestamp": self.timestamp,
            "aggregate_power_w": self.aggregate_power,
            "sum_disaggregated_w": self.sum_disaggregated,
            "residual_w": self.residual,
            "categories": [asdict(c) for c in self.categories],
            "metadata": {
                "rebalanced": self.rebalanced,
                "rebalance_factor": self.rebalance_factor,
                "constraint_passed": self.constraint_passed,
                "flags": self.flags,
            },
        }


# ============================================================
# Fusion processor
# ============================================================

class FusionProcessor:
    """
    Final postprocessing ก่อน output.
    
    Public API:
        process(predictions, aggregate_power, constraint_result) → FusionOutput
    """
    def __init__(self, config: FusionConfig = FUSION_CONFIG):
        self.cfg = config

    # ----------------------------------------------------------------
    # Step 1: Decision Fusion
    # ----------------------------------------------------------------
    def _decide_active(self, pred: CategoryPrediction) -> bool:
        """
        Final on/off decision combining all signals.
        
        Active = routed AND status>threshold AND consistent AND confidence sufficient
        """
        if not pred.routed:
            return False
        if pred.status < self.cfg.active_threshold:
            return False
        if not pred.consistent:
            return False  # status/power mismatch → ไม่ trust
        if pred.confidence < self.cfg.min_confidence_to_activate:
            return False
        if pred.confidence_level == ConfidenceLevel.LOW_UNKNOWN:
            return False  # Phase 2 จะ pickup จาก flag
        return True

    # ----------------------------------------------------------------
    # Step 2: Confidence-weighted Energy Rebalancing
    # ----------------------------------------------------------------
    def _rebalance_powers(
        self,
        active_preds: List[CategoryPrediction],
        target_total: float,
        max_iterations: int = 5,
    ) -> Dict[str, float]:
        """
        Confidence-weighted proportional reduction เมื่อ over-budget.
        
        Algorithm:
            1. คำนวณ inverse-confidence weights (low conf takes bigger cut)
            2. distribute excess proportionally to inverse weights
            3. clip ที่ 0 (power ติดลบไม่ได้)
            4. iterate ถ้าหลัง clip ยัง over (เพราะ clipping makes others take more)
        
        Args:
            active_preds:  predictions ที่ active เท่านั้น
            target_total:  aggregate power × (1 + tolerance) — upper bound
            max_iterations: cap iterations (rare ที่ต้องเกิน 2-3)
        Returns:
            dict[category → adjusted power (W)]
        """
        powers = {p.category: p.power for p in active_preds}
        confidences = {p.category: max(p.confidence, 1e-3) for p in active_preds}

        for _ in range(max_iterations):
            current_sum = sum(powers.values())
            excess = current_sum - target_total
            if excess <= 0:
                break  # within budget, done

            # คำนวณ inverse-weight (เน้นตัด category ที่ confidence ต่ำ)
            # weight_i = conf_i^2 → inv_weight_i = 1/conf_i^2
            inv_weights = {
                cat: 1.0 / (confidences[cat] ** 2)
                for cat, p in powers.items() if p > 0
            }
            total_inv = sum(inv_weights.values())
            if total_inv == 0:
                break  # nothing left to cut

            # Distribute excess ตาม inverse weight
            for cat in list(powers.keys()):
                if powers[cat] <= 0:
                    continue
                cut = excess * (inv_weights[cat] / total_inv)
                powers[cat] = max(0.0, powers[cat] - cut)

        return powers

    # ----------------------------------------------------------------
    # Step 3: Residual Allocation
    # ----------------------------------------------------------------
    def _compute_residual(self, sum_disaggregated: float, aggregate: float) -> float:
        """
        Compute unaccounted energy.
        
        ถ้า sum << aggregate → มี devices ที่ไม่ใช่ 4 categories ของเรา
        (ตู้แช่ใหม่, เครื่องชงกาแฟ, ฯลฯ)
        """
        residual = aggregate - sum_disaggregated
        # Threshold check: ถ้า residual น้อยมาก (< 10% ของ aggregate) → ignore (noise)
        if abs(residual) < aggregate * self.cfg.residual_threshold:
            return 0.0
        return max(0.0, residual)

    # ----------------------------------------------------------------
    # Main: process()
    # ----------------------------------------------------------------
    def process(
        self,
        predictions: List[CategoryPrediction],
        aggregate_power: float,
        constraint_result: ConstraintResult,
        flags: List[str],
        timestamp: Optional[str] = None,
    ) -> FusionOutput:
        """
        Full fusion pipeline.
        
        Args:
            predictions:        per-category predictions จาก confidence check
            aggregate_power:    meter reading (W)
            constraint_result:  result จาก ConstraintChecker
            flags:              flags จาก inference pipeline
            timestamp:          ISO 8601 UTC string (default = now)
        Returns:
            FusionOutput ready to serialize
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()

        # ---- Step 1: Decision fusion (final on/off per category) ----
        active_preds = [p for p in predictions if self._decide_active(p)]
        active_cat_names = {p.category for p in active_preds}

        # ---- Step 2: Energy rebalancing (if over budget) ----
        rebalance_factor = 1.0
        rebalanced = False
        adjusted_powers = {p.category: p.power for p in active_preds}

        if self.cfg.enable_rebalancing and len(active_preds) > 0:
            current_sum = sum(adjusted_powers.values())
            target_upper = aggregate_power * (1.0 + self.cfg.overshoot_tolerance)

            if current_sum > target_upper and aggregate_power > 0:
                adjusted_powers = self._rebalance_powers(active_preds, target_upper)
                rebalanced = True
                new_sum = sum(adjusted_powers.values())
                rebalance_factor = new_sum / current_sum if current_sum > 0 else 1.0

        # ---- Build per-category outputs ----
        cat_outputs = []
        for p in predictions:
            is_active = p.category in active_cat_names
            adjusted = adjusted_powers.get(p.category, 0.0) if is_active else 0.0

            # Adjust current proportionally if power was scaled
            if is_active and p.power > 0 and adjusted != p.power:
                adjusted_current = p.current * (adjusted / p.power)
            elif is_active:
                adjusted_current = p.current
            else:
                adjusted_current = 0.0

            cat_outputs.append(CategoryOutput(
                category=p.category,
                active=is_active,
                power=round(adjusted, self.cfg.round_decimal_places),
                current=round(adjusted_current, self.cfg.round_decimal_places),
                confidence=round(p.confidence, 3),
                raw_power=round(p.power, self.cfg.round_decimal_places),
            ))

        # ---- Step 3: Residual allocation ("Other" virtual category) ----
        sum_disaggregated = sum(c.power for c in cat_outputs if c.active)
        residual = 0.0
        if self.cfg.enable_residual:
            residual = self._compute_residual(sum_disaggregated, aggregate_power)
            if residual > 0:
                # Append "Other" category with residual
                cat_outputs.append(CategoryOutput(
                    category=self.cfg.residual_category_name,
                    active=True,
                    power=round(residual, self.cfg.round_decimal_places),
                    current=0.0,         # ไม่รู้ I — เป็น aggregate residual
                    confidence=0.0,      # by design ไม่มั่นใจ
                    raw_power=round(residual, self.cfg.round_decimal_places),
                ))
                sum_disaggregated += residual

        return FusionOutput(
            timestamp=timestamp,
            aggregate_power=round(aggregate_power, self.cfg.round_decimal_places),
            sum_disaggregated=round(sum_disaggregated, self.cfg.round_decimal_places),
            residual=round(residual, self.cfg.round_decimal_places),
            categories=cat_outputs,
            rebalanced=rebalanced,
            rebalance_factor=round(rebalance_factor, 4),
            flags=flags,
            constraint_passed=constraint_result.passed,
        )
