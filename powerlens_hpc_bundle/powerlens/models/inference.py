"""
Confidence Check + Constraint Validation
=========================================
Stage 3 ของ DRL-STFN inference pipeline (หลัง Router + Experts)

Flow ตาม flowchart:
    DRL-STFN Output → Confident Check → Fusion & Constraints → Constraint Check → Output
                          │
                          ├─ High Confident → Fusion ตรงๆ
                          ├─ Low Confident (Known Device) → Fallback (Phase 2)
                          └─ Low Confident (New Device)   → Auto-Calibrate (Phase 2)
    
    ถ้า Constraint Check ไม่ผ่าน → retry กลับไป Router (max 3 retries)

MVP scope:
    ✅ Confidence check (status + power consistency)
    ✅ Constraint check (energy conservation + power-current consistency)
    ✅ Retry loop (force high epsilon)
    ✅ Flag low-confidence events → ส่ง S3 logging
    ❌ Fallback Strategy เต็ม (Phase 2)
    ❌ Unknown Device Detector (Phase 2)
"""
from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import torch

from .config import (
    ConfidenceConfig, ConstraintConfig,
    CONFIDENCE_CONFIG, CONSTRAINT_CONFIG,
)


# ============================================================
# Confidence levels (per category)
# ============================================================

class ConfidenceLevel(Enum):
    """Per-category confidence classification"""
    HIGH = "high"                # ≥ high_threshold → straight to fusion
    LOW_KNOWN = "low_known"      # in [low, high) → fallback (Phase 2: just flag)
    LOW_UNKNOWN = "low_unknown"  # < low_threshold → unknown device candidate


@dataclass
class CategoryPrediction:
    """
    Per-category inference result + diagnostic info.
    
    Used by both: confidence checker (input) และ logging/fusion (output)
    """
    category: str
    routed: bool                # Router เลือก category นี้?
    status: float               # sigmoid output 0-1
    power: float                # Watts
    current: float              # Amperes
    confidence: float           # ∈ [0, 1]
    confidence_level: ConfidenceLevel
    consistent: bool            # status/power ขัดแย้งกันไหม?

    def is_active(self, threshold: float = 0.5) -> bool:
        """Final on/off decision after confidence + consistency check."""
        return self.routed and self.status >= threshold and self.consistent


# ============================================================
# Confidence Check
# ============================================================

class ConfidenceChecker:
    """
    Per-category confidence + consistency checking.
    
    Confidence formula:
        status_conf = 2 * |status - 0.5|        # ∈ [0, 1]
        power_active = power > P_min[category]
        consistent = (status > 0.5) == power_active
        
        if consistent:
            confidence = status_conf
        else:
            confidence = status_conf * inconsistency_penalty
    
    Classification:
        confidence ≥ high_threshold       → HIGH
        confidence ∈ [low, high)          → LOW_KNOWN
        confidence < low_threshold        → LOW_UNKNOWN
    """
    def __init__(self, config: ConfidenceConfig = CONFIDENCE_CONFIG):
        self.cfg = config

    def check_single(
        self, category: str, status: float, power: float, current: float, routed: bool
    ) -> CategoryPrediction:
        """
        Confidence check for one category prediction.
        
        Args:
            category: category name
            status:   sigmoid output (0-1)
            power:    predicted power (W)
            current:  predicted current (A)
            routed:   Router เลือก category นี้ไหม
        Returns:
            CategoryPrediction with confidence + level
        """
        # ถ้า Router ไม่เลือก category นี้ → ไม่มี prediction → confidence = 1 (confident off)
        if not routed:
            return CategoryPrediction(
                category=category, routed=False,
                status=0.0, power=0.0, current=0.0,
                confidence=1.0, confidence_level=ConfidenceLevel.HIGH,
                consistent=True,
            )

        # Status confidence — ใกล้ 0 หรือ 1 = confident
        status_conf = 2.0 * abs(status - 0.5)

        # Consistency: status_on ⇔ power_active
        p_min = self.cfg.power_threshold.get(category, 10.0)
        status_on = status >= self.cfg.status_threshold
        power_active = power > p_min
        consistent = status_on == power_active

        # Apply inconsistency penalty
        confidence = status_conf if consistent else status_conf * self.cfg.inconsistency_penalty

        # Classify level
        if confidence >= self.cfg.high_conf_threshold:
            level = ConfidenceLevel.HIGH
        elif confidence >= self.cfg.low_conf_threshold:
            level = ConfidenceLevel.LOW_KNOWN
        else:
            level = ConfidenceLevel.LOW_UNKNOWN

        return CategoryPrediction(
            category=category, routed=True,
            status=status, power=power, current=current,
            confidence=confidence, confidence_level=level,
            consistent=consistent,
        )

    def check_batch(
        self,
        categories: List[str],
        action_mask: torch.Tensor,
        status_preds: torch.Tensor,
        power_preds: torch.Tensor,
        current_preds: torch.Tensor,
    ) -> List[List[CategoryPrediction]]:
        """
        Batched confidence check.
        
        Args:
            categories:    list of category names (length N)
            action_mask:   (B, N) ∈ {0, 1} — Router decisions
            status_preds:  (B, N) — sigmoid outputs
            power_preds:   (B, N) — predicted power (W)
            current_preds: (B, N) — predicted current (A)
        Returns:
            List of lists — outer = batch, inner = per-category predictions
        """
        B, N = action_mask.shape
        assert N == len(categories)
        results = []

        for b in range(B):
            row = []
            for i, cat in enumerate(categories):
                pred = self.check_single(
                    category=cat,
                    status=float(status_preds[b, i]),
                    power=float(power_preds[b, i]),
                    current=float(current_preds[b, i]),
                    routed=bool(action_mask[b, i].item()),
                )
                row.append(pred)
            results.append(row)
        return results


# ============================================================
# Constraint Check
# ============================================================

@dataclass
class ConstraintViolation:
    """รายละเอียด violation ที่ตรวจเจอ"""
    name: str
    actual: float
    expected: float
    margin: float          # |actual - expected| / expected (relative)


@dataclass
class ConstraintResult:
    """Result ของ constraint check"""
    passed: bool
    violations: List[ConstraintViolation]


class ConstraintChecker:
    """
    Physical constraint validation บน aggregate disaggregated outputs.
    
    Constraints checked:
        1. Energy conservation:  Σ power_i ≤ aggregate × (1 + tolerance)
        2. Power-current consistency:  per-category |P - V·I·PF| / P ≤ tolerance
        
        Non-negativity ถูก enforce ที่ Expert layer (ReLU) แล้ว — ไม่ check ที่นี่
    """
    def __init__(self, config: ConstraintConfig = CONSTRAINT_CONFIG):
        self.cfg = config

    def check(
        self,
        predictions: List[CategoryPrediction],
        aggregate_power: float,
        v_rms: Optional[float] = None,
        pf: Optional[float] = None,
    ) -> ConstraintResult:
        """
        Run all constraints, collect violations.
        
        Args:
            predictions:     per-category predictions
            aggregate_power: meter reading aggregate (W)
            v_rms:           voltage RMS (V) — สำหรับ power-current consistency
            pf:              power factor — ถ้ามีจะ check P=V·I·PF
        Returns:
            ConstraintResult with passed flag + violation details
        """
        violations = []

        # --- Constraint 1: Energy conservation ---
        sum_power = sum(p.power for p in predictions if p.is_active())
        upper_bound = aggregate_power * (1.0 + self.cfg.energy_tolerance)

        if sum_power > upper_bound:
            margin = (sum_power - aggregate_power) / aggregate_power if aggregate_power > 0 else float('inf')
            violations.append(ConstraintViolation(
                name="energy_conservation",
                actual=sum_power,
                expected=aggregate_power,
                margin=margin,
            ))

        # --- Constraint 2: Power-current consistency (per active category) ---
        if v_rms is not None and pf is not None and v_rms > 0:
            for p in predictions:
                if not p.is_active():
                    continue
                # P_expected = V × I × PF
                p_expected = v_rms * p.current * pf
                if p_expected < 1e-6:
                    continue  # avoid divide-by-zero ถ้า current ใกล้ 0
                relative_err = abs(p.power - p_expected) / max(p_expected, 1e-6)
                if relative_err > self.cfg.power_current_tolerance:
                    violations.append(ConstraintViolation(
                        name=f"power_current_consistency_{p.category}",
                        actual=p.power,
                        expected=p_expected,
                        margin=relative_err,
                    ))

        return ConstraintResult(passed=len(violations) == 0, violations=violations)


# ============================================================
# Inference Pipeline with Retry Loop
# ============================================================

@dataclass
class InferenceResult:
    """Final inference output"""
    predictions: List[CategoryPrediction]
    constraint_result: ConstraintResult
    retries_used: int
    final_action_mask: torch.Tensor      # (N,) — final action ที่ใช้
    flagged_for_logging: List[str]       # category names ที่ confidence ต่ำ → log to S3
    fusion_output: Optional["FusionOutput"] = None    # populated ถ้า fusion ถูกเปิด


class InferencePipeline:
    """
    Full inference pipeline with retry loop + optional fusion stage.
    
    Flow:
        1. Router → action mask
        2. Run selected experts → predictions
        3. Confidence check per category
        4. Constraint check
        5. ถ้า not pass + retries left → step 1 with higher epsilon
        6. (Optional) Fusion postprocessing → FusionOutput ready for MQTT
        7. Return final result
    """
    def __init__(
        self,
        router,                # BranchingDQNRouter
        experts,               # MultiCategoryExperts
        confidence_checker: ConfidenceChecker,
        constraint_checker: ConstraintChecker,
        constraint_config: ConstraintConfig = CONSTRAINT_CONFIG,
        fusion_processor=None,  # Optional FusionProcessor — None = skip fusion stage
    ):
        self.router = router
        self.experts = experts
        self.confidence_checker = confidence_checker
        self.constraint_checker = constraint_checker
        self.fusion_processor = fusion_processor
        self.cfg = constraint_config
        self.categories = router.categories

    @torch.no_grad()
    def _run_single_pass(
        self, state: torch.Tensor, epsilon: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        One pass: Router → Experts → predictions (no checks yet).
        
        Args:
            state:   (1, T, F)
            epsilon: exploration rate (0 = greedy)
        Returns:
            action_mask: (1, N)
            status:      (1, N) sigmoid
            power:       (1, N) Watts
            current:     (1, N) Amperes
        """
        action_mask, _ = self.router.select_action(state, epsilon=epsilon)
        N = self.router.n_heads

        status = torch.zeros(1, N, device=state.device)
        power = torch.zeros(1, N, device=state.device)
        current = torch.zeros(1, N, device=state.device)

        for i, cat in enumerate(self.categories):
            if action_mask[0, i].item() == 1:
                s, p, c = self.experts(state, category=cat)
                status[0, i] = s.squeeze()
                power[0, i] = p.squeeze()
                current[0, i] = c.squeeze()

        return action_mask, status, power, current

    @torch.no_grad()
    def infer(
        self,
        state: torch.Tensor,
        aggregate_power: float,
        v_rms: Optional[float] = None,
        pf: Optional[float] = None,
        timestamp: Optional[str] = None,
    ) -> InferenceResult:
        """
        Full inference with retry loop + optional fusion postprocessing.
        
        Args:
            state:           (1, T, F) input window
            aggregate_power: meter aggregate reading (W) — for energy constraint
            v_rms, pf:       optional — for power-current consistency check
            timestamp:       ISO 8601 string for FusionOutput (default = now)
        Returns:
            InferenceResult with optional fusion_output populated
        """
        self.router.eval()
        self.experts.eval()

        best_predictions = None
        best_result = None
        best_action_mask = None

        for attempt in range(self.cfg.max_retries + 1):
            if attempt == 0:
                epsilon = 0.0
            else:
                idx = min(attempt - 1, len(self.cfg.retry_epsilons) - 1)
                epsilon = self.cfg.retry_epsilons[idx]

            action_mask, status, power, current = self._run_single_pass(state, epsilon=epsilon)

            preds_batch = self.confidence_checker.check_batch(
                self.categories, action_mask, status, power, current,
            )
            preds = preds_batch[0]

            result = self.constraint_checker.check(
                preds, aggregate_power=aggregate_power, v_rms=v_rms, pf=pf,
            )

            if best_result is None or self._is_better(result, best_result):
                best_predictions = preds
                best_result = result
                best_action_mask = action_mask[0].clone()

            if result.passed:
                break

        # Build base result
        flags = self._collect_flags(best_predictions)
        fusion_output = None
        if self.fusion_processor is not None:
            fusion_output = self.fusion_processor.process(
                predictions=best_predictions,
                aggregate_power=aggregate_power,
                constraint_result=best_result,
                flags=flags,
                timestamp=timestamp,
            )

        return InferenceResult(
            predictions=best_predictions,
            constraint_result=best_result,
            retries_used=attempt if best_result.passed else self.cfg.max_retries,
            final_action_mask=best_action_mask,
            flagged_for_logging=flags,
            fusion_output=fusion_output,
        )

    @staticmethod
    def _is_better(new: ConstraintResult, old: ConstraintResult) -> bool:
        """Compare 2 results — fewer violations OR lower margin = better."""
        if new.passed and not old.passed:
            return True
        if old.passed and not new.passed:
            return False
        # Both passed or both failed → compare violation counts
        if len(new.violations) < len(old.violations):
            return True
        if len(new.violations) > len(old.violations):
            return False
        # Same count → compare total margin
        new_margin = sum(v.margin for v in new.violations)
        old_margin = sum(v.margin for v in old.violations)
        return new_margin < old_margin

    @staticmethod
    def _collect_flags(predictions: List[CategoryPrediction]) -> List[str]:
        """
        Collect categories ที่ควร log ไป S3 (low confidence + inconsistency events).
        
        Priority order (1 flag per category, more specific wins):
            1. INCONSISTENT (status/power conflict — ของไม่ทำงานตามที่คิด)
            2. UNKNOWN_CANDIDATE (LOW_UNKNOWN — น่าจะเป็น device ใหม่)
            3. LOW_CONF (LOW_KNOWN — confident แค่ทรง ๆ)
        """
        flags = []
        for p in predictions:
            if not p.routed:
                continue
            # Inconsistency takes priority — physical contradiction is the most actionable signal
            if not p.consistent:
                flags.append(f"{p.category}:INCONSISTENT")
            elif p.confidence_level == ConfidenceLevel.LOW_UNKNOWN:
                flags.append(f"{p.category}:UNKNOWN_CANDIDATE")
            elif p.confidence_level == ConfidenceLevel.LOW_KNOWN:
                flags.append(f"{p.category}:LOW_CONF")
        return flags
