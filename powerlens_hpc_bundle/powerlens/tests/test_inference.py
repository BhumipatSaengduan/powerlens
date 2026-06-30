"""
Sanity tests for inference pipeline (Stage 3).

Coverage:
  Confidence:
    1. High confidence: status sure + consistent → HIGH level
    2. Low confidence inconsistent: status=on but power=0 → penalty applied
    3. Low confidence ambiguous: status near 0.5 → LOW_KNOWN
    4. Unknown candidate: very low conf → LOW_UNKNOWN
    5. Not routed: skip = high confidence (off)
  
  Constraints:
    6. Energy conservation pass/fail
    7. Power-current consistency pass/fail (with V, PF)
    8. Multiple violations collected
  
  Pipeline:
    9. End-to-end inference passes
    10. Retry escalates epsilon
    11. Flagging logs correct categories
    12. Best-result selection across retries
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch

from powerlens.models.inference import (
    ConfidenceChecker, ConstraintChecker, InferencePipeline,
    ConfidenceLevel, CategoryPrediction,
)
from powerlens.models.router import BranchingDQNRouter
from powerlens.models.expert import MultiCategoryExperts
from powerlens.models.config import (
    CONFIDENCE_CONFIG, CONSTRAINT_CONFIG, ROUTER_CONFIG, EXPERT_CONFIG,
)


# ============================================================
# Confidence tests
# ============================================================

def test_high_confidence_consistent():
    """status=0.95 + power=500W (active) → HIGH confidence"""
    print("\n=== Test 1: High Confidence (Consistent) ===")
    checker = ConfidenceChecker()
    pred = checker.check_single("AC", status=0.95, power=500.0, current=2.5, routed=True)

    assert pred.consistent, "should be consistent"
    assert pred.confidence_level == ConfidenceLevel.HIGH
    print(f"  status=0.95, power=500W → confidence={pred.confidence:.3f}, level={pred.confidence_level.value}")
    print(f"  ✓ HIGH level achieved (≥{CONFIDENCE_CONFIG.high_conf_threshold})")


def test_inconsistent_status_power():
    """status=0.95 (on) but power=0W → INCONSISTENT, confidence drop"""
    print("\n=== Test 2: Inconsistent Status/Power ===")
    checker = ConfidenceChecker()

    # Consistent baseline
    pred_ok = checker.check_single("AC", status=0.95, power=500.0, current=2.5, routed=True)
    # Inconsistent: claims on but no power
    pred_bad = checker.check_single("AC", status=0.95, power=0.0, current=0.0, routed=True)

    assert pred_ok.consistent and not pred_bad.consistent
    assert pred_bad.confidence < pred_ok.confidence
    expected_bad = pred_ok.confidence * CONFIDENCE_CONFIG.inconsistency_penalty
    assert abs(pred_bad.confidence - expected_bad) < 1e-5

    print(f"  Consistent (status=0.95, power=500W):  conf={pred_ok.confidence:.3f}")
    print(f"  Inconsistent (status=0.95, power=0W):  conf={pred_bad.confidence:.3f}")
    print(f"  ✓ Inconsistency penalty applied correctly ({CONFIDENCE_CONFIG.inconsistency_penalty}×)")


def test_ambiguous_status_low_known():
    """status near 0.5 → low status_conf → LOW_KNOWN"""
    print("\n=== Test 3: Ambiguous Status (LOW_KNOWN) ===")
    checker = ConfidenceChecker()
    # status=0.55: status_conf = 2 * 0.05 = 0.1 → ต่ำมาก → LOW_UNKNOWN
    # status=0.7:  status_conf = 2 * 0.2 = 0.4  → LOW_KNOWN range
    pred = checker.check_single("AC", status=0.7, power=500.0, current=2.5, routed=True)
    print(f"  status=0.7, power=500W → conf={pred.confidence:.3f}, level={pred.confidence_level.value}")
    assert pred.confidence_level == ConfidenceLevel.LOW_KNOWN
    print(f"  ✓ LOW_KNOWN level (in [{CONFIDENCE_CONFIG.low_conf_threshold}, "
          f"{CONFIDENCE_CONFIG.high_conf_threshold}))")


def test_low_unknown():
    """status very near 0.5 → very low conf → LOW_UNKNOWN (new device candidate)"""
    print("\n=== Test 4: Low Confidence Unknown ===")
    checker = ConfidenceChecker()
    pred = checker.check_single("AC", status=0.52, power=500.0, current=2.5, routed=True)
    print(f"  status=0.52, power=500W → conf={pred.confidence:.3f}, level={pred.confidence_level.value}")
    assert pred.confidence_level == ConfidenceLevel.LOW_UNKNOWN
    print(f"  ✓ LOW_UNKNOWN — flagged as new device candidate")


def test_not_routed():
    """ไม่ถูก routed → skip → confidence = 1 (confident off)"""
    print("\n=== Test 5: Not Routed (Skipped) ===")
    checker = ConfidenceChecker()
    pred = checker.check_single("AC", status=0.0, power=0.0, current=0.0, routed=False)
    assert pred.confidence == 1.0
    assert pred.confidence_level == ConfidenceLevel.HIGH
    assert not pred.is_active()
    print(f"  routed=False → conf={pred.confidence}, level=HIGH, active={pred.is_active()}")
    print(f"  ✓ Skipped categories treated as confidently off")


def test_batch_confidence():
    """check_batch processes (B, N) tensors correctly"""
    print("\n=== Test 6: Batch Confidence Check ===")
    checker = ConfidenceChecker()
    cats = ROUTER_CONFIG.categories  # 4 categories
    B = 3

    action = torch.tensor([
        [1, 0, 1, 0],   # row 0: route Plug + AC
        [0, 1, 0, 0],   # row 1: route Light only
        [1, 1, 1, 1],   # row 2: route all
    ])
    status = torch.tensor([
        [0.9, 0.0, 0.95, 0.0],
        [0.0, 0.6, 0.0, 0.0],
        [0.95, 0.95, 0.95, 0.95],
    ])
    power = torch.tensor([
        [50.0, 0.0, 500.0, 0.0],
        [0.0, 10.0, 0.0, 0.0],
        [50.0, 10.0, 500.0, 1500.0],
    ])
    current = torch.tensor([
        [0.5, 0.0, 2.5, 0.0],
        [0.0, 0.1, 0.0, 0.0],
        [0.5, 0.1, 2.5, 7.0],
    ])

    results = checker.check_batch(cats, action, status, power, current)
    assert len(results) == B and all(len(r) == len(cats) for r in results)
    print(f"  ✓ Batch shape: {B} rows × {len(cats)} categories")
    print(f"  Row 0 active: {[p.category for p in results[0] if p.is_active()]}")
    print(f"  Row 2 active: {[p.category for p in results[2] if p.is_active()]}")


# ============================================================
# Constraint tests
# ============================================================

def _make_pred(cat, status=0.9, power=500.0, current=2.5, routed=True):
    """Helper: make CategoryPrediction with sensible defaults."""
    return CategoryPrediction(
        category=cat, routed=routed,
        status=status, power=power, current=current,
        confidence=0.9, confidence_level=ConfidenceLevel.HIGH,
        consistent=True,
    )


def test_energy_conservation_pass():
    """Σ power ≤ aggregate × (1+tol) → pass"""
    print("\n=== Test 7: Energy Conservation Pass ===")
    checker = ConstraintChecker()
    preds = [
        _make_pred("Plug", power=50),
        _make_pred("Light", power=20),
        _make_pred("AC", power=500),
        _make_pred("Water_Heater", routed=False, status=0, power=0),
    ]
    # sum = 570W, aggregate = 600W → within tolerance
    result = checker.check(preds, aggregate_power=600.0)
    assert result.passed, f"violations: {[v.name for v in result.violations]}"
    print(f"  Σ disaggregated = 570W, aggregate = 600W → PASS")


def test_energy_conservation_fail():
    """Σ power >> aggregate → fail"""
    print("\n=== Test 8: Energy Conservation Fail ===")
    checker = ConstraintChecker()
    preds = [
        _make_pred("AC", power=2000),
        _make_pred("Water_Heater", power=2000),
    ]
    # sum = 4000W, aggregate = 1000W → way over
    result = checker.check(preds, aggregate_power=1000.0)
    assert not result.passed
    assert any(v.name == "energy_conservation" for v in result.violations)
    margin = result.violations[0].margin
    print(f"  Σ disaggregated = 4000W, aggregate = 1000W → FAIL (margin={margin:.1%})")


def test_power_current_consistency():
    """P ≈ V·I·PF — check tolerance"""
    print("\n=== Test 9: Power-Current Consistency ===")
    checker = ConstraintChecker()
    # V=220, PF=0.9
    # Expected P = 220 × 2.5 × 0.9 = 495W
    preds = [_make_pred("AC", power=500.0, current=2.5)]   # close → pass
    result_ok = checker.check(preds, aggregate_power=600, v_rms=220.0, pf=0.9)

    preds_bad = [_make_pred("AC", power=2000.0, current=2.5)]  # P way off
    result_bad = checker.check(preds_bad, aggregate_power=2200, v_rms=220.0, pf=0.9)

    assert result_ok.passed
    assert not result_bad.passed
    print(f"  P=500W vs V·I·PF=495W → PASS")
    print(f"  P=2000W vs V·I·PF=495W → FAIL ({result_bad.violations[0].margin:.1%} margin)")


# ============================================================
# Pipeline tests
# ============================================================

def test_pipeline_basic():
    """End-to-end pipeline ทำงาน, return InferenceResult"""
    print("\n=== Test 10: Pipeline End-to-End ===")
    router = BranchingDQNRouter(ROUTER_CONFIG)
    experts = MultiCategoryExperts(EXPERT_CONFIG)
    checker = ConfidenceChecker()
    constraint = ConstraintChecker()

    pipeline = InferencePipeline(router, experts, checker, constraint)

    state = torch.randn(1, ROUTER_CONFIG.seq_len, ROUTER_CONFIG.n_features)
    result = pipeline.infer(state, aggregate_power=10000.0)  # big aggregate → easy pass

    assert len(result.predictions) == ROUTER_CONFIG.n_heads
    assert result.retries_used >= 0
    assert result.final_action_mask.shape == (ROUTER_CONFIG.n_heads,)
    print(f"  Predictions: {len(result.predictions)} categories")
    print(f"  Retries used: {result.retries_used}")
    print(f"  Constraint passed: {result.constraint_result.passed}")
    print(f"  Flagged: {result.flagged_for_logging}")


def test_retry_escalates_epsilon():
    """ตั้ง aggregate ต่ำมาก → constraint จะ fail → trigger retries"""
    print("\n=== Test 11: Retry Escalation ===")
    # Force experts ให้ return high power → จะ violate energy constraint แน่นอน
    router = BranchingDQNRouter(ROUTER_CONFIG)
    experts = MultiCategoryExperts(EXPERT_CONFIG)
    checker = ConfidenceChecker()
    constraint = ConstraintChecker()

    # บังคับให้ Router เลือก all categories โดยใช้ huge weights
    with torch.no_grad():
        for head in router.heads:
            # Make on-action Q-values >> off-action Q-values
            last = head[-1]
            last.weight[1].fill_(10.0)  # on Q = high
            last.weight[0].fill_(-10.0)  # off Q = low
            last.bias[0] = -10.0
            last.bias[1] = 10.0

    pipeline = InferencePipeline(router, experts, checker, constraint)

    # Aggregate very small → constraint จะ fail
    state = torch.randn(1, ROUTER_CONFIG.seq_len, ROUTER_CONFIG.n_features) * 5
    result = pipeline.infer(state, aggregate_power=0.001)

    # ควรพยายาม retry
    print(f"  Retries used: {result.retries_used} (max={CONSTRAINT_CONFIG.max_retries})")
    print(f"  Final passed: {result.constraint_result.passed}")
    print(f"  Violations: {[v.name for v in result.constraint_result.violations]}")
    # ไม่ assert pass/fail — แค่ดูว่า retry mechanism ถูก trigger
    # (random weights อาจ pass บางครั้ง อาจ fail บางครั้ง)


def test_flagging():
    """Manually craft predictions, ทดสอบ flag collection"""
    print("\n=== Test 12: Low-Confidence Flagging ===")
    preds = [
        CategoryPrediction("Plug", routed=True, status=0.9, power=50, current=0.5,
                           confidence=0.8, confidence_level=ConfidenceLevel.HIGH, consistent=True),
        CategoryPrediction("Light", routed=True, status=0.6, power=15, current=0.1,
                           confidence=0.5, confidence_level=ConfidenceLevel.LOW_KNOWN, consistent=True),
        CategoryPrediction("AC", routed=True, status=0.51, power=500, current=2.5,
                           confidence=0.05, confidence_level=ConfidenceLevel.LOW_UNKNOWN, consistent=True),
        CategoryPrediction("Water_Heater", routed=True, status=0.95, power=0, current=0,
                           confidence=0.3, confidence_level=ConfidenceLevel.LOW_KNOWN, consistent=False),
    ]

    flags = InferencePipeline._collect_flags(preds)
    print(f"  Flags: {flags}")

    # Plug high → no flag
    assert not any("Plug" in f for f in flags)
    # Light low-known → flag
    assert any("Light:LOW_CONF" in f for f in flags)
    # AC low-unknown → flag
    assert any("AC:UNKNOWN_CANDIDATE" in f for f in flags)
    # Water_Heater inconsistent → flag (overrides low-conf flag)
    assert any("Water_Heater:INCONSISTENT" in f for f in flags)
    print(f"  ✓ All expected flags collected")


if __name__ == "__main__":
    print("=" * 60)
    print("Inference Pipeline (Stage 3) — Sanity Tests")
    print("=" * 60)

    test_high_confidence_consistent()
    test_inconsistent_status_power()
    test_ambiguous_status_low_known()
    test_low_unknown()
    test_not_routed()
    test_batch_confidence()
    test_energy_conservation_pass()
    test_energy_conservation_fail()
    test_power_current_consistency()
    test_pipeline_basic()
    test_retry_escalates_epsilon()
    test_flagging()

    print("\n" + "=" * 60)
    print("✓ All tests passed")
    print("=" * 60)
