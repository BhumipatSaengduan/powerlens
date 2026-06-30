"""
Sanity tests for Fusion & Constraints Postprocessing (Stage 4).

Coverage:
  Decision Fusion:
    1. Active categories pass all gates
    2. Low confidence → not active (flagged)
    3. Inconsistent → not active
    4. LOW_UNKNOWN → not active (will be Phase 2 input)
  
  Energy Rebalancing:
    5. Within budget — no rebalance
    6. Within tolerance — no rebalance
    7. Over budget — confidence-weighted scaling
    8. Low-conf categories take bigger cut than high-conf
  
  Residual Allocation:
    9. Sum >= aggregate → no residual
    10. Sum << aggregate → "Other" added with residual
    11. Disabled in config → no "Other" appended
  
  End-to-end:
    12. InferencePipeline + FusionProcessor → JSON-serializable output
    13. Output schema completeness
"""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from dataclasses import replace

from powerlens.models.fusion import FusionProcessor, FusionOutput, CategoryOutput
from powerlens.models.inference import (
    CategoryPrediction, ConfidenceLevel, ConstraintResult, ConstraintViolation,
    ConfidenceChecker, ConstraintChecker, InferencePipeline,
)
from powerlens.models.router import BranchingDQNRouter
from powerlens.models.expert import MultiCategoryExperts
from powerlens.models.config import (
    FUSION_CONFIG, ROUTER_CONFIG, EXPERT_CONFIG, CONFIDENCE_CONFIG,
)


def _make_pred(cat, status=0.9, power=500.0, current=2.5,
               confidence=0.85, level=None, consistent=True, routed=True):
    """Helper for clear test fixtures."""
    if level is None:
        level = (
            ConfidenceLevel.HIGH if confidence >= 0.7
            else ConfidenceLevel.LOW_KNOWN if confidence >= 0.3
            else ConfidenceLevel.LOW_UNKNOWN
        )
    return CategoryPrediction(
        category=cat, routed=routed, status=status,
        power=power, current=current, confidence=confidence,
        confidence_level=level, consistent=consistent,
    )


def _ok_constraint():
    return ConstraintResult(passed=True, violations=[])


# ============================================================
# Decision Fusion tests
# ============================================================

def test_active_predictions_pass():
    """High confidence + consistent + status>0.5 → active in output"""
    print("\n=== Test 1: Active Predictions Pass ===")
    fusion = FusionProcessor()
    preds = [
        _make_pred("Plug", power=50, confidence=0.85),
        _make_pred("AC", power=500, confidence=0.9),
    ]
    out = fusion.process(preds, aggregate_power=10000, constraint_result=_ok_constraint(), flags=[])
    active = [c.category for c in out.categories if c.active and c.category != "Other"]
    print(f"  Active: {active}")
    assert "Plug" in active and "AC" in active


def test_low_confidence_inactive():
    """Confidence < min_confidence_to_activate → not active (still in output, flagged)"""
    print("\n=== Test 2: Low Confidence Inactive ===")
    fusion = FusionProcessor()
    preds = [
        _make_pred("AC", power=500, confidence=0.85),
        _make_pred("Plug", power=50, confidence=0.15,  # < 0.3 → inactive
                   level=ConfidenceLevel.LOW_UNKNOWN),
    ]
    out = fusion.process(preds, aggregate_power=10000, constraint_result=_ok_constraint(), flags=[])
    active_cats = {c.category for c in out.categories if c.active and c.category != "Other"}
    print(f"  Active: {active_cats}")
    assert "AC" in active_cats
    assert "Plug" not in active_cats


def test_inconsistent_inactive():
    """status=on but power=0 (inconsistent) → not active"""
    print("\n=== Test 3: Inconsistent Inactive ===")
    fusion = FusionProcessor()
    preds = [
        _make_pred("AC", status=0.9, power=0, confidence=0.27, consistent=False),
    ]
    out = fusion.process(preds, aggregate_power=1000, constraint_result=_ok_constraint(), flags=[])
    active = [c for c in out.categories if c.active and c.category != "Other"]
    assert len(active) == 0
    print(f"  ✓ Inconsistent prediction excluded from active")


def test_low_unknown_inactive():
    """LOW_UNKNOWN → not active (will be Auto-Calibrate input in Phase 2)"""
    print("\n=== Test 4: LOW_UNKNOWN Not Active ===")
    fusion = FusionProcessor()
    preds = [
        _make_pred("AC", confidence=0.05, level=ConfidenceLevel.LOW_UNKNOWN),
    ]
    out = fusion.process(preds, aggregate_power=1000, constraint_result=_ok_constraint(), flags=[])
    assert all(not c.active or c.category == "Other" for c in out.categories)
    print(f"  ✓ LOW_UNKNOWN excluded — handled by Phase 2 path")


# ============================================================
# Energy Rebalancing tests
# ============================================================

def test_within_budget_no_rebalance():
    """Σ < aggregate → no rebalance, factor = 1.0"""
    print("\n=== Test 5: Within Budget — No Rebalance ===")
    fusion = FusionProcessor()
    preds = [_make_pred("Plug", power=50), _make_pred("AC", power=500)]
    out = fusion.process(preds, aggregate_power=600, constraint_result=_ok_constraint(), flags=[])
    print(f"  Σ=550, aggregate=600 → rebalanced={out.rebalanced}")
    assert not out.rebalanced
    assert out.rebalance_factor == 1.0


def test_within_tolerance_no_rebalance():
    """Σ slightly over aggregate but ≤ tolerance → no rebalance"""
    print("\n=== Test 6: Within Tolerance — No Rebalance ===")
    fusion = FusionProcessor()
    # tolerance = 5% → 1000 + 5% = 1050 upper bound
    preds = [_make_pred("AC", power=520), _make_pred("Plug", power=510)]  # sum=1030
    out = fusion.process(preds, aggregate_power=1000, constraint_result=_ok_constraint(), flags=[])
    print(f"  Σ=1030, aggregate=1000, tolerance=5% (≤1050) → rebalanced={out.rebalanced}")
    assert not out.rebalanced


def test_over_budget_rebalance():
    """Σ way over → rebalance triggered, sum should be ≤ target_upper"""
    print("\n=== Test 7: Over Budget — Rebalance Triggered ===")
    fusion = FusionProcessor()
    preds = [
        _make_pred("AC", power=2000, confidence=0.9),
        _make_pred("Water_Heater", power=2000, confidence=0.9),
    ]
    out = fusion.process(preds, aggregate_power=1000, constraint_result=_ok_constraint(), flags=[])
    sum_after = sum(c.power for c in out.categories if c.active and c.category != "Other")
    target_upper = 1000 * (1 + FUSION_CONFIG.overshoot_tolerance)
    print(f"  Before: Σ=4000, aggregate=1000 (upper bound={target_upper})")
    print(f"  After:  Σ={sum_after:.1f}, rebalance_factor={out.rebalance_factor}")
    assert out.rebalanced
    assert sum_after <= target_upper + 1.0  # allow small float tolerance


def test_confidence_weighted_cut():
    """Low-confidence category should take bigger relative cut than high-confidence"""
    print("\n=== Test 8: Confidence-weighted Cut ===")
    fusion = FusionProcessor()
    # Both 1000W, but AC very confident vs Plug barely confident
    preds = [
        _make_pred("AC", power=1000, confidence=0.95),
        _make_pred("Plug", power=1000, confidence=0.35),
    ]
    out = fusion.process(preds, aggregate_power=500, constraint_result=_ok_constraint(), flags=[])
    pwr = {c.category: c.power for c in out.categories if c.active and c.category != "Other"}
    print(f"  Original: AC=1000 (conf=0.95), Plug=1000 (conf=0.35)")
    print(f"  After: AC={pwr.get('AC', 0):.1f}, Plug={pwr.get('Plug', 0):.1f}")
    # AC ควรเหลือ power มากกว่า Plug
    assert pwr.get("AC", 0) > pwr.get("Plug", 0), \
        "High-confidence category ต้องเหลือ power มากกว่า"
    print(f"  ✓ AC retained more power (high conf), Plug took bigger cut")


# ============================================================
# Residual tests
# ============================================================

def test_no_residual_full_coverage():
    """sum near aggregate → no residual (within threshold)"""
    print("\n=== Test 9: No Residual (Full Coverage) ===")
    fusion = FusionProcessor()
    preds = [_make_pred("AC", power=950)]
    out = fusion.process(preds, aggregate_power=1000, constraint_result=_ok_constraint(), flags=[])
    other = [c for c in out.categories if c.category == "Other"]
    print(f"  Σ=950, aggregate=1000 (threshold=10% → 100W) → Other: {other}")
    assert len(other) == 0  # 50W gap < 100W threshold


def test_residual_under_coverage():
    """sum << aggregate → 'Other' added"""
    print("\n=== Test 10: Residual — Under Coverage ===")
    fusion = FusionProcessor()
    preds = [_make_pred("AC", power=300)]
    out = fusion.process(preds, aggregate_power=1000, constraint_result=_ok_constraint(), flags=[])
    other = [c for c in out.categories if c.category == "Other"]
    assert len(other) == 1
    print(f"  Σ=300, aggregate=1000 → residual={other[0].power}W in 'Other'")
    assert abs(other[0].power - 700) < 1.0


def test_residual_disabled():
    """enable_residual=False → no 'Other' even when under-budget"""
    print("\n=== Test 11: Residual Disabled ===")
    cfg = replace(FUSION_CONFIG, enable_residual=False)
    fusion = FusionProcessor(cfg)
    preds = [_make_pred("AC", power=300)]
    out = fusion.process(preds, aggregate_power=1000, constraint_result=_ok_constraint(), flags=[])
    other = [c for c in out.categories if c.category == "Other"]
    assert len(other) == 0
    print(f"  ✓ enable_residual=False → no 'Other' category")


# ============================================================
# End-to-end with InferencePipeline
# ============================================================

def test_pipeline_with_fusion():
    """Full pipeline: Router + Experts + Confidence + Constraint + Fusion"""
    print("\n=== Test 12: End-to-End with Fusion ===")
    router = BranchingDQNRouter(ROUTER_CONFIG)
    experts = MultiCategoryExperts(EXPERT_CONFIG)
    pipeline = InferencePipeline(
        router=router,
        experts=experts,
        confidence_checker=ConfidenceChecker(),
        constraint_checker=ConstraintChecker(),
        fusion_processor=FusionProcessor(),
    )

    state = torch.randn(1, ROUTER_CONFIG.seq_len, ROUTER_CONFIG.n_features)
    result = pipeline.infer(state, aggregate_power=2000.0, v_rms=220.0, pf=0.9)

    assert result.fusion_output is not None
    out = result.fusion_output
    print(f"  Constraint passed: {out.constraint_passed}")
    print(f"  Aggregate: {out.aggregate_power}W")
    print(f"  Sum disaggregated: {out.sum_disaggregated}W")
    print(f"  Residual: {out.residual}W")
    print(f"  Rebalanced: {out.rebalanced} (factor={out.rebalance_factor})")
    print(f"  Categories: {len(out.categories)} (expected ≥4)")
    assert len(out.categories) >= ROUTER_CONFIG.n_heads
    print(f"  ✓ Full pipeline produces FusionOutput")


def test_output_json_serializable():
    """FusionOutput.to_dict() → JSON-serializable for MQTT"""
    print("\n=== Test 13: Output JSON Schema ===")
    fusion = FusionProcessor()
    preds = [
        _make_pred("Plug", power=50, current=0.5, confidence=0.85),
        _make_pred("AC", power=500, current=2.5, confidence=0.9),
    ]
    out = fusion.process(
        preds, aggregate_power=600,
        constraint_result=_ok_constraint(),
        flags=["AC:LOW_CONF"],
        timestamp="2026-04-29T12:00:00+00:00",
    )

    d = out.to_dict()
    js = json.dumps(d, indent=2)  # ต้อง serialize ได้
    print(f"  JSON keys: {list(d.keys())}")
    print(f"  metadata keys: {list(d['metadata'].keys())}")

    # Schema completeness
    assert "timestamp" in d
    assert "aggregate_power_w" in d
    assert "sum_disaggregated_w" in d
    assert "residual_w" in d
    assert "categories" in d
    assert "metadata" in d
    assert d["metadata"]["flags"] == ["AC:LOW_CONF"]

    # Per-category schema
    cat = d["categories"][0]
    for k in ["category", "active", "power", "current", "confidence", "raw_power"]:
        assert k in cat, f"Missing key: {k}"
    print(f"  ✓ Schema complete, JSON length={len(js)} bytes")


if __name__ == "__main__":
    print("=" * 60)
    print("Fusion & Postprocessing (Stage 4) — Sanity Tests")
    print("=" * 60)

    test_active_predictions_pass()
    test_low_confidence_inactive()
    test_inconsistent_inactive()
    test_low_unknown_inactive()
    test_within_budget_no_rebalance()
    test_within_tolerance_no_rebalance()
    test_over_budget_rebalance()
    test_confidence_weighted_cut()
    test_no_residual_full_coverage()
    test_residual_under_coverage()
    test_residual_disabled()
    test_pipeline_with_fusion()
    test_output_json_serializable()

    print("\n" + "=" * 60)
    print("✓ All tests passed")
    print("=" * 60)
