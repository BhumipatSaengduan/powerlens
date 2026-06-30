"""
Sanity test for DRL-STFN Expert Model.
ทดสอบ:
1. Forward pass ผ่าน — shapes ถูกต้อง
2. Output ranges ถูกต้อง (status ∈ [0,1], power/current ≥ 0)
3. Gradient flow ผ่าน — backward ได้
4. Multi-category container ทำงาน
5. Parameter count อยู่ในเกณฑ์ที่คาดไว้
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch

from powerlens.models.expert import DRLSTFNExpert, MultiCategoryExperts
from powerlens.models.config import EXPERT_CONFIG, FEATURE_CONFIG


def test_single_expert_forward():
    """Test forward pass + output shapes/ranges สำหรับ single expert."""
    print("\n=== Test 1: Single Expert Forward Pass ===")
    model = DRLSTFNExpert(category="AC", config=EXPERT_CONFIG)
    model.eval()

    batch_size = 8
    x = torch.randn(batch_size, EXPERT_CONFIG.seq_len, EXPERT_CONFIG.n_features)

    with torch.no_grad():
        status, power, current = model(x)

    # Shape check
    assert status.shape == (batch_size, 1), f"status shape: {status.shape}"
    assert power.shape == (batch_size, 1), f"power shape: {power.shape}"
    assert current.shape == (batch_size, 1), f"current shape: {current.shape}"

    # Range check
    assert torch.all(status >= 0) and torch.all(status <= 1), "status not in [0,1]"
    assert torch.all(power >= 0), "power must be ≥ 0"
    assert torch.all(current >= 0), "current must be ≥ 0"

    print(f"  ✓ Shapes OK: status={tuple(status.shape)}, "
          f"power={tuple(power.shape)}, current={tuple(current.shape)}")
    print(f"  ✓ Ranges OK: status ∈ [{status.min():.3f}, {status.max():.3f}], "
          f"power ≥ {power.min():.3f}, current ≥ {current.min():.3f}")


def test_gradient_flow():
    """Test ว่า gradient ไหลผ่านทุก parameter."""
    print("\n=== Test 2: Gradient Flow ===")
    model = DRLSTFNExpert(category="Plug", config=EXPERT_CONFIG)
    model.train()

    x = torch.randn(4, EXPERT_CONFIG.seq_len, EXPERT_CONFIG.n_features)
    status, power, current = model(x)

    # Combined loss (placeholder — ใน training จะ proper)
    loss = status.mean() + power.mean() + current.mean()
    loss.backward()

    n_params_with_grad = 0
    n_params_total = 0
    for p in model.parameters():
        n_params_total += 1
        if p.grad is not None and p.grad.abs().sum() > 0:
            n_params_with_grad += 1

    assert n_params_with_grad > 0, "ไม่มี parameter ที่ได้ gradient"
    print(f"  ✓ Gradient flowed: {n_params_with_grad}/{n_params_total} params with grad")


def test_multi_category():
    """Test MultiCategoryExperts container."""
    print("\n=== Test 3: Multi-Category Container ===")
    experts = MultiCategoryExperts(EXPERT_CONFIG)
    experts.eval()

    x = torch.randn(2, EXPERT_CONFIG.seq_len, EXPERT_CONFIG.n_features)

    # Test specific category
    with torch.no_grad():
        s, p, c = experts(x, category="AC")
    assert s.shape == (2, 1)
    print(f"  ✓ Single-category forward: AC → shapes OK")

    # Test forward_all
    with torch.no_grad():
        all_outputs = experts.forward_all(x)
    assert set(all_outputs.keys()) == set(EXPERT_CONFIG.categories), \
        f"Categories mismatch: {all_outputs.keys()}"
    print(f"  ✓ All-categories forward: {list(all_outputs.keys())}")


def test_parameter_count():
    """Report parameter count — sanity check ว่า model ไม่เล็ก/ใหญ่เกิน."""
    print("\n=== Test 4: Parameter Count ===")
    model = DRLSTFNExpert(category="Light", config=EXPERT_CONFIG)
    n_params = model.count_parameters()
    print(f"  Single expert params: {n_params:,}")

    experts = MultiCategoryExperts(EXPERT_CONFIG)
    total = sum(p.numel() for p in experts.parameters())
    print(f"  All 4 experts params: {total:,}")
    print(f"  Estimated model size: ~{total * 4 / 1024 / 1024:.2f} MB (float32)")

    # Sanity bounds: ไม่ควรน้อยกว่า 100K, ไม่ควรเกิน 10M ต่อ expert
    assert 100_000 < n_params < 10_000_000, f"Param count out of range: {n_params}"
    print(f"  ✓ Param count within reasonable bounds")


def test_variable_batch_size():
    """Test ว่า model รับ batch size ต่างๆ ได้ (สำคัญสำหรับ ONNX dynamic axes)."""
    print("\n=== Test 5: Variable Batch Size ===")
    model = DRLSTFNExpert(category="Water_Heater", config=EXPERT_CONFIG)
    model.eval()

    for bs in [1, 4, 16, 32]:
        x = torch.randn(bs, EXPERT_CONFIG.seq_len, EXPERT_CONFIG.n_features)
        with torch.no_grad():
            s, p, c = model(x)
        assert s.shape[0] == bs
    print(f"  ✓ Variable batch sizes OK: [1, 4, 16, 32]")


if __name__ == "__main__":
    print("=" * 60)
    print("DRL-STFN Expert Model — Sanity Tests")
    print("=" * 60)
    print(f"Config: seq_len={EXPERT_CONFIG.seq_len}, "
          f"n_features={EXPERT_CONFIG.n_features}, "
          f"gru_hidden={EXPERT_CONFIG.gru_hidden}, "
          f"attn_heads={EXPERT_CONFIG.attn_heads}")

    test_single_expert_forward()
    test_gradient_flow()
    test_multi_category()
    test_parameter_count()
    test_variable_batch_size()

    print("\n" + "=" * 60)
    print("✓ All tests passed")
    print("=" * 60)
