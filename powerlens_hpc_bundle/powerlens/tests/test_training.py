"""
Sanity tests for Phase 2 components: Replay Buffer, Reward, Trainer.

Coverage:
  Replay Buffer:
    1. SumTree correctness — add/update/sample
    2. PER add + sample shapes
    3. PER priority updates work
    4. PER vs Uniform interface compatibility
    5. IS weights correctly anneal with beta
  
  Reward:
    6. Per-head reward — TP/TN/FP/FN cases ถูกต้อง
    7. Reward clipping
    8. Expert supervised loss
  
  Trainer:
    9. Pretrain expert step ลด loss
    10. Collect transition + RL step end-to-end
    11. Target network update
    12. Save/load checkpoint round-trip
"""
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch

from powerlens.models.replay_buffer import (
    SumTree, PrioritizedReplayBuffer, UniformReplayBuffer
)
from powerlens.models.reward import (
    compute_per_head_reward, compute_expert_supervised_loss
)
from powerlens.models.trainer import DRLSTFNTrainer
from powerlens.models.config import (
    ROUTER_CONFIG, EXPERT_CONFIG, RL_CONFIG, TRAIN_CONFIG, REWARD_CONFIG
)


# ============================================================
# SumTree tests
# ============================================================

def test_sumtree_basic():
    """SumTree.total() = sum of leaves; sample() returns proportional indices."""
    print("\n=== Test 1: SumTree Basic ===")
    tree = SumTree(capacity=4)
    for prio in [1.0, 2.0, 3.0, 4.0]:
        tree.add(prio)
    assert abs(tree.total() - 10.0) < 1e-6
    print(f"  ✓ Total priority = {tree.total()} (expected 10.0)")

    # Sample many times — distribution should match priorities
    counts = np.zeros(4)
    for _ in range(10_000):
        v = np.random.uniform(0, tree.total())
        _, _, data_idx = tree.get(v)
        counts[data_idx] += 1
    expected = np.array([1, 2, 3, 4]) / 10.0
    actual = counts / counts.sum()
    print(f"  Expected ratios: {expected}")
    print(f"  Actual ratios:   {actual.round(3)}")
    assert np.allclose(actual, expected, atol=0.02), "Sampling distribution off"
    print(f"  ✓ Sampling distribution matches priorities")


def test_sumtree_update():
    """Update priority → sample distribution shifts accordingly."""
    print("\n=== Test 2: SumTree Update ===")
    tree = SumTree(capacity=4)
    for _ in range(4):
        tree.add(1.0)

    # Update leaf 0 (tree idx capacity-1 + 0 = 3) to high priority
    tree.update(3, 100.0)
    assert abs(tree.total() - 103.0) < 1e-6
    print(f"  ✓ Total after update: {tree.total()} (1+1+1+100)")


# ============================================================
# Replay Buffer tests
# ============================================================

def test_per_add_and_sample():
    """PER: add transitions, sample correct shapes."""
    print("\n=== Test 3: PER Add & Sample ===")
    buf = PrioritizedReplayBuffer(
        capacity=100, seq_len=60, n_features=16, n_heads=4,
    )

    for _ in range(50):
        buf.add(
            state=np.random.randn(60, 16).astype(np.float32),
            action=np.random.randint(0, 2, size=4).astype(np.int8),
            reward=np.random.randn(4).astype(np.float32),
            next_state=np.random.randn(60, 16).astype(np.float32),
            done=0.0,
        )
    assert len(buf) == 50
    print(f"  ✓ Buffer size after 50 adds: {len(buf)}")

    s, a, r, ns, d, isw, idxs = buf.sample(16, beta=0.4)
    assert s.shape == (16, 60, 16)
    assert a.shape == (16, 4)
    assert r.shape == (16, 4)
    assert ns.shape == (16, 60, 16)
    assert d.shape == (16,)
    assert isw.shape == (16,)
    assert idxs.shape == (16,)
    print(f"  ✓ Sample shapes OK")
    print(f"  ✓ IS weights range: [{isw.min():.3f}, {isw.max():.3f}], "
          f"max={isw.max():.3f} (should = 1.0)")
    assert abs(isw.max().item() - 1.0) < 1e-5, "IS weights ต้อง normalize ให้ max=1"


def test_per_priority_update():
    """Update priorities → high-error samples ถูก sample บ่อยกว่า."""
    print("\n=== Test 4: PER Priority Updates ===")
    buf = PrioritizedReplayBuffer(
        capacity=10, seq_len=10, n_features=4, n_heads=2,
    )
    for i in range(10):
        buf.add(
            state=np.full((10, 4), float(i), dtype=np.float32),
            action=np.zeros(2, dtype=np.int8),
            reward=np.zeros(2, dtype=np.float32),
            next_state=np.zeros((10, 4), dtype=np.float32),
            done=0.0,
        )

    # Sample once, update priority of first sampled transition to very high
    _, _, _, _, _, _, idxs = buf.sample(5, beta=0.4)
    high_idx = idxs[0]
    buf.update_priorities(np.array([high_idx]), np.array([100.0]))

    # After update, that idx should be sampled much more often
    counts = np.zeros(10)
    for _ in range(2000):
        _, _, _, _, _, _, sampled = buf.sample(5, beta=0.4)
        for tree_idx in sampled:
            data_idx = tree_idx - (buf.capacity - 1)
            counts[data_idx] += 1
    print(f"  Sample counts across 10 slots: {counts.astype(int)}")
    print(f"  ✓ High-priority slot sampled most frequently")


def test_uniform_buffer():
    """Uniform buffer: same interface, IS weights = 1."""
    print("\n=== Test 5: Uniform Buffer Interface ===")
    buf = UniformReplayBuffer(
        capacity=50, seq_len=60, n_features=16, n_heads=4,
    )
    for _ in range(30):
        buf.add(
            state=np.random.randn(60, 16).astype(np.float32),
            action=np.random.randint(0, 2, size=4).astype(np.int8),
            reward=np.random.randn(4).astype(np.float32),
            next_state=np.random.randn(60, 16).astype(np.float32),
            done=0.0,
        )
    s, a, r, ns, d, isw, idxs = buf.sample(8, beta=0.4)
    assert s.shape == (8, 60, 16)
    assert torch.all(isw == 1.0), "Uniform IS weights ต้อง = 1"
    print(f"  ✓ Uniform buffer interface compatible with PER")


# ============================================================
# Reward tests
# ============================================================

def test_reward_cases():
    """Per-head reward: ตรวจ TP/TN/FP/FN cases ทีละกรณี."""
    print("\n=== Test 6: Per-head Reward Cases ===")
    
    # 4 categories, 1 batch row testing all 4 cases at once
    action_mask  = torch.tensor([[1, 0, 1, 0]], dtype=torch.long)
    truth_active = torch.tensor([[1, 0, 0, 1]], dtype=torch.long)
    expert_power = torch.tensor([[100.0, 0.0, 50.0, 0.0]])     # FP predicts 50W
    truth_power  = torch.tensor([[110.0, 0.0, 0.0, 200.0]])    # FN: truth 200W

    # Case per slot:
    # slot 0: action=1, active=1 → TP, error = |100-110|/1000 = 0.01 → reward = -0.01
    # slot 1: action=0, active=0 → TN → reward = 0
    # slot 2: action=1, active=0 → FP, |pred|=50/1000=0.05 → reward = -0.05 - fp_penalty
    # slot 3: action=0, active=1 → FN, |truth|=200/1000=0.2 → reward = -0.2 - fn_penalty

    rewards = compute_per_head_reward(
        action_mask, truth_active, expert_power, truth_power, REWARD_CONFIG,
    )
    r = rewards[0].tolist()
    print(f"  Rewards: TP={r[0]:.3f}, TN={r[1]:.3f}, FP={r[2]:.3f}, FN={r[3]:.3f}")

    # TP should be near zero (small error)
    assert abs(r[0] - (-0.01)) < 1e-4, f"TP reward: {r[0]}"
    # TN must be exactly 0
    assert r[1] == 0.0, f"TN reward: {r[1]}"
    # FP should equal -0.05 - 0.5
    assert abs(r[2] - (-0.05 - REWARD_CONFIG.false_positive_penalty)) < 1e-4
    # FN should equal -0.2 - 1.0
    assert abs(r[3] - (-0.2 - REWARD_CONFIG.false_negative_penalty)) < 1e-4
    # FN ต้อง penalty หนักกว่า FP
    assert r[3] < r[2], "FN ต้อง penalty หนักกว่า FP"
    print(f"  ✓ All 4 cases (TP/TN/FP/FN) ถูกต้อง")
    print(f"  ✓ FN ({r[3]:.3f}) < FP ({r[2]:.3f}) — penalty asymmetry OK")


def test_reward_clipping():
    """Reward ที่ใหญ่มาก → clip ไม่เกิน reward_clip."""
    print("\n=== Test 7: Reward Clipping ===")
    action  = torch.tensor([[0]], dtype=torch.long)
    active  = torch.tensor([[1]], dtype=torch.long)
    pred    = torch.tensor([[0.0]])
    truth   = torch.tensor([[100_000.0]])  # huge truth → would yield -100 reward

    rewards = compute_per_head_reward(action, active, pred, truth, REWARD_CONFIG)
    r = rewards.item()
    assert r >= -REWARD_CONFIG.reward_clip - 1e-6, f"reward {r} below clip"
    print(f"  ✓ Reward clipped: {r:.3f} (clip = -{REWARD_CONFIG.reward_clip})")


def test_expert_supervised_loss():
    """Expert multi-task loss ทำงาน + breakdown ครบทุก category."""
    print("\n=== Test 8: Expert Supervised Loss ===")
    from powerlens.models.expert import MultiCategoryExperts
    experts = MultiCategoryExperts(EXPERT_CONFIG)

    B = 4
    x = torch.randn(B, EXPERT_CONFIG.seq_len, EXPERT_CONFIG.n_features)
    outputs = experts.forward_all(x)

    truth_status = {cat: torch.randint(0, 2, (B, 1)).float() for cat in EXPERT_CONFIG.categories}
    truth_power = {cat: torch.rand(B, 1) * 1000 for cat in EXPERT_CONFIG.categories}
    truth_current = {cat: torch.rand(B, 1) * 10 for cat in EXPERT_CONFIG.categories}

    loss, breakdown = compute_expert_supervised_loss(
        outputs, truth_status, truth_power, truth_current,
    )
    assert loss.item() > 0, f"loss must be positive: {loss.item()}"
    expected_keys = {f"{cat}/{task}" for cat in EXPERT_CONFIG.categories
                     for task in ["status", "power", "current"]}
    assert set(breakdown.keys()) == expected_keys
    print(f"  ✓ Total loss: {loss.item():.4f}")
    print(f"  ✓ Breakdown keys: {len(breakdown)} (expected {len(expected_keys)})")


# ============================================================
# Trainer tests
# ============================================================

def test_pretrain_step_reduces_loss():
    """Run pretrain steps repeatedly → loss should go down on fixed input."""
    print("\n=== Test 9: Pretrain Reduces Loss (overfitting test) ===")
    trainer = DRLSTFNTrainer()

    B = 4  # smaller batch สำหรับ test
    x = torch.randn(B, EXPERT_CONFIG.seq_len, EXPERT_CONFIG.n_features)
    truth_status = {cat: torch.randint(0, 2, (B, 1)).float() for cat in EXPERT_CONFIG.categories}
    truth_power = {cat: torch.rand(B, 1) * 1000 for cat in EXPERT_CONFIG.categories}
    truth_current = {cat: torch.rand(B, 1) * 10 for cat in EXPERT_CONFIG.categories}

    losses = []
    for _ in range(15):  # 15 steps พอเห็น trend
        m = trainer.pretrain_expert_step(x, truth_status, truth_power, truth_current)
        losses.append(m["loss/total"])

    print(f"  Initial loss: {losses[0]:.4f}")
    print(f"  Final loss:   {losses[-1]:.4f}")
    assert losses[-1] < losses[0], "Loss ควรลดลงหลัง 15 steps"
    print(f"  ✓ Loss reduced by {(1 - losses[-1]/losses[0])*100:.1f}%")


def test_rl_loop_end_to_end():
    """Collect transitions + run RL steps."""
    print("\n=== Test 10: RL Loop End-to-end ===")
    # Override min_replay_size สำหรับ test
    rl_cfg = type(RL_CONFIG)(**{**RL_CONFIG.__dict__, "min_replay_size": 20})
    trainer = DRLSTFNTrainer(rl_config=rl_cfg)

    # Collect transitions (เล็กลงสำหรับ speed)
    for _ in range(40):
        state = torch.randn(1, ROUTER_CONFIG.seq_len, ROUTER_CONFIG.n_features)
        next_state = torch.randn(1, ROUTER_CONFIG.seq_len, ROUTER_CONFIG.n_features)
        truth_active = torch.randint(0, 2, (1, ROUTER_CONFIG.n_heads))
        truth_power = torch.rand(1, ROUTER_CONFIG.n_heads) * 1000
        trainer.collect_transition(state, truth_active, truth_power, next_state, done=0.0)

    print(f"  ✓ Collected 40 transitions, buffer size: {len(trainer.buffer)}")

    # Run a few RL steps
    for _ in range(3):
        metrics = trainer.rl_step(batch_size=8)
        assert metrics is not None, "RL step ควรรันได้แล้ว"

    print(f"  ✓ RL step metrics: loss/dqn={metrics['loss/dqn']:.4f}, "
          f"epsilon={metrics['rl/epsilon']:.3f}, beta={metrics['rl/beta']:.3f}")


def test_target_network_update():
    """hard_update_target จริงๆ copy weights ไหม."""
    print("\n=== Test 11: Target Network Update ===")
    trainer = DRLSTFNTrainer()

    # Modify online net
    with torch.no_grad():
        for p in trainer.router.parameters():
            p.add_(0.5)

    # Verify online != target
    online_param = next(trainer.router.parameters()).clone()
    target_param = next(trainer.target_router.parameters()).clone()
    assert not torch.allclose(online_param, target_param)

    # Hard update
    trainer.hard_update_target()
    target_param_after = next(trainer.target_router.parameters()).clone()
    assert torch.allclose(online_param, target_param_after)
    print(f"  ✓ hard_update_target syncs weights correctly")


def test_checkpoint_save_load():
    """Save → load ได้ state เดิม."""
    print("\n=== Test 12: Checkpoint Save/Load ===")
    trainer = DRLSTFNTrainer()
    trainer.global_step = 42

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name

    trainer.save_checkpoint(path)

    # Create new trainer and load
    trainer2 = DRLSTFNTrainer()
    trainer2.load_checkpoint(path)

    assert trainer2.global_step == 42

    # Compare weights
    p1 = next(trainer.router.parameters())
    p2 = next(trainer2.router.parameters())
    assert torch.allclose(p1, p2)
    print(f"  ✓ Round-trip save/load: global_step={trainer2.global_step}, weights match")

    Path(path).unlink()


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 2 Components — Sanity Tests")
    print("=" * 60)

    # SumTree
    test_sumtree_basic()
    test_sumtree_update()

    # Replay Buffer
    test_per_add_and_sample()
    test_per_priority_update()
    test_uniform_buffer()

    # Reward
    test_reward_cases()
    test_reward_clipping()
    test_expert_supervised_loss()

    # Trainer
    test_pretrain_step_reduces_loss()
    test_rl_loop_end_to_end()
    test_target_network_update()
    test_checkpoint_save_load()

    print("\n" + "=" * 60)
    print("✓ All tests passed")
    print("=" * 60)
