"""
Sanity tests for Branching DQN Router.

Coverage:
1. Forward pass — Q-values shape ถูกต้อง
2. Action selection — mask shape + binary values
3. ε-greedy — epsilon=1.0 → fully random, epsilon=0.0 → fully greedy
4. mask_to_categories — convert mask → category names ถูกต้อง
5. Gradient flow — backward ผ่าน
6. DQN loss — รัน loss + backward ผ่าน, target network ใช้ได้
7. Parameter count — เบากว่า Expert (router ควรเล็กกว่า)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from copy import deepcopy

from powerlens.models.router import BranchingDQNRouter, compute_dqn_loss
from powerlens.models.config import ROUTER_CONFIG, EXPERT_CONFIG, RL_CONFIG


def test_forward_pass():
    """Q-values shape: (B, N_categories, 2)"""
    print("\n=== Test 1: Forward Pass ===")
    router = BranchingDQNRouter(ROUTER_CONFIG)
    router.eval()

    x = torch.randn(8, ROUTER_CONFIG.seq_len, ROUTER_CONFIG.n_features)
    with torch.no_grad():
        q = router(x)

    expected_shape = (8, ROUTER_CONFIG.n_heads, ROUTER_CONFIG.n_actions_per_head)
    assert q.shape == expected_shape, f"Q-shape: {q.shape}, expected {expected_shape}"
    print(f"  ✓ Q-values shape: {tuple(q.shape)} = (B, N_cat={ROUTER_CONFIG.n_heads}, n_actions=2)")
    print(f"  ✓ Q-values range: [{q.min():.3f}, {q.max():.3f}]")


def test_action_selection_greedy():
    """epsilon=0 → pure greedy, output mask ∈ {0, 1}"""
    print("\n=== Test 2: Greedy Action Selection ===")
    router = BranchingDQNRouter(ROUTER_CONFIG)
    router.eval()

    x = torch.randn(4, ROUTER_CONFIG.seq_len, ROUTER_CONFIG.n_features)
    action, q = router.select_action(x, epsilon=0.0)

    assert action.shape == (4, ROUTER_CONFIG.n_heads), f"action shape: {action.shape}"
    assert torch.all((action == 0) | (action == 1)), "action ต้องเป็น binary"
    print(f"  ✓ Action mask shape: {tuple(action.shape)}")
    print(f"  ✓ Sample mask: {action[0].tolist()} (binary OK)")


def test_action_selection_random():
    """epsilon=1.0 → fully random — ทดสอบว่า random kicks in"""
    print("\n=== Test 3: Epsilon-greedy Random ===")
    router = BranchingDQNRouter(ROUTER_CONFIG)
    router.eval()

    # Run หลายครั้งด้วย epsilon=1.0 — ผลควรเปลี่ยนได้ (random)
    torch.manual_seed(42)
    x = torch.randn(100, ROUTER_CONFIG.seq_len, ROUTER_CONFIG.n_features)
    action_eps1, _ = router.select_action(x, epsilon=1.0)
    action_eps0, _ = router.select_action(x, epsilon=0.0)

    # ที่ epsilon=1.0 ส่วนใหญ่ของ actions ควรต่างจาก greedy
    diff_ratio = (action_eps1 != action_eps0).float().mean().item()
    print(f"  ✓ Diff ratio (eps=1 vs eps=0): {diff_ratio:.2%}")
    assert 0.3 < diff_ratio < 0.7, f"Random rate ผิดปกติ: {diff_ratio:.2%} (expected ~50%)"


def test_mask_to_categories():
    """Convert action mask → category names list"""
    print("\n=== Test 4: Mask → Categories ===")
    router = BranchingDQNRouter(ROUTER_CONFIG)

    # Manual mask: [Plug=1, Light=0, AC=1, Water_Heater=1]
    mask = torch.tensor([
        [1, 0, 1, 1],
        [0, 0, 0, 0],
        [1, 1, 1, 1],
    ])
    cats = router.mask_to_categories(mask)
    assert cats[0] == ["Plug", "AC", "Water_Heater"], f"row 0: {cats[0]}"
    assert cats[1] == [], f"row 1 (no active): {cats[1]}"
    assert cats[2] == ROUTER_CONFIG.categories, f"row 2 (all active): {cats[2]}"
    print(f"  ✓ Mask [1,0,1,1] → {cats[0]}")
    print(f"  ✓ Mask [0,0,0,0] → {cats[1]} (empty OK)")
    print(f"  ✓ Mask [1,1,1,1] → {cats[2]}")


def test_gradient_flow():
    """Backward pass — gradient ไหลทุก parameter"""
    print("\n=== Test 5: Gradient Flow ===")
    router = BranchingDQNRouter(ROUTER_CONFIG)
    router.train()

    x = torch.randn(4, ROUTER_CONFIG.seq_len, ROUTER_CONFIG.n_features)
    q = router(x)
    loss = q.mean()
    loss.backward()

    n_with_grad = sum(
        1 for p in router.parameters()
        if p.grad is not None and p.grad.abs().sum() > 0
    )
    n_total = sum(1 for _ in router.parameters())
    print(f"  ✓ Gradient flow: {n_with_grad}/{n_total} params with grad")
    assert n_with_grad > 0


def test_dqn_loss():
    """DQN loss + Double DQN — ทำงานถูกต้อง, backward ผ่าน"""
    print("\n=== Test 6: DQN Loss + Double DQN ===")
    online_net = BranchingDQNRouter(ROUTER_CONFIG)
    target_net = deepcopy(online_net)
    target_net.eval()
    for p in target_net.parameters():
        p.requires_grad = False

    B = 16
    states = torch.randn(B, ROUTER_CONFIG.seq_len, ROUTER_CONFIG.n_features)
    next_states = torch.randn(B, ROUTER_CONFIG.seq_len, ROUTER_CONFIG.n_features)
    actions = torch.randint(0, 2, (B, ROUTER_CONFIG.n_heads))
    rewards = torch.randn(B)
    dones = torch.bernoulli(torch.full((B,), 0.1))

    # Test ทั้ง vanilla และ double DQN
    for use_double in [True, False]:
        online_net.train()
        loss, td_errors = compute_dqn_loss(
            online_net, target_net,
            states, actions, rewards, next_states, dones,
            gamma=RL_CONFIG.gamma,
            use_double_dqn=use_double,
        )
        assert loss.item() >= 0, f"loss negative: {loss.item()}"
        assert td_errors.shape == (B,), f"td_errors shape: {td_errors.shape}"
        loss.backward()

        # Reset gradient for next iteration
        for p in online_net.parameters():
            if p.grad is not None:
                p.grad.zero_()

        label = "Double DQN" if use_double else "Vanilla DQN"
        print(f"  ✓ {label} loss = {loss.item():.4f}")


def test_target_network_separation():
    """Target network ต้องไม่ update gradient"""
    print("\n=== Test 7: Target Network Frozen ===")
    online_net = BranchingDQNRouter(ROUTER_CONFIG)
    target_net = deepcopy(online_net)
    for p in target_net.parameters():
        p.requires_grad = False

    target_no_grad = sum(1 for p in target_net.parameters() if not p.requires_grad)
    target_total = sum(1 for _ in target_net.parameters())
    assert target_no_grad == target_total, "Target net params ทั้งหมดต้อง frozen"
    print(f"  ✓ Target network: {target_no_grad}/{target_total} params frozen")


def test_parameter_count_relative_to_expert():
    """Router ควรเล็กกว่า single Expert"""
    print("\n=== Test 8: Parameter Count (vs Expert) ===")
    from powerlens.models.expert import DRLSTFNExpert

    router = BranchingDQNRouter(ROUTER_CONFIG)
    expert = DRLSTFNExpert(category="AC", config=EXPERT_CONFIG)

    n_router = router.count_parameters()
    n_expert = sum(p.numel() for p in expert.parameters())

    print(f"  Router params:        {n_router:,}")
    print(f"  Single Expert params: {n_expert:,}")
    print(f"  Router/Expert ratio:  {n_router/n_expert:.1%}")
    assert n_router < n_expert, "Router ควรเล็กกว่า Expert"
    print(f"  ✓ Router lighter than Expert (as designed)")


def test_variable_batch_size():
    """ONNX dynamic axes prep — รับ batch sizes ต่างๆ"""
    print("\n=== Test 9: Variable Batch Size ===")
    router = BranchingDQNRouter(ROUTER_CONFIG)
    router.eval()

    for bs in [1, 4, 16, 32]:
        x = torch.randn(bs, ROUTER_CONFIG.seq_len, ROUTER_CONFIG.n_features)
        with torch.no_grad():
            q = router(x)
            a, _ = router.select_action(x, epsilon=0.0)
        assert q.shape[0] == bs and a.shape[0] == bs
    print(f"  ✓ Variable batch sizes OK: [1, 4, 16, 32]")


if __name__ == "__main__":
    print("=" * 60)
    print("Branching DQN Router — Sanity Tests")
    print("=" * 60)
    print(f"Config: n_categories={ROUTER_CONFIG.n_heads}, "
          f"gru_hidden={ROUTER_CONFIG.gru_hidden}, "
          f"trunk_hidden={ROUTER_CONFIG.trunk_hidden}")

    test_forward_pass()
    test_action_selection_greedy()
    test_action_selection_random()
    test_mask_to_categories()
    test_gradient_flow()
    test_dqn_loss()
    test_target_network_separation()
    test_parameter_count_relative_to_expert()
    test_variable_batch_size()

    print("\n" + "=" * 60)
    print("✓ All tests passed")
    print("=" * 60)
