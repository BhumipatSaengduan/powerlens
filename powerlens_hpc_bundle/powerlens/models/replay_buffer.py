"""
Prioritized Experience Replay (PER) Buffer
===========================================
Replay buffer สำหรับ DRL training — sample transitions ตาม TD error magnitude
แทน uniform sampling เพื่อ improve sample efficiency

Reference:
    Schaul et al., "Prioritized Experience Replay" (ICLR 2016)

Implementation:
    - SumTree data structure → O(log N) sample + update priority
    - α (alpha): priority exponent — แค่ไหนใช้ priority (0=uniform, 1=greedy)
    - β (beta):  importance sampling correction — ลด bias ของ non-uniform sampling
                 anneal จาก beta_start → 1.0 ระหว่าง training
    - ε (epsilon): minimum priority — ป้องกัน priority = 0 (เคย sample แล้วไม่เคย sample อีก)

Storage strategy:
    - State/next_state: float32 (B, T, F) — ใช้ memory เยอะ ระวัง buffer size
    - 100k transitions × 60 × 16 × 4 bytes × 2 (state + next_state) ≈ 750 MB
    - ถ้า memory จำกัด → ใช้ float16 ลดครึ่งหนึ่ง หรือ shrink buffer
"""
from typing import Tuple
import numpy as np
import torch


class SumTree:
    """
    Binary tree structure ที่เก็บ sum ของ priorities ใน internal nodes.
    
    O(log N) operations:
    - update(idx, priority)
    - sample(value) → returns leaf idx with probability proportional to priority
    - total() → sum of all priorities
    
    Tree layout (capacity=4 example):
        idx 0 = total root
        idx 1, 2 = internal sums
        idx 3, 4, 5, 6 = leaves (data slots)
    
    Leaf idx → data idx mapping: data_idx = leaf_idx - (capacity - 1)
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.write_ptr = 0
        self.size = 0

    def _propagate(self, idx: int, change: float):
        """Propagate priority change up to root."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, value: float) -> int:
        """Find leaf index by binary descent based on cumulative value."""
        left = 2 * idx + 1
        right = left + 1

        # Reached leaf
        if left >= len(self.tree):
            return idx

        if value <= self.tree[left]:
            return self._retrieve(left, value)
        else:
            return self._retrieve(right, value - self.tree[left])

    def total(self) -> float:
        return float(self.tree[0])

    def add(self, priority: float) -> int:
        """Add new entry with priority. Returns data index ที่เขียน (สำหรับ map ไป external storage)."""
        leaf_idx = self.write_ptr + self.capacity - 1
        self.update(leaf_idx, priority)

        data_idx = self.write_ptr
        self.write_ptr = (self.write_ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return data_idx

    def update(self, leaf_idx: int, priority: float):
        """Update priority at leaf index."""
        change = priority - self.tree[leaf_idx]
        self.tree[leaf_idx] = priority
        if leaf_idx != 0:
            self._propagate(leaf_idx, change)

    def get(self, value: float) -> Tuple[int, float, int]:
        """
        Sample leaf by cumulative priority value.
        
        Returns:
            leaf_idx: index in tree array
            priority: priority value at this leaf
            data_idx: index in external data storage
        """
        leaf_idx = self._retrieve(0, value)
        data_idx = leaf_idx - (self.capacity - 1)
        return leaf_idx, float(self.tree[leaf_idx]), data_idx


class PrioritizedReplayBuffer:
    """
    PER Buffer storing (state, action, reward, next_state, done) tuples.
    
    Args:
        capacity:    max number of transitions
        seq_len:     timesteps per state (60)
        n_features:  feature dim per timestep (16)
        n_heads:     number of categories (4) — for per-head action/reward
        alpha:       priority exponent (0.6 default)
        epsilon:     priority floor
        device:      torch device for sampled tensors
    """
    def __init__(
        self,
        capacity: int,
        seq_len: int,
        n_features: int,
        n_heads: int,
        alpha: float = 0.6,
        epsilon: float = 1e-6,
        device: str = "cpu",
    ):
        self.capacity = capacity
        self.seq_len = seq_len
        self.n_features = n_features
        self.n_heads = n_heads
        self.alpha = alpha
        self.epsilon = epsilon
        self.device = device

        self.tree = SumTree(capacity)

        # Pre-allocated storage (numpy arrays — เร็วกว่า list of tensors)
        self.states = np.zeros((capacity, seq_len, n_features), dtype=np.float32)
        self.next_states = np.zeros((capacity, seq_len, n_features), dtype=np.float32)
        self.actions = np.zeros((capacity, n_heads), dtype=np.int8)
        self.rewards = np.zeros((capacity, n_heads), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        # Track max priority สำหรับ new transitions (initialize ด้วย max priority)
        self.max_priority = 1.0

    def __len__(self) -> int:
        return self.tree.size

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_state: np.ndarray,
        done: float,
    ):
        """
        Add transition to buffer with max priority (สำหรับ guarantee ว่าจะถูก sample อย่างน้อย 1 ครั้ง).
        
        Args:
            state:      (T, F) numpy array
            action:     (N,) ∈ {0, 1} per-head action mask
            reward:     (N,) per-head reward OR scalar (will broadcast)
            next_state: (T, F) numpy array
            done:       0.0 or 1.0
        """
        # Broadcast scalar reward → per-head ถ้าจำเป็น
        if np.isscalar(reward):
            reward = np.full(self.n_heads, reward, dtype=np.float32)
        elif reward.ndim == 0:
            reward = np.full(self.n_heads, float(reward), dtype=np.float32)

        # Get next write index
        priority = (self.max_priority + self.epsilon) ** self.alpha
        data_idx = self.tree.add(priority)

        self.states[data_idx] = state
        self.next_states[data_idx] = next_state
        self.actions[data_idx] = action
        self.rewards[data_idx] = reward
        self.dones[data_idx] = done

    def sample(
        self, batch_size: int, beta: float = 0.4
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray,
    ]:
        """
        Sample batch with probability ∝ priority^alpha.
        
        Args:
            batch_size: number of samples
            beta:       importance sampling exponent (anneal toward 1.0)
        
        Returns:
            states:      (B, T, F) tensor
            actions:     (B, N) tensor
            rewards:     (B, N) tensor
            next_states: (B, T, F) tensor
            dones:       (B,) tensor
            is_weights:  (B,) importance sampling weights, normalized
            tree_idxs:   (B,) numpy array — สำหรับ update priorities ภายหลัง
        """
        assert len(self) >= batch_size, \
            f"Buffer size {len(self)} < batch_size {batch_size}"

        tree_idxs = np.empty(batch_size, dtype=np.int64)
        data_idxs = np.empty(batch_size, dtype=np.int64)
        priorities = np.empty(batch_size, dtype=np.float64)

        # Stratified sampling — แบ่ง [0, total] เป็น B segments แล้ว sample 1 ตัวต่อ segment
        # ให้ coverage ดี ลด variance
        total = self.tree.total()
        segment = total / batch_size

        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            value = np.random.uniform(lo, hi)
            tree_idx, prio, data_idx = self.tree.get(value)
            tree_idxs[i] = tree_idx
            data_idxs[i] = data_idx
            priorities[i] = prio

        # Importance sampling weights:
        #   w_i = (1/N · 1/P(i))^β  /  max_j w_j
        # P(i) = priority_i / total_priority
        sampling_probs = priorities / total
        is_weights = (len(self) * sampling_probs) ** (-beta)
        is_weights = is_weights / is_weights.max()  # normalize to [0, 1]

        # Convert to tensors
        states = torch.from_numpy(self.states[data_idxs]).to(self.device)
        actions = torch.from_numpy(self.actions[data_idxs]).long().to(self.device)
        rewards = torch.from_numpy(self.rewards[data_idxs]).to(self.device)
        next_states = torch.from_numpy(self.next_states[data_idxs]).to(self.device)
        dones = torch.from_numpy(self.dones[data_idxs]).to(self.device)
        is_weights_t = torch.from_numpy(is_weights.astype(np.float32)).to(self.device)

        return states, actions, rewards, next_states, dones, is_weights_t, tree_idxs

    def update_priorities(self, tree_idxs: np.ndarray, td_errors: np.ndarray):
        """
        Update priorities หลัง compute TD errors.
        
        Args:
            tree_idxs: (B,) tree indices ที่ sample()
            td_errors: (B,) TD error magnitudes (positive)
        """
        priorities = (np.abs(td_errors) + self.epsilon) ** self.alpha
        for tree_idx, prio in zip(tree_idxs, priorities):
            self.tree.update(int(tree_idx), float(prio))
            self.max_priority = max(self.max_priority, float(np.abs(td_errors).max()))


class UniformReplayBuffer:
    """
    Vanilla uniform replay buffer — fallback ถ้าไม่ใช้ PER
    
    เร็วกว่า PER เพราะไม่ต้อง maintain SumTree, แต่ sample efficiency ต่ำกว่า
    """
    def __init__(
        self,
        capacity: int,
        seq_len: int,
        n_features: int,
        n_heads: int,
        device: str = "cpu",
    ):
        self.capacity = capacity
        self.device = device
        self.size = 0
        self.write_ptr = 0

        self.states = np.zeros((capacity, seq_len, n_features), dtype=np.float32)
        self.next_states = np.zeros((capacity, seq_len, n_features), dtype=np.float32)
        self.actions = np.zeros((capacity, n_heads), dtype=np.int8)
        self.rewards = np.zeros((capacity, n_heads), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def __len__(self) -> int:
        return self.size

    def add(self, state, action, reward, next_state, done):
        if np.isscalar(reward):
            reward = np.full(self.actions.shape[1], reward, dtype=np.float32)

        idx = self.write_ptr
        self.states[idx] = state
        self.next_states[idx] = next_state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = done

        self.write_ptr = (self.write_ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float = 0.0):
        """beta argument ignored — kept for interface compatibility."""
        assert self.size >= batch_size
        idxs = np.random.randint(0, self.size, size=batch_size)

        states = torch.from_numpy(self.states[idxs]).to(self.device)
        actions = torch.from_numpy(self.actions[idxs]).long().to(self.device)
        rewards = torch.from_numpy(self.rewards[idxs]).to(self.device)
        next_states = torch.from_numpy(self.next_states[idxs]).to(self.device)
        dones = torch.from_numpy(self.dones[idxs]).to(self.device)
        # Uniform sampling → IS weights = 1
        is_weights = torch.ones(batch_size, device=self.device)

        return states, actions, rewards, next_states, dones, is_weights, idxs

    def update_priorities(self, idxs, td_errors):
        """No-op for uniform buffer — interface compatibility."""
        pass
