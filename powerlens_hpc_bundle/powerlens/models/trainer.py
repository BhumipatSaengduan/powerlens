"""
DRL-STFN Joint Trainer
=======================
Training orchestrator สำหรับ DRL-STFN — รวม Router (DQN) + Experts (supervised)

Curriculum strategy (3 phases):
    Phase 1 (Pretrain Experts):
        - Freeze Router (random policy or skip)
        - Train Experts ด้วย supervised loss + ground-truth labels
        - Goal: Experts ต้องแม่นพอก่อนใช้ feedback ให้ Router
    
    Phase 2 (Train Router):
        - Freeze Experts
        - Collect experience: state → router action → expert prediction → reward
        - Update Router via DQN loss (per-head reward + PER)
        - Goal: Router เรียนรู้ว่าจะส่ง input ไป expert ไหน
    
    Phase 3 (Joint Fine-tune):
        - Unfreeze ทั้งหมด
        - Train ร่วมกัน — Router policy ดีขึ้น → Expert ได้ข้อมูล routed ที่ดีขึ้น
        - Online adaptive learning หลังจาก deploy

Note:
    File นี้เป็น scaffolding — ตัว execution loop แท้ๆ จะเรียกจาก train script
    แยกข้างนอก (จะเขียน module ถัดไป)
"""
from copy import deepcopy
from dataclasses import asdict
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
from torch.optim import Adam

from .router import BranchingDQNRouter, compute_dqn_loss
from .expert import MultiCategoryExperts
from .replay_buffer import PrioritizedReplayBuffer, UniformReplayBuffer
from .reward import compute_per_head_reward, compute_expert_supervised_loss
from .config import (
    RouterConfig, ExpertConfig, RLConfig, TrainConfig, RewardConfig,
    ROUTER_CONFIG, EXPERT_CONFIG, RL_CONFIG, TRAIN_CONFIG, REWARD_CONFIG,
)


class DRLSTFNTrainer:
    """
    Main trainer orchestrating Router + Experts learning.
    
    Public API:
        - pretrain_expert_step(batch)   → Phase 1
        - rl_step()                     → Phase 2 (sample + update)
        - collect_transition(...)       → Phase 2 (add to replay buffer)
        - joint_step(...)               → Phase 3
        - soft_update_target() / hard_update_target()
    """
    def __init__(
        self,
        router_config: RouterConfig = ROUTER_CONFIG,
        expert_config: ExpertConfig = EXPERT_CONFIG,
        rl_config: RLConfig = RL_CONFIG,
        train_config: TrainConfig = TRAIN_CONFIG,
        reward_config: RewardConfig = REWARD_CONFIG,
        device: str = "cpu",
    ):
        self.router_cfg = router_config
        self.expert_cfg = expert_config
        self.rl_cfg = rl_config
        self.train_cfg = train_config
        self.reward_cfg = reward_config
        self.device = device

        # Models
        self.router = BranchingDQNRouter(router_config).to(device)
        self.target_router = deepcopy(self.router).to(device)
        self.target_router.eval()
        for p in self.target_router.parameters():
            p.requires_grad = False

        self.experts = MultiCategoryExperts(expert_config).to(device)

        # Optimizers (separate — Router and Experts ใช้ lr ต่างกันได้)
        self.router_opt = Adam(
            self.router.parameters(),
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
        )
        self.expert_opt = Adam(
            self.experts.parameters(),
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
        )

        # Replay buffer (PER or uniform)
        if rl_config.use_prioritized_replay:
            self.buffer = PrioritizedReplayBuffer(
                capacity=rl_config.replay_buffer_size,
                seq_len=router_config.seq_len,
                n_features=router_config.n_features,
                n_heads=router_config.n_heads,
                alpha=rl_config.per_alpha,
                epsilon=rl_config.per_epsilon,
                device=device,
            )
        else:
            self.buffer = UniformReplayBuffer(
                capacity=rl_config.replay_buffer_size,
                seq_len=router_config.seq_len,
                n_features=router_config.n_features,
                n_heads=router_config.n_heads,
                device=device,
            )

        # Step counter
        self.global_step = 0

    # ----------------------------------------------------------------
    # Exploration / scheduling
    # ----------------------------------------------------------------
    def get_epsilon(self) -> float:
        """ε-greedy schedule — linear decay."""
        cfg = self.rl_cfg
        progress = min(self.global_step / cfg.epsilon_decay_steps, 1.0)
        return cfg.epsilon_start + (cfg.epsilon_end - cfg.epsilon_start) * progress

    def get_beta(self) -> float:
        """PER β anneal — linear toward 1.0."""
        cfg = self.rl_cfg
        progress = min(self.global_step / cfg.per_beta_anneal_steps, 1.0)
        return cfg.per_beta_start + (cfg.per_beta_end - cfg.per_beta_start) * progress

    # ----------------------------------------------------------------
    # Phase 1: Expert pretraining (supervised)
    # ----------------------------------------------------------------
    def pretrain_expert_step(
        self,
        states: torch.Tensor,
        truth_status: Dict[str, torch.Tensor],
        truth_power: Dict[str, torch.Tensor],
        truth_current: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Single supervised training step for all Experts.
        
        Args:
            states:         (B, T, F)
            truth_*:        dicts mapping category → (B, 1) ground truth
        Returns:
            metrics: dict with total_loss + per-category breakdown
        """
        self.experts.train()
        outputs = self.experts.forward_all(states)

        loss, breakdown = compute_expert_supervised_loss(
            outputs,
            truth_status, truth_power, truth_current,
            loss_weight_status=self.train_cfg.loss_weight_status,
            loss_weight_power=self.train_cfg.loss_weight_power,
            loss_weight_current=self.train_cfg.loss_weight_current,
        )

        self.expert_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.experts.parameters(), max_norm=1.0)
        self.expert_opt.step()

        return {"loss/total": loss.item(), **{f"loss/{k}": v for k, v in breakdown.items()}}

    # ----------------------------------------------------------------
    # Phase 2: Router DRL training
    # ----------------------------------------------------------------
    @torch.no_grad()
    def collect_transition(
        self,
        state: torch.Tensor,
        truth_active: torch.Tensor,
        truth_power: torch.Tensor,
        next_state: torch.Tensor,
        done: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run online: select action via ε-greedy → run experts → compute reward → store
        
        Args:
            state:          (1, T, F) — single transition
            truth_active:   (1, N) — ground truth which categories are on
            truth_power:    (1, N) — ground truth power per category
            next_state:     (1, T, F)
            done:           0.0 or 1.0
        Returns:
            action_mask:    (1, N) action chosen
            reward:         (1, N) per-head reward
        """
        self.router.eval()
        self.experts.eval()

        epsilon = self.get_epsilon()
        action_mask, _ = self.router.select_action(state, epsilon=epsilon)

        # Run only selected experts → predicted power per category
        # Categories ที่ไม่ถูกเลือก → predicted = 0 (consistent with reward computation)
        n_heads = self.router_cfg.n_heads
        expert_power = torch.zeros(1, n_heads, device=self.device)
        for i, cat in enumerate(self.router_cfg.categories):
            if action_mask[0, i].item() == 1:
                _, p_pred, _ = self.experts(state, category=cat)
                expert_power[0, i] = p_pred.squeeze()

        # Compute per-head reward
        reward = compute_per_head_reward(
            action_mask=action_mask,
            truth_active=truth_active,
            expert_power_preds=expert_power,
            truth_power=truth_power,
            config=self.reward_cfg,
        )

        # Store in buffer (numpy)
        self.buffer.add(
            state=state.squeeze(0).cpu().numpy(),
            action=action_mask.squeeze(0).cpu().numpy(),
            reward=reward.squeeze(0).cpu().numpy(),
            next_state=next_state.squeeze(0).cpu().numpy(),
            done=done,
        )

        return action_mask, reward

    def rl_step(self, batch_size: int = 64) -> Optional[Dict[str, float]]:
        """
        One DQN training step — sample batch, compute loss, update Router.
        
        Returns None ถ้า buffer ยังไม่พอ
        """
        if len(self.buffer) < max(self.rl_cfg.min_replay_size, batch_size):
            return None

        beta = self.get_beta()
        states, actions, rewards, next_states, dones, is_weights, idxs = \
            self.buffer.sample(batch_size, beta=beta)

        self.router.train()
        loss, td_errors = compute_dqn_loss(
            online_net=self.router,
            target_net=self.target_router,
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
            gamma=self.rl_cfg.gamma,
            use_double_dqn=self.rl_cfg.use_double_dqn,
            is_weights=is_weights,
        )

        self.router_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.router.parameters(), max_norm=1.0)
        self.router_opt.step()

        # Update PER priorities
        self.buffer.update_priorities(idxs, td_errors.cpu().numpy())

        # Periodic target network update
        self.global_step += 1
        if self.global_step % self.rl_cfg.target_update_freq == 0:
            self.hard_update_target()

        return {
            "loss/dqn": loss.item(),
            "rl/epsilon": self.get_epsilon(),
            "rl/beta": beta,
            "rl/buffer_size": len(self.buffer),
            "rl/td_error_mean": float(td_errors.mean()),
            "rl/reward_mean": float(rewards.mean()),
        }

    # ----------------------------------------------------------------
    # Target network management
    # ----------------------------------------------------------------
    def hard_update_target(self):
        """Copy online → target (used periodically)."""
        self.target_router.load_state_dict(self.router.state_dict())

    def soft_update_target(self, tau: float = 0.005):
        """Polyak averaging — alternative to hard update."""
        for tgt_p, online_p in zip(
            self.target_router.parameters(), self.router.parameters()
        ):
            tgt_p.data.mul_(1.0 - tau).add_(online_p.data, alpha=tau)

    # ----------------------------------------------------------------
    # Save / load
    # ----------------------------------------------------------------
    def save_checkpoint(self, path: str):
        """Save full state (model + optimizers + step counter)."""
        feature_config = getattr(self, "feature_config", None)
        torch.save({
            "router": self.router.state_dict(),
            "target_router": self.target_router.state_dict(),
            "experts": self.experts.state_dict(),
            "router_opt": self.router_opt.state_dict(),
            "expert_opt": self.expert_opt.state_dict(),
            "global_step": self.global_step,
            "router_config": asdict(self.router_cfg),
            "expert_config": asdict(self.expert_cfg),
            "rl_config": asdict(self.rl_cfg),
            "train_config": asdict(self.train_cfg),
            "reward_config": asdict(self.reward_cfg),
            "feature_config": asdict(feature_config) if feature_config is not None else None,
        }, path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.router.load_state_dict(ckpt["router"])
        self.target_router.load_state_dict(ckpt["target_router"])
        self.experts.load_state_dict(ckpt["experts"])
        self.router_opt.load_state_dict(ckpt["router_opt"])
        self.expert_opt.load_state_dict(ckpt["expert_opt"])
        self.global_step = ckpt["global_step"]
