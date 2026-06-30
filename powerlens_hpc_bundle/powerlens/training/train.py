"""
Training Script — CLI runner for DRL-STFN
==========================================
Orchestrates 3-phase curriculum training and saves checkpoints + logs.

Usage:
    # Run full curriculum (default settings)
    python -m powerlens.training.train --output-dir runs/exp1

    # Override phases / steps
    python -m powerlens.training.train \\
        --output-dir runs/exp2 \\
        --pretrain-steps 1000 \\
        --rl-steps 5000 \\
        --eval-every 100 \\
        --device cuda

    # Skip phases (for partial reruns)
    python -m powerlens.training.train --skip-pretrain --output-dir runs/exp3

Output:
    {output_dir}/
    ├── pretrain.csv      ← Phase 1 metrics
    ├── rl.csv            ← Phase 2 metrics
    ├── eval.csv          ← Validation metrics (across phases)
    ├── checkpoints/
    │   ├── pretrain_latest.pt
    │   ├── rl_latest.pt
    │   └── best.pt       ← best by validation TECA
    └── config.json       ← Snapshot ของ config ใช้
"""
import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch

# Project imports — works both as module (-m) and direct script (python train.py)
import sys
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from powerlens.models.trainer import DRLSTFNTrainer
from powerlens.models.config import (
    FeatureConfig,
    ROUTER_CONFIG, EXPERT_CONFIG, RL_CONFIG, TRAIN_CONFIG, REWARD_CONFIG,
)
from powerlens.data.data_module import CSVDataModule, SyntheticDataModule
from powerlens.training.evaluator import Evaluator
from powerlens.training.csv_logger import MultiPhaseLogger


def _csv_list(value: str):
    """Parse comma-separated CLI values into a clean list."""
    if value is None:
        return None
    items = [v.strip() for v in value.split(",")]
    return [v for v in items if v]


def build_data_modules(args, device: str):
    """Create train/validation data modules from synthetic or real CSV data."""
    if args.data_csv:
        categories = _csv_list(args.categories)
        feature_cols = _csv_list(args.feature_cols)
        train_module = CSVDataModule(
            csv_path=args.data_csv,
            categories=categories,
            feature_columns=feature_cols,
            timestamp_col=args.timestamp_col,
            status_suffix=args.status_suffix,
            power_suffix=args.power_suffix,
            current_suffix=args.current_suffix,
            window_size=args.seq_len,
            train=True,
            train_ratio=args.train_ratio,
            scale=not args.no_scale,
            seed=args.seed,
            device=device,
        )
        val_module = CSVDataModule(
            csv_path=args.data_csv,
            categories=train_module.categories,
            feature_columns=train_module.feature_columns,
            timestamp_col=args.timestamp_col,
            status_suffix=args.status_suffix,
            power_suffix=args.power_suffix,
            current_suffix=args.current_suffix,
            window_size=args.seq_len,
            train=False,
            train_ratio=args.train_ratio,
            scale=not args.no_scale,
            scaler_state=train_module.scaler_state,
            seed=args.seed + 999,
            device=device,
        )
        return train_module, val_module

    data_module = SyntheticDataModule(
        window_size=args.seq_len,
        seed=args.seed,
        device=device,
    )
    val_module = SyntheticDataModule(
        window_size=args.seq_len,
        seed=args.seed + 999,
        device=device,
    )
    return data_module, val_module


# ============================================================
# Phase functions
# ============================================================

def run_pretrain(trainer, data_module, logger, evaluator,
                 n_steps: int, batch_size: int, eval_every: int,
                 ckpt_dir: Path):
    """Phase 1: Supervised expert pretraining"""
    print(f"\n{'='*60}\nPhase 1: Pretrain Experts ({n_steps} steps)\n{'='*60}")
    t_start = time.time()
    best_teca = -float('inf')

    for step in range(n_steps):
        batch = data_module.get_supervised_batch(batch_size)
        metrics = trainer.pretrain_expert_step(
            states=batch.features,
            truth_status=batch.truth_status,
            truth_power=batch.truth_power,
            truth_current=batch.truth_current,
        )
        metrics["step"] = step
        logger.pretrain.log(metrics)

        if step % 50 == 0:
            print(f"  [pretrain {step:5d}/{n_steps}] loss={metrics['loss/total']:.4f}")

        # Periodic eval
        if eval_every > 0 and step > 0 and step % eval_every == 0:
            report = evaluator.run(n_batches=5, batch_size=batch_size)
            eval_metrics = {"step": step, "phase": "pretrain", **report.to_dict()}
            logger.eval.log(eval_metrics)
            print(f"  [eval @ {step}] TECA={report.teca:.4f}, "
                  f"router_acc={report.router_action_accuracy:.4f}")

            if report.teca > best_teca:
                best_teca = report.teca
                trainer.save_checkpoint(str(ckpt_dir / "best.pt"))

    # Save end-of-phase checkpoint
    trainer.save_checkpoint(str(ckpt_dir / "pretrain_latest.pt"))
    print(f"  ✓ Pretrain done in {time.time()-t_start:.1f}s, best TECA={best_teca:.4f}")


def run_rl(trainer, data_module, logger, evaluator,
           n_steps: int, batch_size: int, eval_every: int,
           ckpt_dir: Path):
    """Phase 2: RL Router training (with frozen experts)"""
    print(f"\n{'='*60}\nPhase 2: Train Router via DQN ({n_steps} steps)\n{'='*60}")
    t_start = time.time()
    best_teca = -float('inf')

    # Try loading current best checkpoint as starting point
    best_path = ckpt_dir / "best.pt"
    if best_path.exists():
        try:
            current_best = trainer.global_step
            # Re-load best — already in trainer
        except Exception:
            pass

    # Freeze experts during RL phase
    for p in trainer.experts.parameters():
        p.requires_grad = False

    for step in range(n_steps):
        # Collect transition
        transition = data_module.get_rl_step()
        trainer.collect_transition(
            state=transition.state,
            truth_active=transition.truth_active,
            truth_power=transition.truth_power_per_cat,
            next_state=transition.next_state,
            done=transition.done,
        )

        # RL update step
        metrics = trainer.rl_step(batch_size=batch_size)
        if metrics is not None:
            metrics["step"] = step
            logger.rl.log(metrics)

            if step % 100 == 0:
                print(f"  [rl {step:5d}/{n_steps}] loss/dqn={metrics['loss/dqn']:.4f}, "
                      f"eps={metrics['rl/epsilon']:.3f}, "
                      f"buf={metrics['rl/buffer_size']}")

        # Periodic eval
        if eval_every > 0 and step > 0 and step % eval_every == 0:
            report = evaluator.run(n_batches=5, batch_size=batch_size)
            eval_metrics = {"step": step, "phase": "rl", **report.to_dict()}
            logger.eval.log(eval_metrics)
            print(f"  [eval @ {step}] TECA={report.teca:.4f}, "
                  f"router_acc={report.router_action_accuracy:.4f}")

            if report.teca > best_teca:
                best_teca = report.teca
                trainer.save_checkpoint(str(ckpt_dir / "best.pt"))

    # Restore experts trainable for joint phase (if any)
    for p in trainer.experts.parameters():
        p.requires_grad = True

    trainer.save_checkpoint(str(ckpt_dir / "rl_latest.pt"))
    print(f"  ✓ RL done in {time.time()-t_start:.1f}s, best TECA={best_teca:.4f}")


# ============================================================
# Main entrypoint
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="DRL-STFN Training")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for logs + checkpoints")
    parser.add_argument("--data-csv", type=str, default=None,
                        help="Wide CSV dataset. If omitted, synthetic data is used.")
    parser.add_argument("--feature-cols", type=str, default=None,
                        help="Comma-separated feature columns. Default: all numeric non-target columns.")
    parser.add_argument("--categories", type=str, default=None,
                        help="Comma-separated target categories. Default: infer from <category>_status columns.")
    parser.add_argument("--timestamp-col", type=str, default="timestamp")
    parser.add_argument("--status-suffix", type=str, default="_status")
    parser.add_argument("--power-suffix", type=str, default="_power")
    parser.add_argument("--current-suffix", type=str, default="_current")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seq-len", type=int, default=60,
                        help="Sequence window length.")
    parser.add_argument("--no-scale", action="store_true",
                        help="Disable robust feature scaling for CSV data.")
    parser.add_argument("--pretrain-steps", type=int, default=500)
    parser.add_argument("--rl-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=TRAIN_CONFIG.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=TRAIN_CONFIG.weight_decay)
    parser.add_argument("--loss-weight-status", type=float, default=TRAIN_CONFIG.loss_weight_status)
    parser.add_argument("--loss-weight-power", type=float, default=TRAIN_CONFIG.loss_weight_power)
    parser.add_argument("--loss-weight-current", type=float, default=TRAIN_CONFIG.loss_weight_current)
    parser.add_argument("--eval-every", type=int, default=100,
                        help="Run validation every N steps (0 = disable)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="cpu / cuda / cuda:0 etc")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-pretrain", action="store_true")
    parser.add_argument("--skip-rl", action="store_true")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Checkpoint path to resume from")
    args = parser.parse_args()

    # Setup output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    # Seed
    torch.manual_seed(args.seed)

    # Device check
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print(f"⚠️  CUDA requested but not available, falling back to CPU")
        device = "cpu"

    print(f"DRL-STFN Training")
    print(f"  Output:    {output_dir}")
    print(f"  Device:    {device}")
    print(f"  Seed:      {args.seed}")
    print(f"  Pretrain:  {args.pretrain_steps} steps")
    print(f"  RL:        {args.rl_steps} steps")

    # Build data modules first because real CSV controls features/categories.
    data_module, val_module = build_data_modules(args, device=device)
    feature_config = FeatureConfig(
        feature_names=list(data_module.feature_columns),
        source="csv_all_numeric" if args.data_csv else "synthetic_16_default",
    )
    router_config = replace(
        ROUTER_CONFIG,
        seq_len=args.seq_len,
        n_features=feature_config.n_features,
        categories=list(data_module.categories),
    )
    expert_config = replace(
        EXPERT_CONFIG,
        seq_len=args.seq_len,
        n_features=feature_config.n_features,
        categories=list(data_module.categories),
    )
    train_config = replace(
        TRAIN_CONFIG,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        loss_weight_status=args.loss_weight_status,
        loss_weight_power=args.loss_weight_power,
        loss_weight_current=args.loss_weight_current,
    )

    print(f"  Data:      {'CSV' if args.data_csv else 'synthetic'}")
    print(f"  Features:  {feature_config.n_features}")
    print(f"  Categories:{len(data_module.categories)}")
    if args.data_csv and getattr(data_module, "missing_power_targets", []):
        print("  Note:      missing power targets filled with 0 for "
              f"{data_module.missing_power_targets}; consider --loss-weight-power 0")

    # Snapshot dynamic config after data schema is known.
    preprocessing = (
        data_module.preprocessing_manifest()
        if hasattr(data_module, "preprocessing_manifest")
        else {
            "format": "synthetic",
            "feature_columns": data_module.feature_columns,
            "categories": data_module.categories,
            "window_size": args.seq_len,
        }
    )
    config_snapshot = {
        "args": vars(args),
        "feature_config": asdict(feature_config),
        "router_config": asdict(router_config),
        "expert_config": asdict(expert_config),
        "rl_config": asdict(RL_CONFIG),
        "train_config": asdict(train_config),
        "reward_config": asdict(REWARD_CONFIG),
        "preprocessing": preprocessing,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_snapshot, f, indent=2)
    with open(output_dir / "preprocessing_manifest.json", "w") as f:
        json.dump(preprocessing, f, indent=2)

    # Build trainer + evaluator
    trainer = DRLSTFNTrainer(
        router_config=router_config,
        expert_config=expert_config,
        train_config=train_config,
        device=device,
    )
    trainer.feature_config = feature_config
    if args.resume_from:
        print(f"  Resuming from {args.resume_from}")
        trainer.load_checkpoint(args.resume_from)

    evaluator = Evaluator(trainer, val_module, router_config.categories, device=device)

    with MultiPhaseLogger(str(output_dir)) as logger:
        # Phase 1
        if not args.skip_pretrain:
            run_pretrain(
                trainer, data_module, logger, evaluator,
                n_steps=args.pretrain_steps,
                batch_size=args.batch_size,
                eval_every=args.eval_every,
                ckpt_dir=ckpt_dir,
            )

        # Phase 2
        if not args.skip_rl:
            run_rl(
                trainer, data_module, logger, evaluator,
                n_steps=args.rl_steps,
                batch_size=args.batch_size,
                eval_every=args.eval_every,
                ckpt_dir=ckpt_dir,
            )

    # Final eval
    print(f"\n{'='*60}\nFinal Evaluation\n{'='*60}")
    report = evaluator.run(n_batches=20, batch_size=args.batch_size)
    print(json.dumps(report.to_dict(), indent=2))

    print(f"\n✓ Training complete. Outputs in {output_dir}")


if __name__ == "__main__":
    main()
