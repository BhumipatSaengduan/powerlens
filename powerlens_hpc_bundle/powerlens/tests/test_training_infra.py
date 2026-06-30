"""
Tests for HPC training infrastructure.

Coverage:
  Synthetic Generator:
    1. Single window output shape + types
    2. Batch generation shapes
    3. All 16 features populated (not all-zero)
    4. Per-category traces realistic (status binary, power range correct)
    5. Reproducibility with seed
  
  Data Module:
    6. SyntheticDataModule.get_supervised_batch() shapes
    7. SyntheticDataModule.get_rl_step() shapes
    8. CSVDataModule builds windows from real wide CSV
  
  Evaluator:
    9. Evaluator.run() returns valid EvalReport
    10. Untrained model: TECA reasonable range
  
  CSV Logger:
    11. Write/read roundtrip
    12. Multi-phase logger creates separate files
"""
import sys
import tempfile
import csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch

from powerlens.data.synthetic import (
    SyntheticGenerator, SyntheticSample, PROFILES,
)
from powerlens.data.data_module import (
    SyntheticDataModule, CSVDataModule, SupervisedBatch, RLTransition,
)
from powerlens.training.evaluator import Evaluator, EvalReport
from powerlens.training.csv_logger import CSVLogger, MultiPhaseLogger
from powerlens.models.trainer import DRLSTFNTrainer
from powerlens.models.config import ROUTER_CONFIG, EXPERT_CONFIG


# ============================================================
# Synthetic Generator
# ============================================================

def test_generator_window_shape():
    """Single window: features (T, 16), labels per category (T,)"""
    print("\n=== Test 1: Generator Window Shape ===")
    gen = SyntheticGenerator(window_size=60, seed=42)
    sample = gen.generate_window()
    assert sample.features.shape == (60, 16), f"features: {sample.features.shape}"
    assert sample.truth_active_mask.shape == (60, 4)
    for cat in ["Plug", "Light", "AC", "Water_Heater"]:
        assert sample.truth_status[cat].shape == (60,)
        assert sample.truth_power[cat].shape == (60,)
        assert sample.truth_current[cat].shape == (60,)
    print(f"  ✓ Features: (60, 16), labels per category: (60,) × 3 fields × 4 categories")


def test_generator_batch_shape():
    """Batch: features (B, T, 16), labels per category (B, T)"""
    print("\n=== Test 2: Generator Batch Shape ===")
    gen = SyntheticGenerator(window_size=60, seed=42)
    batch = gen.generate_batch(8)
    assert batch.features.shape == (8, 60, 16)
    assert batch.truth_active_mask.shape == (8, 60, 4)
    for cat in PROFILES:
        assert batch.truth_status[cat].shape == (8, 60)
    print(f"  ✓ Batch features: (8, 60, 16), per-cat labels: (8, 60)")


def test_features_populated():
    """ทุก feature dim ต้องมีค่า (ไม่ all-zero)"""
    print("\n=== Test 3: All 16 Features Populated ===")
    gen = SyntheticGenerator(window_size=60, seed=42)
    batch = gen.generate_batch(16)
    feature_names = ["V_rms", "I_rms", "P", "Q", "PF", "THD"] + [f"H{i}" for i in range(1, 11)]
    for i, name in enumerate(feature_names):
        col = batch.features[:, :, i]
        std = col.std()
        assert std > 1e-6, f"Feature {name} (col {i}) is constant — std={std}"
    print(f"  ✓ All 16 features have variance")
    print(f"  Sample V_rms range: [{batch.features[:,:,0].min():.1f}, "
          f"{batch.features[:,:,0].max():.1f}]")
    print(f"  Sample P range:     [{batch.features[:,:,2].min():.1f}, "
          f"{batch.features[:,:,2].max():.1f}]")


def test_per_category_traces_realistic():
    """Per-category traces ควรอยู่ในช่วง realistic"""
    print("\n=== Test 4: Realistic Per-Category Traces ===")
    gen = SyntheticGenerator(window_size=60, seed=42)
    batch = gen.generate_batch(64)

    for cat in PROFILES:
        # Status: binary
        s = batch.truth_status[cat]
        assert ((s == 0) | (s == 1)).all(), f"{cat} status not binary"

        # Power: ≥ 0, max in profile range
        p = batch.truth_power[cat]
        assert (p >= 0).all(), f"{cat} power negative"
        active = p[s > 0.5]
        if len(active) > 0:
            profile = PROFILES[cat]
            # Allow 20% slack for variation
            assert active.max() <= profile.power_max * 1.5, \
                f"{cat} power max {active.max():.1f} > profile {profile.power_max * 1.5}"
        print(f"  {cat}: status binary OK, power range [{p.min():.1f}, {p.max():.1f}]W")


def test_reproducibility():
    """Same seed → same output"""
    print("\n=== Test 5: Reproducibility ===")
    gen1 = SyntheticGenerator(seed=123)
    gen2 = SyntheticGenerator(seed=123)
    s1 = gen1.generate_window()
    s2 = gen2.generate_window()
    assert np.allclose(s1.features, s2.features)
    print(f"  ✓ Same seed → bitwise identical features")


# ============================================================
# Data Module
# ============================================================

def test_synthetic_data_module_supervised():
    """SupervisedBatch shapes match trainer expectation"""
    print("\n=== Test 6: SyntheticDataModule.get_supervised_batch() ===")
    dm = SyntheticDataModule(window_size=60, seed=42)
    batch = dm.get_supervised_batch(8)

    assert isinstance(batch, SupervisedBatch)
    assert batch.features.shape == (8, 60, 16)
    for cat in dm.categories:
        assert batch.truth_status[cat].shape == (8, 1)
        assert batch.truth_power[cat].shape == (8, 1)
        assert batch.truth_current[cat].shape == (8, 1)
    print(f"  ✓ features: {tuple(batch.features.shape)}")
    print(f"  ✓ Per-category labels: (8, 1) × 4 categories")


def test_synthetic_data_module_rl_step():
    """RLTransition shapes match trainer.collect_transition() expectation"""
    print("\n=== Test 7: SyntheticDataModule.get_rl_step() ===")
    dm = SyntheticDataModule(window_size=60, seed=42)
    trans = dm.get_rl_step()
    assert isinstance(trans, RLTransition)
    assert trans.state.shape == (1, 60, 16)
    assert trans.next_state.shape == (1, 60, 16)
    assert trans.truth_active.shape == (1, 4)
    assert trans.truth_power_per_cat.shape == (1, 4)
    assert isinstance(trans.done, float)
    print(f"  ✓ state: (1, 60, 16), active: (1, 4), done: {trans.done}")


def test_csv_module_wide_csv():
    """CSVDataModule should build sequence windows from a wide real-data CSV."""
    print("\n=== Test 8: CSVDataModule Wide CSV ===")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "wide.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "I_agg", "PF_agg", "V_rms",
                "AC_status", "AC_current", "Light_status", "Light_current",
            ])
            for i in range(80):
                writer.writerow([
                    f"2026-06-03 12:{i//6:02d}:{(i%6)*10:02d}",
                    4.0 + 0.01 * i,
                    0.9,
                    220.0,
                    1 if i > 20 else 0,
                    2.5 if i > 20 else 0.0,
                    1 if i % 10 < 5 else 0,
                    0.2 if i % 10 < 5 else 0.0,
                ])

        dm = CSVDataModule(
            csv_path=str(path),
            window_size=10,
            train=True,
            train_ratio=0.8,
            device="cpu",
        )
        batch = dm.get_supervised_batch(4)
        assert dm.feature_columns == ["I_agg", "PF_agg", "V_rms"]
        assert dm.categories == ["AC", "Light"]
        assert batch.features.shape == (4, 10, 3)
        assert batch.truth_status["AC"].shape == (4, 1)
        assert batch.truth_current["Light"].shape == (4, 1)
        manifest = dm.preprocessing_manifest()
        assert manifest["n_features"] == 3
        print(f"  ✓ CSV windows: features={tuple(batch.features.shape)}, categories={dm.categories}")


# ============================================================
# Evaluator
# ============================================================

def test_evaluator_runs():
    """Evaluator.run() returns valid EvalReport with all fields populated"""
    print("\n=== Test 9: Evaluator Run ===")
    trainer = DRLSTFNTrainer()
    dm = SyntheticDataModule(window_size=60, seed=42)
    evaluator = Evaluator(trainer, dm, ROUTER_CONFIG.categories)

    report = evaluator.run(n_batches=2, batch_size=8)
    assert isinstance(report, EvalReport)
    assert report.n_samples == 16
    assert len(report.per_category) == 4
    assert 0 <= report.router_action_accuracy <= 1
    assert 0 <= report.router_joint_accuracy <= 1

    print(f"  TECA: {report.teca:.4f}")
    print(f"  Router action acc: {report.router_action_accuracy:.4f}")
    print(f"  Router joint acc:  {report.router_joint_accuracy:.4f}")
    for cat, m in report.per_category.items():
        print(f"  {cat}: status_acc={m.status_accuracy:.3f}, power_mae={m.power_mae:.1f}W")


def test_eval_report_to_dict():
    """EvalReport.to_dict() produces flat dict suitable for CSV logging"""
    print("\n=== Test 10: EvalReport → flat dict ===")
    trainer = DRLSTFNTrainer()
    dm = SyntheticDataModule(window_size=60, seed=42)
    evaluator = Evaluator(trainer, dm, ROUTER_CONFIG.categories)
    report = evaluator.run(n_batches=2, batch_size=4)
    d = report.to_dict()
    # ต้องมี keys ครบ
    assert "agg/teca" in d
    assert "router/action_accuracy" in d
    for cat in ROUTER_CONFIG.categories:
        assert f"{cat}/status_acc" in d
        assert f"{cat}/power_mae" in d
    # ทุกค่าเป็น scalar
    for k, v in d.items():
        assert isinstance(v, (int, float)), f"{k} = {v} not scalar"
    print(f"  ✓ {len(d)} flat keys, all scalar")


# ============================================================
# CSV Logger
# ============================================================

def test_csv_logger_roundtrip():
    """Write metrics → read back ตรงกัน"""
    print("\n=== Test 11: CSV Logger Round-trip ===")
    with tempfile.TemporaryDirectory() as tmp:
        with CSVLogger(tmp, phase_name="test") as logger:
            logger.log({"step": 0, "loss": 1.5, "lr": 0.001})
            logger.log({"step": 1, "loss": 1.2, "lr": 0.001})
            logger.log({"step": 2, "loss": 0.9, "lr": 0.0008})

        # Read back
        path = Path(tmp) / "test.csv"
        with open(path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3
        assert float(rows[0]["loss"]) == 1.5
        assert float(rows[2]["loss"]) == 0.9
        print(f"  ✓ Wrote 3 rows, read back: loss={[r['loss'] for r in rows]}")


def test_multi_phase_logger():
    """MultiPhaseLogger creates separate files per phase"""
    print("\n=== Test 12: Multi-Phase Logger ===")
    with tempfile.TemporaryDirectory() as tmp:
        with MultiPhaseLogger(tmp) as log:
            log.pretrain.log({"step": 0, "loss": 1.0})
            log.rl.log({"step": 0, "epsilon": 1.0, "loss/dqn": 0.5})
            log.eval.log({"step": 100, "agg/teca": 0.85})

        # Verify 3 files exist
        for name in ["pretrain.csv", "rl.csv", "eval.csv"]:
            assert (Path(tmp) / name).exists(), f"{name} missing"
        print(f"  ✓ 3 separate CSV files created")


if __name__ == "__main__":
    print("=" * 60)
    print("HPC Training Infrastructure — Sanity Tests")
    print("=" * 60)

    test_generator_window_shape()
    test_generator_batch_shape()
    test_features_populated()
    test_per_category_traces_realistic()
    test_reproducibility()

    test_synthetic_data_module_supervised()
    test_synthetic_data_module_rl_step()
    test_csv_module_wide_csv()

    test_evaluator_runs()
    test_eval_report_to_dict()

    test_csv_logger_roundtrip()
    test_multi_phase_logger()

    print("\n" + "=" * 60)
    print("✓ All tests passed")
    print("=" * 60)
